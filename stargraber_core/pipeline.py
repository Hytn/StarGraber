"""
Pipeline Orchestrator
Connects all 7 layers into an end-to-end workflow:

  Data Layer → Research Layer → Implementation Layer → Experiment Layer
       ↑                                                     ↓
  Review Layer ← Execution Layer ← Decision Layer ←──────────┘
"""

import os
import logging
from dataclasses import asdict

from .models import JsonStore
from .data_layer import DataLayer
from .research_layer import ResearchLayer
from .implementation_layer import ImplementationLayer
from .experiment_layer import ExperimentLayer
from .decision_layer import DecisionLayer
from .execution_layer import ExecutionLayer
from .review_layer import ReviewLayer

logger = logging.getLogger(__name__)


class StarGraberPipeline:
    """
    Full pipeline orchestrator.

    Runs the complete cycle:
    1. Data Layer:           Collect data + seed knowledge base
    2. Research Layer:       Generate factor ideas from knowledge
    3. Implementation Layer: Translate ideas to code + validate
    4. Experiment Layer:     Backtest validated factors
    5. Decision Layer:       Select factors + construct portfolio
    6. Execution Layer:      Simulate trading
    7. Review Layer:         Analyze results + feed back insights
    """

    def __init__(self, workspace_dir: str = "./workspace"):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.store = JsonStore(self.workspace_dir)

        # Initialize all layers
        self.data_layer = DataLayer(self.workspace_dir)
        self.research_layer = ResearchLayer(self.store)
        self.implementation_layer = ImplementationLayer(self.store)
        self.experiment_layer = ExperimentLayer(self.store)
        self.decision_layer = DecisionLayer(
            self.implementation_layer.code_library
        )
        self.execution_layer = ExecutionLayer()
        self.review_layer = ReviewLayer(self.store, self.data_layer)

    def run(self, use_llm: bool = False, llm_client=None) -> dict:
        """
        Execute the full pipeline end-to-end.

        Args:
            use_llm: If True, uses LLM for idea generation and code generation
            llm_client: LLM client instance (required if use_llm=True)

        Returns:
            dict with results from each layer
        """
        logger.info("\n" + "=" * 60)
        logger.info("  STARGRABER - AI-Driven Quant Research Pipeline")
        logger.info("  Running full pipeline cycle")
        logger.info("=" * 60 + "\n")

        results = {}

        # ─── LAYER 1: DATA ───────────────────────────────
        market_data = self.data_layer.initialize(seed_knowledge=True)
        prices = market_data["prices"]
        volumes = market_data["volumes"]
        knowledge = self.data_layer.get_knowledge()
        results["data"] = {
            "n_stocks": prices.shape[1],
            "n_days": prices.shape[0],
            "n_knowledge_items": len(knowledge),
        }

        # ─── LAYER 2: RESEARCH ───────────────────────────
        ideas = self.research_layer.generate_ideas(
            knowledge, use_llm=use_llm, llm_client=llm_client
        )
        results["research"] = {"n_ideas": len(ideas)}

        # ─── LAYER 3: IMPLEMENTATION ─────────────────────
        implemented = self.implementation_layer.implement_and_validate(
            ideas, prices, volumes,
            use_llm=use_llm, llm_client=llm_client
        )
        results["implementation"] = {
            "total": len(implemented),
            "validated": sum(1 for _, fc in implemented if fc.validated),
        }

        # ─── LAYER 4: EXPERIMENT ─────────────────────────
        experiment_results = self.experiment_layer.run_experiments(
            implemented, prices, volumes
        )
        results["experiment"] = {
            "total": len(experiment_results),
            "passed": sum(1 for _, r in experiment_results if r.passed),
            "details": {
                idea.get("name", idea.get("id", "?")): {
                    "sharpe": r.sharpe_ratio,
                    "annual_return": r.annual_return,
                    "max_drawdown": r.max_drawdown,
                    "passed": r.passed,
                }
                for idea, r in experiment_results
            },
        }

        # ─── LAYER 5: DECISION ───────────────────────────
        decision = self.decision_layer.make_decisions(
            experiment_results, prices, volumes
        )
        target_weights = decision["target_weights"]
        selected = decision.get("selected", [])
        results["decision"] = {
            "n_selected_factors": len(selected),
            "selected_names": [
                idea.get("name", "") for idea, _ in selected
            ],
        }

        # ─── LAYER 6: EXECUTION ──────────────────────────
        portfolio_state = self.execution_layer.run_simulation(
            target_weights, prices, trade_period="last_60d"
        )
        results["execution"] = {
            "n_trades": len(portfolio_state.trade_log),
        }
        if portfolio_state.nav_history:
            initial = portfolio_state.nav_history[0][1]
            final = portfolio_state.nav_history[-1][1]
            results["execution"]["initial_nav"] = initial
            results["execution"]["final_nav"] = final
            results["execution"]["total_return"] = round(
                (final - initial) / initial, 4
            )

        # ─── LAYER 7: REVIEW ─────────────────────────────
        review_report = self.review_layer.review(
            experiment_results, portfolio_state
        )
        results["review"] = {
            "n_insights_generated": len(review_report.insights),
            "n_training_entries": len(review_report.training_data),
            "knowledge_base_size": len(self.data_layer.get_knowledge()),
        }

        # ─── FINAL SUMMARY ───────────────────────────────
        self._print_summary(results)

        return results

    def _print_summary(self, results: dict):
        """Print final pipeline summary."""
        logger.info("\n" + "=" * 60)
        logger.info("  PIPELINE COMPLETE - Summary")
        logger.info("=" * 60)

        logger.info(f"\n  Data:           {results['data']['n_stocks']} stocks, "
                     f"{results['data']['n_days']} days, "
                     f"{results['data']['n_knowledge_items']} knowledge items")

        logger.info(f"  Research:       {results['research']['n_ideas']} ideas generated")

        logger.info(f"  Implementation: {results['implementation']['validated']}/"
                     f"{results['implementation']['total']} validated")

        logger.info(f"  Experiment:     {results['experiment']['passed']}/"
                     f"{results['experiment']['total']} passed quality gate")

        for name, detail in results['experiment']['details'].items():
            status = "✓" if detail['passed'] else "✗"
            logger.info(f"    {status} {name}: Sharpe={detail['sharpe']:.3f}, "
                         f"Return={detail['annual_return']:.2%}, "
                         f"MaxDD={detail['max_drawdown']:.2%}")

        logger.info(f"  Decision:       {results['decision']['n_selected_factors']} factors selected "
                     f"{results['decision']['selected_names']}")

        exec_data = results["execution"]
        logger.info(f"  Execution:      {exec_data.get('n_trades', 0)} trades")
        if "total_return" in exec_data:
            logger.info(f"    NAV: ${exec_data['initial_nav']:,.0f} → "
                         f"${exec_data['final_nav']:,.0f} "
                         f"({exec_data['total_return']:+.2%})")

        review = results["review"]
        logger.info(f"  Review:         {review['n_insights_generated']} insights "
                     f"→ KB size now {review['knowledge_base_size']}")

        logger.info("\n  ✓ Full feedback loop complete. "
                     "Knowledge base enriched for next cycle.\n")
