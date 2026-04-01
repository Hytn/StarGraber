"""
Layer 7: Review Layer
- Analyzes experiment results and live trading performance
- Generates structured insights for the knowledge base
- Produces training data for future model improvement
- Feeds high-quality knowledge back to Layer 1
"""

import numpy as np
from dataclasses import asdict
from .models import KnowledgeItem, ReviewReport
import logging

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """Analyze backtest and live trading results."""

    def analyze_experiments(self, experiment_results: list) -> dict:
        """Analyze cross-experiment patterns."""
        if not experiment_results:
            return {"patterns": [], "best_factor": None}

        passed = [(idea, r) for idea, r in experiment_results if r.passed]
        failed = [(idea, r) for idea, r in experiment_results if not r.passed]

        # Identify best factor
        best = None
        if passed:
            best = max(passed, key=lambda x: x[1].sharpe_ratio)

        # Cross-factor correlation analysis
        patterns = []
        if len(passed) >= 2:
            sharpes = [r.sharpe_ratio for _, r in passed]
            if max(sharpes) > 2 * min(sharpes):
                patterns.append(
                    "Large performance dispersion among factors suggests "
                    "market regime sensitivity. Consider regime-conditional "
                    "factor allocation."
                )

        # Failure analysis
        if failed:
            failure_reasons = []
            for idea, r in failed:
                if r.sharpe_ratio < 0:
                    failure_reasons.append(f"{idea['name']}: negative Sharpe (contrarian?)")
                elif r.max_drawdown < -0.3:
                    failure_reasons.append(f"{idea['name']}: excessive drawdown")
                elif r.ic_mean <= 0.01:
                    failure_reasons.append(f"{idea['name']}: no predictive power")
            patterns.extend(failure_reasons)

        return {
            "patterns": patterns,
            "best_factor": (best[0]["name"], best[1].sharpe_ratio) if best else None,
            "pass_rate": len(passed) / len(experiment_results) if experiment_results else 0,
        }

    def analyze_portfolio(self, portfolio_state) -> dict:
        """Analyze live trading performance."""
        if not portfolio_state.nav_history:
            return {}

        navs = [n for _, n in portfolio_state.nav_history]
        initial = navs[0]
        final = navs[-1]
        total_return = (final - initial) / initial

        # Daily returns from NAV
        nav_arr = np.array(navs)
        daily_rets = np.diff(nav_arr) / nav_arr[:-1]

        sharpe = 0
        if len(daily_rets) > 1 and np.std(daily_rets) > 0:
            sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)

        # Max drawdown
        peak = np.maximum.accumulate(nav_arr)
        dd = (nav_arr - peak) / peak
        max_dd = dd.min()

        return {
            "total_return": round(total_return, 4),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": round(float(max_dd), 4),
            "n_trading_days": len(navs),
            "n_trades": len(portfolio_state.trade_log),
            "final_nav": final,
        }


class InsightGenerator:
    """Generate structured insights from analysis."""

    def generate(self, experiment_analysis: dict,
                 portfolio_analysis: dict,
                 experiment_results: list) -> list:
        """Produce KnowledgeItem insights to feed back to Layer 1."""
        insights = []

        # Insight from best factor performance
        best = experiment_analysis.get("best_factor")
        if best:
            insights.append(KnowledgeItem(
                source_type="review_insight",
                title=f"Top Factor: {best[0]} (Sharpe {best[1]:.2f})",
                content=(
                    f"In the latest experiment cycle, {best[0]} achieved the "
                    f"highest Sharpe ratio of {best[1]:.2f}. This factor should "
                    f"be prioritized in future research iterations and its "
                    f"mechanism studied for potential improvements."
                ),
                tags=["factor-performance", "top-performer", best[0]],
            ))

        # Insight from failure patterns
        patterns = experiment_analysis.get("patterns", [])
        if patterns:
            insights.append(KnowledgeItem(
                source_type="review_insight",
                title="Experiment Cycle Patterns",
                content=(
                    f"Observed patterns from this cycle:\n" +
                    "\n".join(f"- {p}" for p in patterns)
                ),
                tags=["meta-research", "patterns"],
            ))

        # Insight from live portfolio
        if portfolio_analysis:
            ret = portfolio_analysis.get("total_return", 0)
            sharpe = portfolio_analysis.get("sharpe_ratio", 0)
            insights.append(KnowledgeItem(
                source_type="review_insight",
                title=f"Live Simulation: {ret:+.2%} return, Sharpe {sharpe:.2f}",
                content=(
                    f"Live paper trading simulation achieved {ret:+.2%} total return "
                    f"over {portfolio_analysis.get('n_trading_days', 0)} days with "
                    f"Sharpe {sharpe:.2f} and max drawdown "
                    f"{portfolio_analysis.get('max_drawdown', 0):.2%}. "
                    f"Total trades executed: {portfolio_analysis.get('n_trades', 0)}."
                ),
                tags=["live-performance", "portfolio"],
            ))

        return insights

    def generate_training_data(self, experiment_results: list) -> list:
        """
        Generate (idea, outcome) pairs for future model training.
        Each entry: {idea_features, backtest_metrics, label}
        """
        training_data = []
        for idea, result in experiment_results:
            entry = {
                "input": {
                    "name": idea.get("name", ""),
                    "description": idea.get("description", ""),
                    "hypothesis": idea.get("hypothesis", ""),
                    "factor_formula": idea.get("factor_formula", ""),
                },
                "output": {
                    "sharpe_ratio": result.sharpe_ratio,
                    "ic_mean": result.ic_mean,
                    "max_drawdown": result.max_drawdown,
                    "passed": result.passed,
                },
                "label": 1 if result.passed else 0,
            }
            training_data.append(entry)
        return training_data


class ReviewLayer:
    """Main interface for Layer 7."""

    def __init__(self, store, data_layer):
        self.store = store
        self.data_layer = data_layer  # For feeding back insights
        self.analyzer = ResultAnalyzer()
        self.insight_gen = InsightGenerator()

    def review(self, experiment_results: list,
               portfolio_state=None) -> ReviewReport:
        """
        Full review cycle:
        1. Analyze experiment results
        2. Analyze portfolio performance
        3. Generate insights
        4. Feed insights back to knowledge base (Layer 1)
        5. Generate training data
        """
        logger.info("=" * 60)
        logger.info("LAYER 7: REVIEW LAYER - Analysis & Feedback")
        logger.info("=" * 60)

        # Analyze experiments
        exp_analysis = self.analyzer.analyze_experiments(experiment_results)
        logger.info(f"  Pass rate: {exp_analysis['pass_rate']:.0%}")
        if exp_analysis["best_factor"]:
            name, sharpe = exp_analysis["best_factor"]
            logger.info(f"  Best factor: {name} (Sharpe {sharpe:.3f})")

        # Analyze portfolio
        port_analysis = {}
        if portfolio_state and portfolio_state.nav_history:
            port_analysis = self.analyzer.analyze_portfolio(portfolio_state)
            logger.info(f"  Portfolio return: {port_analysis.get('total_return', 0):+.2%}")
            logger.info(f"  Portfolio Sharpe: {port_analysis.get('sharpe_ratio', 0):.3f}")

        # Generate insights and feed back to Layer 1
        insights = self.insight_gen.generate(
            exp_analysis, port_analysis, experiment_results
        )
        logger.info(f"\n  Generated {len(insights)} insights → feeding back to Knowledge Base")
        for insight in insights:
            self.data_layer.add_knowledge(insight)
            logger.info(f"    → {insight.title}")

        # Generate training data
        training_data = self.insight_gen.generate_training_data(experiment_results)
        logger.info(f"  Generated {len(training_data)} training data entries")

        # Save training data
        for i, td in enumerate(training_data):
            self.store.save("training_data", f"entry_{i:04d}", td)

        # Build report
        report = ReviewReport(
            total_ideas_tested=len(experiment_results),
            passed_ideas=[
                idea.get("name", "") for idea, r in experiment_results if r.passed
            ],
            failed_ideas=[
                idea.get("name", "") for idea, r in experiment_results if not r.passed
            ],
            insights=[asdict(i) for i in insights],
            portfolio_metrics=port_analysis,
            training_data=training_data,
        )

        self.store.save("reports", "latest", asdict(report))
        logger.info("\n  Review complete. Feedback loop closed.")
        return report
