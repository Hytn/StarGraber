"""
Layer 2: Research Layer
- Reads knowledge base items
- Generates factor/investment ideas using LLM or rule-based logic
- Manages the idea pool
"""

from dataclasses import asdict
from .models import Idea, JsonStore
import logging

logger = logging.getLogger(__name__)


class IdeaGenerator:
    """Generate factor ideas from knowledge base content."""

    def generate_from_knowledge(self, knowledge_items: list,
                                 use_llm: bool = False,
                                 llm_client=None) -> list:
        """
        Read knowledge items and produce factor ideas.

        In demo mode: uses rule-based mapping from knowledge to ideas.
        In LLM mode: sends knowledge to LLM for idea generation.
        """
        if use_llm and llm_client:
            return self._generate_with_llm(knowledge_items, llm_client)
        return self._generate_demo_ideas(knowledge_items)

    def _generate_demo_ideas(self, knowledge_items: list) -> list:
        """Demo mode: map knowledge items to pre-designed ideas."""
        idea_templates = {
            "mean-reversion": Idea(
                id="idea_001",
                name="short_term_reversal_5d",
                description="5-day return reversal factor. Stocks that declined "
                            "over the past 5 trading days are expected to bounce back.",
                hypothesis="Short-term overreaction leads to mean reversion within 5 days",
                factor_formula="factor = -1 * (price_today / price_5d_ago - 1)",
                source="research",
                status="new",
                metadata={"knowledge_source": "kb_001", "timeframe": "5d"}
            ),
            "volume": Idea(
                id="idea_002",
                name="volume_price_divergence",
                description="Volume-price divergence factor. Identifies stocks where "
                            "volume surged on a down day, signaling potential reversal.",
                hypothesis="High volume on down days indicates capitulation, "
                           "predicting short-term bounce",
                factor_formula=(
                    "vol_ratio = volume_today / avg_volume_20d; "
                    "down_day = (return_today < 0); "
                    "factor = vol_ratio * down_day * abs(return_today)"
                ),
                source="research",
                status="new",
                metadata={"knowledge_source": "kb_002", "timeframe": "1d"}
            ),
            "momentum": Idea(
                id="idea_003",
                name="momentum_20d",
                description="20-day cross-sectional momentum. Go long winners, "
                            "short losers over the past 20 trading days.",
                hypothesis="Medium-term momentum persists due to slow "
                           "information diffusion",
                factor_formula="factor = price_today / price_20d_ago - 1",
                source="research",
                status="new",
                metadata={"knowledge_source": "kb_003", "timeframe": "20d"}
            ),
        }

        ideas = []
        for kb_item in knowledge_items:
            tags = kb_item.get("tags", [])
            for tag_key, idea in idea_templates.items():
                if tag_key in tags and idea not in ideas:
                    ideas.append(idea)
                    break

        logger.info(f"  Generated {len(ideas)} ideas from {len(knowledge_items)} knowledge items")
        return ideas

    def _generate_with_llm(self, knowledge_items, llm_client) -> list:
        """LLM mode: use Claude to generate ideas from knowledge."""
        from .llm_client import IdeaGeneratorLLM
        generator = IdeaGeneratorLLM(llm_client)
        return generator.generate_ideas(knowledge_items)

    def _build_prompt(self, knowledge_items: list) -> str:
        kb_text = "\n\n".join(
            f"[{item.get('source_type', 'unknown')}] {item.get('title', '')}\n"
            f"{item.get('content', '')}"
            for item in knowledge_items
        )
        return (
            "You are a quantitative researcher. Based on the following research "
            "materials, generate 2-3 concrete, implementable factor ideas for "
            "equity trading.\n\n"
            f"Research Materials:\n{kb_text}\n\n"
            "For each idea, provide:\n"
            "- name: snake_case identifier\n"
            "- description: what the factor captures\n"
            "- hypothesis: why it should work\n"
            "- factor_formula: mathematical/pseudocode formula\n\n"
            "Respond in JSON array format."
        )

    def _parse_llm_response(self, response: str) -> list:
        """Parse LLM response into Idea objects."""
        import json
        try:
            data = json.loads(response)
            return [Idea(**{k: v for k, v in item.items() if k in Idea.__dataclass_fields__})
                    for item in data]
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return []


class IdeaManager:
    """Manages the idea pool lifecycle."""

    def __init__(self, store: JsonStore):
        self.store = store

    def add_idea(self, idea: Idea):
        self.store.save("ideas", idea.id, asdict(idea))

    def get_idea(self, idea_id: str) -> dict:
        return self.store.load("ideas", idea_id)

    def get_all_ideas(self) -> list:
        return self.store.load_all("ideas")

    def get_ideas_by_status(self, status: str) -> list:
        return [i for i in self.get_all_ideas() if i.get("status") == status]

    def update_status(self, idea_id: str, new_status: str):
        data = self.store.load("ideas", idea_id)
        if data:
            data["status"] = new_status
            self.store.save("ideas", idea_id, data)


class ResearchLayer:
    """Main interface for Layer 2."""

    def __init__(self, store: JsonStore):
        self.generator = IdeaGenerator()
        self.idea_manager = IdeaManager(store)

    def generate_ideas(self, knowledge_items: list,
                       use_llm: bool = False, llm_client=None) -> list:
        """Read knowledge and generate ideas."""
        logger.info("=" * 60)
        logger.info("LAYER 2: RESEARCH LAYER - Generating Ideas")
        logger.info("=" * 60)

        ideas = self.generator.generate_from_knowledge(
            knowledge_items, use_llm=use_llm, llm_client=llm_client
        )

        for idea in ideas:
            self.idea_manager.add_idea(idea)
            logger.info(f"  Idea: [{idea.id}] {idea.name}")
            logger.info(f"    Hypothesis: {idea.hypothesis}")

        return ideas
