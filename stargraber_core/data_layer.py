"""
Layer 1: Data Layer
- Collects and stores market data
- Maintains knowledge base (papers, news, curated insights)
- For MVP: generates synthetic data with exploitable patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import asdict
from .models import KnowledgeItem, JsonStore
import logging

logger = logging.getLogger(__name__)


class MarketDataGenerator:
    """Generate synthetic market data with embedded factor patterns."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate(self, n_stocks: int = 10, n_days: int = 252,
                 start_date: str = "2024-01-02") -> dict:
        """
        Generate synthetic price/volume data.

        Embeds patterns:
        - Mean-reversion: short-term overreaction followed by reversal
        - Volume-price: high volume on down days signals reversal
        """
        dates = pd.bdate_range(start=start_date, periods=n_days)
        tickers = [f"STK_{i:02d}" for i in range(n_stocks)]

        # Base parameters per stock
        annual_vol = self.rng.uniform(0.15, 0.40, n_stocks)
        daily_vol = annual_vol / np.sqrt(252)
        annual_drift = self.rng.uniform(-0.05, 0.15, n_stocks)
        daily_drift = annual_drift / 252

        # Mean-reversion strength per stock (higher = more reverting)
        mr_strength = self.rng.uniform(0.02, 0.08, n_stocks)

        prices = np.zeros((n_days, n_stocks))
        volumes = np.zeros((n_days, n_stocks))
        prices[0] = self.rng.uniform(20, 200, n_stocks)
        volumes[0] = self.rng.uniform(1e6, 5e6, n_stocks)

        for t in range(1, n_days):
            # Random return
            noise = self.rng.randn(n_stocks) * daily_vol
            ret = daily_drift + noise

            # Mean-reversion component: if recent 5-day return is extreme, push back
            if t >= 5:
                recent_ret = (prices[t-1] - prices[t-5]) / prices[t-5]
                ret -= mr_strength * recent_ret  # Revert proportional to recent move

            prices[t] = prices[t-1] * (1 + ret)
            prices[t] = np.maximum(prices[t], 1.0)  # Floor at $1

            # Volume: higher on big move days (realistic), extra high on down days
            abs_ret = np.abs(ret)
            vol_base = self.rng.uniform(1e6, 5e6, n_stocks)
            vol_shock = vol_base * (1 + 5 * abs_ret)
            # Extra volume on down days (signals reversal in our synthetic world)
            down_mask = ret < -daily_vol
            vol_shock[down_mask] *= 1.5
            volumes[t] = vol_shock

        prices_df = pd.DataFrame(prices, index=dates, columns=tickers)
        volumes_df = pd.DataFrame(volumes, index=dates, columns=tickers)

        logger.info(f"Generated synthetic data: {n_stocks} stocks, {n_days} days")
        return {"prices": prices_df, "volumes": volumes_df}


class KnowledgeBase:
    """Manages the knowledge base: papers, news, curated insights."""

    def __init__(self, store: JsonStore):
        self.store = store

    def add_item(self, item: KnowledgeItem):
        self.store.save("knowledge", item.id, asdict(item))
        logger.info(f"Knowledge base: added '{item.title}' [{item.source_type}]")

    def get_all(self) -> list:
        return self.store.load_all("knowledge")

    def get_by_type(self, source_type: str) -> list:
        return [k for k in self.get_all() if k.get("source_type") == source_type]

    def seed_demo_knowledge(self):
        """Seed with demo research content for the MVP pipeline."""
        items = [
            KnowledgeItem(
                id="kb_001",
                source_type="arxiv",
                title="Short-Term Mean Reversion in Equity Markets",
                content=(
                    "Empirical evidence shows that stocks experiencing extreme short-term "
                    "returns over 3-5 day windows tend to partially reverse. This effect is "
                    "stronger in high-volatility regimes and for smaller-cap stocks. A simple "
                    "5-day return reversal factor generates positive risk-adjusted returns "
                    "with Sharpe ratios of 0.5-1.0 after transaction costs."
                ),
                tags=["mean-reversion", "factor", "short-term", "alpha"]
            ),
            KnowledgeItem(
                id="kb_002",
                source_type="arxiv",
                title="Volume-Price Relationship and Return Predictability",
                content=(
                    "High trading volume on down days often signals capitulation selling "
                    "and predicts short-term bounces. We construct a volume-price divergence "
                    "factor by identifying stocks where volume surges while price declines, "
                    "finding significant next-day return predictability."
                ),
                tags=["volume", "price", "factor", "signal"]
            ),
            KnowledgeItem(
                id="kb_003",
                source_type="news",
                title="Momentum Factor Performance in 2024",
                content=(
                    "Cross-sectional momentum strategies based on 20-day returns have shown "
                    "mixed performance in 2024. While medium-term momentum (1-3 months) "
                    "continues to work, very short-term momentum (1 week) has been crowded "
                    "and shows negative returns after costs."
                ),
                tags=["momentum", "factor", "crowding", "performance"]
            ),
        ]
        for item in items:
            self.add_item(item)
        logger.info(f"Seeded {len(items)} demo knowledge items")


class DataLayer:
    """Main interface for Layer 1."""

    def __init__(self, workspace_dir: str):
        self.store = JsonStore(workspace_dir)
        self.knowledge_base = KnowledgeBase(self.store)
        self.market_gen = MarketDataGenerator()
        self._market_data = None

    def initialize(self, seed_knowledge: bool = True) -> dict:
        """Initialize data layer: generate market data + seed knowledge."""
        logger.info("=" * 60)
        logger.info("LAYER 1: DATA LAYER - Initializing")
        logger.info("=" * 60)

        # Generate synthetic market data
        self._market_data = self.market_gen.generate()
        prices = self._market_data["prices"]
        logger.info(
            f"  Market data: {prices.shape[1]} stocks, "
            f"{prices.shape[0]} days ({prices.index[0].date()} to {prices.index[-1].date()})"
        )

        # Seed knowledge base
        if seed_knowledge:
            self.knowledge_base.seed_demo_knowledge()

        return self._market_data

    def get_market_data(self) -> dict:
        if self._market_data is None:
            self._market_data = self.market_gen.generate()
        return self._market_data

    def get_knowledge(self) -> list:
        return self.knowledge_base.get_all()

    def add_knowledge(self, item: KnowledgeItem):
        self.knowledge_base.add_item(item)
