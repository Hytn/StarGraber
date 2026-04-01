"""
Layer 5: Decision Layer
- Selects best factors for live deployment
- Constructs portfolio combining multiple signals
- Applies risk management constraints
"""

import numpy as np
import pandas as pd
from .models import PortfolioState
import logging

logger = logging.getLogger(__name__)


class FactorSelector:
    """Select which validated factors to deploy."""

    def select(self, experiment_results: list,
               max_factors: int = 3,
               min_sharpe: float = 0.3) -> list:
        """
        Select factors for deployment based on backtest quality.
        Returns list of (idea_dict, BacktestResult) for selected factors.
        """
        candidates = [
            (idea, result)
            for idea, result in experiment_results
            if result.passed and result.sharpe_ratio >= min_sharpe
        ]

        # Sort by Sharpe ratio
        candidates.sort(key=lambda x: x[1].sharpe_ratio, reverse=True)
        selected = candidates[:max_factors]

        return selected


class PortfolioConstructor:
    """Construct portfolio from factor signals."""

    def __init__(self, max_position_pct: float = 0.20,
                 max_gross_leverage: float = 2.0):
        self.max_pos = max_position_pct
        self.max_leverage = max_gross_leverage

    def construct(self, factor_signals: dict,
                  prices: pd.DataFrame,
                  weights: dict = None) -> pd.DataFrame:
        """
        Combine multiple factor signals into target portfolio weights.

        factor_signals: {idea_id: pd.DataFrame of factor values}
        weights: {idea_id: float} - blend weights (equal if None)
        Returns: pd.DataFrame of target weights (dates x tickers)
        """
        if not factor_signals:
            return pd.DataFrame()

        # Default equal blend
        if weights is None:
            w = 1.0 / len(factor_signals)
            weights = {k: w for k in factor_signals}

        # Normalize each factor to z-scores cross-sectionally
        normalized = {}
        for idea_id, fv in factor_signals.items():
            mean = fv.mean(axis=1)
            std = fv.std(axis=1)
            std = std.replace(0, 1)
            z = fv.sub(mean, axis=0).div(std, axis=0)
            normalized[idea_id] = z * weights[idea_id]

        # Blend
        blended = sum(normalized.values())

        # Convert z-scores to weights via rank-based approach
        target_weights = blended.rank(axis=1, pct=True) - 0.5  # Center around 0
        # Scale so gross leverage ~ 1.0
        gross = target_weights.abs().sum(axis=1)
        gross = gross.replace(0, 1)
        target_weights = target_weights.div(gross, axis=0)

        # Apply position limits
        target_weights = target_weights.clip(-self.max_pos, self.max_pos)

        # Re-normalize to target leverage
        gross = target_weights.abs().sum(axis=1)
        scale = np.minimum(self.max_leverage / gross.replace(0, 1), 1.0)
        target_weights = target_weights.mul(scale, axis=0)

        return target_weights


class RiskManager:
    """Apply risk constraints to target portfolio."""

    def __init__(self, max_daily_loss_pct: float = 0.02,
                 max_drawdown_pct: float = 0.10):
        self.max_daily_loss = max_daily_loss_pct
        self.max_drawdown = max_drawdown_pct

    def check(self, portfolio_state: PortfolioState) -> dict:
        """
        Check risk limits and return scaling factor.
        Returns {'scale': float, 'warnings': list}
        """
        warnings = []
        scale = 1.0

        if len(portfolio_state.nav_history) >= 2:
            navs = [n for _, n in portfolio_state.nav_history]
            daily_ret = (navs[-1] - navs[-2]) / navs[-2]

            if daily_ret < -self.max_daily_loss:
                scale = 0.5  # Halve exposure after big loss
                warnings.append(
                    f"DAILY_LOSS_LIMIT: {daily_ret:.2%} < -{self.max_daily_loss:.2%}, "
                    f"reducing exposure to {scale:.0%}"
                )

            # Drawdown check
            peak = max(navs)
            dd = (navs[-1] - peak) / peak
            if dd < -self.max_drawdown:
                scale = 0.0  # Stop trading
                warnings.append(
                    f"MAX_DRAWDOWN: {dd:.2%} < -{self.max_drawdown:.2%}, "
                    f"halting trading"
                )

        return {"scale": scale, "warnings": warnings}


class DecisionLayer:
    """Main interface for Layer 5."""

    def __init__(self, code_library):
        self.selector = FactorSelector()
        self.constructor = PortfolioConstructor()
        self.risk_manager = RiskManager()
        self.code_library = code_library

    def make_decisions(self, experiment_results: list,
                       prices: pd.DataFrame,
                       volumes: pd.DataFrame) -> dict:
        """
        Select factors, construct portfolio, apply risk management.
        Returns target weights DataFrame and metadata.
        """
        logger.info("=" * 60)
        logger.info("LAYER 5: DECISION LAYER - Portfolio Construction")
        logger.info("=" * 60)

        # Select best factors
        selected = self.selector.select(experiment_results)
        logger.info(f"  Selected {len(selected)} factors for deployment:")
        for idea, result in selected:
            logger.info(
                f"    [{idea['id']}] {idea['name']} "
                f"(Sharpe: {result.sharpe_ratio:.3f})"
            )

        if not selected:
            logger.warning("  No factors passed selection criteria!")
            return {"target_weights": pd.DataFrame(), "selected": []}

        # Compute live factor signals
        factor_signals = {}
        for idea, result in selected:
            fn = self.code_library.get_function(idea["id"])
            if fn:
                factor_signals[idea["id"]] = fn(prices.copy(), volumes.copy())

        # Construct portfolio
        target_weights = self.constructor.construct(factor_signals, prices)
        logger.info(
            f"  Portfolio: {target_weights.shape[1]} instruments, "
            f"avg gross leverage: {target_weights.abs().sum(axis=1).mean():.2f}"
        )

        return {
            "target_weights": target_weights,
            "selected": selected,
        }
