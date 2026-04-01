"""
Layer 4: Experiment Layer
- Runs backtests on validated factors
- Computes performance metrics
- Stores and compares results
"""

import numpy as np
import pandas as pd
from dataclasses import asdict
from .models import BacktestResult, JsonStore
import logging

logger = logging.getLogger(__name__)


class Backtester:
    """Simple cross-sectional factor backtesting engine."""

    def __init__(self, transaction_cost_bps: float = 5.0):
        self.tc_bps = transaction_cost_bps / 10000.0

    def run(self, factor_values: pd.DataFrame,
            prices: pd.DataFrame,
            idea_id: str,
            long_pct: float = 0.3,
            short_pct: float = 0.3,
            holding_period: int = 1) -> BacktestResult:
        """
        Cross-sectional backtest.

        Each day:
        1. Rank stocks by factor value
        2. Long top `long_pct`, short bottom `short_pct` (equal weight)
        3. Compute next-day return of the long-short portfolio
        """
        returns = prices.pct_change().shift(-1)  # Next-day returns (what we'd earn)
        # IMPORTANT: shift(-1) means return from t to t+1, aligned to signal at t

        n_stocks = prices.shape[1]
        n_long = max(1, int(n_stocks * long_pct))
        n_short = max(1, int(n_stocks * short_pct))

        portfolio_returns = []
        turnover_list = []
        ic_list = []
        prev_weights = pd.Series(0.0, index=prices.columns)

        valid_dates = factor_values.dropna(how="all").index
        # Use only dates where we have both factor values AND next-day returns
        valid_dates = valid_dates.intersection(returns.dropna(how="all").index)

        for i, date in enumerate(valid_dates):
            if i % holding_period != 0:
                continue

            fv = factor_values.loc[date].dropna()
            ret = returns.loc[date].dropna()

            # Need enough stocks
            common = fv.index.intersection(ret.index)
            if len(common) < n_long + n_short:
                continue

            fv = fv[common]
            ret = ret[common]

            # Rank and assign weights
            ranks = fv.rank(ascending=True)
            weights = pd.Series(0.0, index=common)

            top = ranks.nlargest(n_long).index
            bottom = ranks.nsmallest(n_short).index

            weights[top] = 1.0 / n_long
            weights[bottom] = -1.0 / n_short

            # Portfolio return
            port_ret = (weights * ret).sum()

            # Transaction cost
            weight_change = weights.reindex(prev_weights.index, fill_value=0) - \
                           prev_weights.reindex(weights.index, fill_value=0)
            turnover = weight_change.abs().sum() / 2
            cost = turnover * self.tc_bps
            port_ret -= cost

            portfolio_returns.append(port_ret)
            turnover_list.append(turnover)
            prev_weights = weights

            # Information coefficient: rank correlation of factor vs return
            if len(common) >= 5:
                ic = fv[common].corr(ret[common], method="spearman")
                if not np.isnan(ic):
                    ic_list.append(ic)

        return self._compute_metrics(
            portfolio_returns, turnover_list, ic_list, idea_id
        )

    def _compute_metrics(self, returns: list, turnovers: list,
                         ics: list, idea_id: str) -> BacktestResult:
        """Compute comprehensive backtest metrics."""
        if len(returns) < 10:
            logger.warning(f"    Too few data points ({len(returns)}) for reliable metrics")
            return BacktestResult(idea_id=idea_id, passed=False)

        ret_arr = np.array(returns)
        cum_ret = np.cumprod(1 + ret_arr)

        # Sharpe (annualized, assuming daily)
        mean_ret = np.mean(ret_arr)
        std_ret = np.std(ret_arr, ddof=1)
        sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0

        # Annual return
        n_years = len(ret_arr) / 252
        total_ret = cum_ret[-1] - 1
        annual_ret = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

        # Max drawdown
        peak = np.maximum.accumulate(cum_ret)
        drawdown = (cum_ret - peak) / peak
        max_dd = drawdown.min()

        # IC stats
        ic_mean = np.mean(ics) if ics else 0
        ic_std = np.std(ics, ddof=1) if len(ics) > 1 else 1
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0

        # Win rate
        win_rate = np.mean(ret_arr > 0) if len(ret_arr) > 0 else 0

        # Avg turnover
        avg_turnover = np.mean(turnovers) if turnovers else 0

        # Quality gate
        passed = (sharpe > 0.3 and ic_mean > 0.01 and max_dd > -0.3)

        return BacktestResult(
            idea_id=idea_id,
            sharpe_ratio=round(sharpe, 3),
            annual_return=round(annual_ret, 4),
            max_drawdown=round(max_dd, 4),
            ic_mean=round(ic_mean, 4),
            ic_ir=round(ic_ir, 4),
            turnover=round(avg_turnover, 4),
            total_return=round(total_ret, 4),
            win_rate=round(win_rate, 4),
            equity_curve=cum_ret.tolist(),
            daily_returns=ret_arr.tolist(),
            passed=passed,
        )


class ExperimentLayer:
    """Main interface for Layer 4."""

    def __init__(self, store: JsonStore):
        self.store = store
        self.backtester = Backtester()

    def run_experiments(self, validated_factors: list,
                        prices: pd.DataFrame,
                        volumes: pd.DataFrame) -> list:
        """
        Backtest all validated factors.
        validated_factors: list of (idea_dict, FactorCode) tuples.
        """
        logger.info("=" * 60)
        logger.info("LAYER 4: EXPERIMENT LAYER - Backtesting")
        logger.info("=" * 60)

        results = []

        for idea_dict, factor_code in validated_factors:
            if not factor_code.validated:
                continue

            idea_id = idea_dict.get("id", "unknown")
            idea_name = idea_dict.get("name", "unknown")
            logger.info(f"\n  Backtesting: [{idea_id}] {idea_name}")

            # Compile and compute factor values
            namespace = {"pd": pd, "np": np}
            exec(factor_code.code, namespace)
            factor_fn = namespace["compute_factor"]
            factor_values = factor_fn(prices.copy(), volumes.copy())

            # Run backtest
            bt_result = self.backtester.run(factor_values, prices, idea_id)

            # Store result
            self.store.save("experiments", idea_id, asdict(bt_result))

            # Log results
            status = "✓ PASSED" if bt_result.passed else "✗ FAILED"
            logger.info(f"    {status}")
            logger.info(
                f"    Sharpe: {bt_result.sharpe_ratio:>7.3f}  |  "
                f"Annual Return: {bt_result.annual_return:>7.2%}  |  "
                f"Max DD: {bt_result.max_drawdown:>7.2%}"
            )
            logger.info(
                f"    IC Mean: {bt_result.ic_mean:>7.4f}  |  "
                f"IC IR: {bt_result.ic_ir:>7.4f}  |  "
                f"Win Rate: {bt_result.win_rate:>7.2%}"
            )

            results.append((idea_dict, bt_result))

        passed = sum(1 for _, r in results if r.passed)
        logger.info(f"\n  Summary: {passed}/{len(results)} factors passed quality gate")
        return results
