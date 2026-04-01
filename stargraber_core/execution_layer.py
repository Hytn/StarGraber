"""
Layer 6: Execution Layer
- Simulates trade execution
- Tracks portfolio P&L
- Records all trades
"""

import numpy as np
import pandas as pd
from .models import PortfolioState
import logging

logger = logging.getLogger(__name__)


class SimulatedExecutor:
    """Paper trading engine that simulates real execution."""

    def __init__(self, initial_capital: float = 1_000_000,
                 slippage_bps: float = 2.0):
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps / 10000.0

    def execute(self, target_weights: pd.DataFrame,
                prices: pd.DataFrame,
                start_idx: int = None,
                end_idx: int = None) -> PortfolioState:
        """
        Simulate portfolio execution over the given period.

        target_weights: DatetimeIndex x tickers, target weight per day
        prices: DatetimeIndex x tickers, daily prices
        """
        if target_weights.empty:
            return PortfolioState(cash=self.initial_capital)

        # Align dates
        common_dates = target_weights.index.intersection(prices.index)
        if start_idx is not None:
            common_dates = common_dates[start_idx:]
        if end_idx is not None:
            common_dates = common_dates[:end_idx]

        if len(common_dates) == 0:
            return PortfolioState(cash=self.initial_capital)

        tickers = target_weights.columns
        state = PortfolioState(
            positions={t: 0.0 for t in tickers},
            cash=self.initial_capital,
        )

        for date in common_dates:
            current_prices = prices.loc[date]
            target_w = target_weights.loc[date]

            # Current portfolio value
            pos_value = sum(
                state.positions.get(t, 0) * current_prices.get(t, 0)
                for t in tickers
                if not np.isnan(current_prices.get(t, np.nan))
            )
            nav = state.cash + pos_value

            # Target positions in shares
            target_positions = {}
            for t in tickers:
                p = current_prices.get(t, np.nan)
                w = target_w.get(t, 0)
                if np.isnan(p) or np.isnan(w) or p <= 0:
                    target_positions[t] = 0
                else:
                    target_positions[t] = (nav * w) / p

            # Execute trades
            for t in tickers:
                current_pos = state.positions.get(t, 0)
                target_pos = target_positions.get(t, 0)
                delta = target_pos - current_pos

                if abs(delta) < 0.01:  # Minimum trade threshold
                    continue

                price = current_prices.get(t, np.nan)
                if np.isnan(price) or price <= 0:
                    continue

                # Apply slippage
                if delta > 0:
                    exec_price = price * (1 + self.slippage_bps)
                else:
                    exec_price = price * (1 - self.slippage_bps)

                trade_value = delta * exec_price
                state.positions[t] = target_pos
                state.cash -= trade_value

                state.trade_log.append({
                    "date": str(date.date()),
                    "ticker": t,
                    "side": "BUY" if delta > 0 else "SELL",
                    "quantity": round(abs(delta), 2),
                    "price": round(exec_price, 2),
                    "value": round(abs(trade_value), 2),
                })

            # Record NAV
            pos_value = sum(
                state.positions.get(t, 0) * current_prices.get(t, 0)
                for t in tickers
                if not np.isnan(current_prices.get(t, np.nan))
            )
            nav = state.cash + pos_value
            state.nav_history.append((str(date.date()), round(nav, 2)))

        return state


class ExecutionLayer:
    """Main interface for Layer 6."""

    def __init__(self):
        self.executor = SimulatedExecutor()

    def run_simulation(self, target_weights: pd.DataFrame,
                       prices: pd.DataFrame,
                       trade_period: str = "last_60d") -> PortfolioState:
        """
        Execute simulated trading.

        trade_period: 'full' or 'last_Nd' for last N trading days
        """
        logger.info("=" * 60)
        logger.info("LAYER 6: EXECUTION LAYER - Simulated Trading")
        logger.info("=" * 60)

        if target_weights.empty:
            logger.warning("  No target weights provided, skipping execution")
            return PortfolioState()

        # Determine trading period
        start_idx = None
        if trade_period.startswith("last_"):
            n_days = int(trade_period.replace("last_", "").replace("d", ""))
            start_idx = max(0, len(target_weights) - n_days)

        state = self.executor.execute(
            target_weights, prices, start_idx=start_idx
        )

        # Summary
        n_trades = len(state.trade_log)
        if state.nav_history:
            initial = state.nav_history[0][1]
            final = state.nav_history[-1][1]
            total_return = (final - initial) / initial
            logger.info(f"  Trading period: {state.nav_history[0][0]} to {state.nav_history[-1][0]}")
            logger.info(f"  Total trades: {n_trades}")
            logger.info(f"  Initial NAV: ${initial:,.0f}")
            logger.info(f"  Final NAV:   ${final:,.0f}")
            logger.info(f"  Return:      {total_return:+.2%}")
        else:
            logger.info("  No trades executed")

        return state
