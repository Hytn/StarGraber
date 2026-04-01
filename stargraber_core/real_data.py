"""
Real Market Data Fetcher
Fetches actual stock data from Yahoo Finance via yfinance.

Usage:
    fetcher = RealDataFetcher()
    data = fetcher.fetch(tickers=['AAPL','MSFT','GOOGL'], period='1y')
    # returns {'prices': pd.DataFrame, 'volumes': pd.DataFrame}
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


# Default universe: liquid US stocks across sectors
DEFAULT_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',   # Tech
    'JPM', 'BAC', 'GS',                           # Finance
    'JNJ', 'PFE',                                  # Healthcare
    'XOM', 'CVX',                                   # Energy
    'WMT', 'KO',                                    # Consumer
    'TSLA', 'NVDA',                                 # Growth
]


class RealDataFetcher:
    """Fetch real market data from Yahoo Finance."""

    def __init__(self, universe: list = None):
        self.universe = universe or DEFAULT_UNIVERSE

    def fetch(self, tickers: list = None, period: str = '1y',
              start: str = None, end: str = None) -> dict:
        """
        Fetch price and volume data.

        Args:
            tickers: List of stock tickers (default: self.universe)
            period: yfinance period string ('1y', '6mo', '2y', etc.)
            start/end: Alternative date range (YYYY-MM-DD format)

        Returns:
            {'prices': pd.DataFrame, 'volumes': pd.DataFrame}
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for real data. "
                "Install with: pip install yfinance"
            )

        tickers = tickers or self.universe
        logger.info(f"Fetching real market data for {len(tickers)} stocks...")

        try:
            if start and end:
                raw = yf.download(tickers, start=start, end=end,
                                  progress=False, auto_adjust=True)
            else:
                raw = yf.download(tickers, period=period,
                                  progress=False, auto_adjust=True)
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise

        if raw.empty:
            raise ValueError("No data returned. Check tickers and date range.")

        # Extract close prices and volumes
        if len(tickers) == 1:
            prices = raw[['Close']].copy()
            prices.columns = tickers
            volumes = raw[['Volume']].copy()
            volumes.columns = tickers
        else:
            prices = raw['Close'].copy()
            volumes = raw['Volume'].copy()

        # Clean: drop tickers with too many NaNs
        valid_ratio = prices.notna().mean()
        good_tickers = valid_ratio[valid_ratio > 0.8].index.tolist()
        if len(good_tickers) < len(tickers):
            dropped = set(tickers) - set(good_tickers)
            logger.warning(f"Dropped tickers with >20% missing data: {dropped}")

        prices = prices[good_tickers].ffill().dropna()
        volumes = volumes[good_tickers].ffill().dropna()

        # Align indices
        common_idx = prices.index.intersection(volumes.index)
        prices = prices.loc[common_idx]
        volumes = volumes.loc[common_idx]

        logger.info(
            f"Fetched: {len(good_tickers)} stocks, {len(prices)} days "
            f"({prices.index[0].date()} to {prices.index[-1].date()})"
        )

        return {"prices": prices, "volumes": volumes}

    def fetch_with_fundamentals(self, tickers: list = None) -> dict:
        """Fetch price data + basic fundamental info (market cap, sector)."""
        import yfinance as yf

        tickers = tickers or self.universe[:10]
        data = self.fetch(tickers)

        fundamentals = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                fundamentals[ticker] = {
                    'sector': info.get('sector', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', None),
                    'name': info.get('shortName', ticker),
                }
            except Exception:
                fundamentals[ticker] = {'sector': 'Unknown', 'name': ticker}

        data['fundamentals'] = fundamentals
        return data
