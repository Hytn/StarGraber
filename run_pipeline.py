#!/usr/bin/env python3
"""
StarGraber - Run the full pipeline.

Usage:
    python run_pipeline.py                          # Demo mode (synthetic data, no LLM)
    python run_pipeline.py --real-data               # Real Yahoo Finance data, no LLM
    python run_pipeline.py --use-llm                 # Synthetic data + Claude for ideas/code
    python run_pipeline.py --real-data --use-llm     # Full mode: real data + LLM

Environment:
    ANTHROPIC_API_KEY   Required for --use-llm mode
"""

import argparse
import logging
import shutil
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stargraber_core.pipeline import StarGraberPipeline


def setup_logging(verbose=False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    parser = argparse.ArgumentParser(
        description="StarGraber - AI-Driven Quant Research Pipeline"
    )
    parser.add_argument("--use-llm", action="store_true",
        help="Use Claude API for idea/code generation (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--real-data", action="store_true",
        help="Use real Yahoo Finance data (needs: pip install yfinance)")
    parser.add_argument("--tickers", type=str, default=None,
        help="Comma-separated tickers (default: 16 US stocks)")
    parser.add_argument("--period", type=str, default="1y",
        help="Data period: 1y, 6mo, 2y (default: 1y)")
    parser.add_argument("--workspace", default="./workspace",
        help="Workspace directory")
    parser.add_argument("--clean", action="store_true",
        help="Clean workspace before running")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.clean and os.path.exists(args.workspace):
        shutil.rmtree(args.workspace)

    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None

    llm_client = None
    if args.use_llm:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logging.error("ERROR: --use-llm requires ANTHROPIC_API_KEY.\n"
                          "  export ANTHROPIC_API_KEY=sk-ant-...")
            return 1
        from stargraber_core.llm_client import AnthropicClient
        llm_client = AnthropicClient(api_key=api_key)

    if args.real_data:
        try:
            import yfinance
        except ImportError:
            logging.error("ERROR: --real-data requires yfinance.\n"
                          "  pip install yfinance")
            return 1

    pipeline = StarGraberPipeline(workspace_dir=args.workspace)
    pipeline.run(
        use_llm=args.use_llm, llm_client=llm_client,
        real_data=args.real_data, tickers=tickers, period=args.period,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
