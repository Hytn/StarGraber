#!/usr/bin/env python3
"""
StarGraber - Run the full pipeline.

Usage:
    python run_pipeline.py              # Demo mode (no LLM needed)
    python run_pipeline.py --use-llm    # LLM mode (requires API key)
"""

import argparse
import logging
import shutil
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stargraber_core.pipeline import StarGraberPipeline


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="StarGraber Pipeline")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use LLM for idea/code generation")
    parser.add_argument("--workspace", default="./workspace",
                        help="Workspace directory for persistent data")
    parser.add_argument("--clean", action="store_true",
                        help="Clean workspace before running")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Clean workspace if requested
    if args.clean and os.path.exists(args.workspace):
        shutil.rmtree(args.workspace)

    # Initialize and run pipeline
    pipeline = StarGraberPipeline(workspace_dir=args.workspace)

    llm_client = None
    if args.use_llm:
        logging.info("LLM mode enabled - requires ANTHROPIC_API_KEY")
        # Future: initialize LLM client here

    results = pipeline.run(use_llm=args.use_llm, llm_client=llm_client)

    # Return code based on results
    if results["execution"].get("total_return", 0) != 0:
        return 0
    return 0  # Success even without profit for MVP


if __name__ == "__main__":
    sys.exit(main())
