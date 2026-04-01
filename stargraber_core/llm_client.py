"""
LLM Client for StarGraber Pipeline
Integrates with Anthropic's Claude API for:
1. Reading knowledge base → generating factor ideas
2. Translating ideas → executable Python code

Requires: ANTHROPIC_API_KEY environment variable
"""

import json
import os
import logging
from dataclasses import asdict
from .models import Idea

logger = logging.getLogger(__name__)

try:
    import httpx
    HTTP_CLIENT = "httpx"
except ImportError:
    import urllib.request
    import urllib.error
    HTTP_CLIENT = "urllib"


class AnthropicClient:
    """Client for Anthropic Claude API."""

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. "
                "Set it via environment variable or pass api_key parameter."
            )

    def _call_api(self, system: str, user_message: str,
                  max_tokens: int = 4096) -> str:
        """Make a raw API call to Claude."""
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user_message}],
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        if HTTP_CLIENT == "httpx":
            resp = httpx.post(self.api_url, json=payload,
                              headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        else:
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))

        # Extract text from response
        text_parts = [
            block["text"]
            for block in data.get("content", [])
            if block.get("type") == "text"
        ]
        return "\n".join(text_parts)

    def generate(self, prompt: str, system: str = "") -> str:
        """Simple generation wrapper."""
        return self._call_api(system or "You are a helpful assistant.", prompt)


class IdeaGeneratorLLM:
    """Use Claude to generate factor ideas from knowledge base."""

    def __init__(self, client: AnthropicClient):
        self.client = client

    def generate_ideas(self, knowledge_items: list,
                       existing_ideas: list = None) -> list:
        """
        Send knowledge base to Claude, get back structured factor ideas.

        Returns list of Idea objects.
        """
        # Build knowledge context
        kb_text = "\n\n".join(
            f"[{item.get('source_type', 'unknown')}] "
            f"{item.get('title', 'Untitled')}\n"
            f"{item.get('content', '')}"
            for item in knowledge_items
        )

        # Build existing ideas context (to avoid duplicates)
        existing_text = ""
        if existing_ideas:
            names = [i.get("name", "") for i in existing_ideas]
            existing_text = (
                f"\n\nAlready explored ideas (do NOT repeat these): "
                f"{', '.join(names)}"
            )

        system_prompt = (
            "You are a senior quantitative researcher at a systematic hedge fund. "
            "You specialize in discovering cross-sectional equity factors from "
            "research literature and market observations. You think rigorously "
            "about statistical validity, economic intuition, and implementation "
            "feasibility."
        )

        user_prompt = f"""Based on the following research materials, generate 2-4 concrete, 
implementable quantitative factor ideas for equity trading.

RESEARCH MATERIALS:
{kb_text}
{existing_text}

For each idea, respond in this exact JSON format (array of objects):
```json
[
  {{
    "name": "snake_case_factor_name",
    "description": "What this factor captures, in 1-2 sentences",
    "hypothesis": "Why this should predict returns, the economic mechanism",
    "factor_formula": "Pseudocode formula using: prices, volumes, returns. Use pandas-like notation."
  }}
]
```

Requirements:
- Each factor must be computable from daily price and volume data only
- Include both the factor construction and the expected direction (long high/short low, or vice versa)
- Be specific about lookback windows and parameters
- Ensure factors are distinct from each other
- Consider transaction costs and turnover in your design

Respond with ONLY the JSON array, no other text."""

        logger.info("  Calling Claude API for idea generation...")
        response = self.client._call_api(system_prompt, user_prompt)

        return self._parse_response(response)

    def _parse_response(self, response: str) -> list:
        """Parse LLM response into Idea objects."""
        # Clean up response (remove markdown code blocks if present)
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]  # Remove first line
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            items = json.loads(text)
            ideas = []
            for item in items:
                idea = Idea(
                    name=item.get("name", "unnamed"),
                    description=item.get("description", ""),
                    hypothesis=item.get("hypothesis", ""),
                    factor_formula=item.get("factor_formula", ""),
                    source="llm",
                )
                ideas.append(idea)
                logger.info(f"    → Idea: {idea.name}")
            return ideas
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error(f"  Failed to parse LLM response: {e}")
            logger.debug(f"  Raw response: {text[:500]}")
            return []


class CodeGeneratorLLM:
    """Use Claude to generate factor implementation code."""

    def __init__(self, client: AnthropicClient):
        self.client = client

    def generate_code(self, idea: dict,
                      existing_code: dict = None) -> str:
        """
        Generate Python factor code from an idea description.

        Returns: Python code string with compute_factor function.
        """
        # Build context from existing code library
        library_context = ""
        if existing_code:
            examples = list(existing_code.values())[:3]
            library_context = (
                "\n\nHere are examples of validated factor functions "
                "in our codebase for reference:\n"
                + "\n---\n".join(
                    ex.get("code", "")[:500] for ex in examples
                )
            )

        system_prompt = (
            "You are an expert Python developer specializing in quantitative "
            "finance. You write clean, efficient, production-quality code for "
            "computing equity factors. You are extremely careful about avoiding "
            "look-ahead bias and handling edge cases."
        )

        user_prompt = f"""Write a Python function that computes the following equity factor.

FACTOR SPECIFICATION:
- Name: {idea.get('name', 'unknown')}
- Description: {idea.get('description', '')}
- Hypothesis: {idea.get('hypothesis', '')}
- Formula: {idea.get('factor_formula', '')}

FUNCTION REQUIREMENTS:
1. Exact signature: def compute_factor(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame
2. `prices` has DatetimeIndex (rows = trading days) and ticker columns (str)
3. `volumes` has the same shape as prices
4. Return a DataFrame of factor values with the SAME shape as input
5. CRITICAL: Use ONLY current and past data. NO future data. Every operation must be backward-looking (.shift(n) with n>0, .rolling(), etc.)
6. Handle NaN values gracefully - early rows with insufficient lookback should be NaN
7. Import pandas as pd and numpy as np at the top of the function if needed

ANTI-PATTERNS TO AVOID:
- prices.mean() without specifying axis (uses all data including future!)
- Any .shift(-n) with negative shift (looks into future)
- Normalizing by full-sample statistics
{library_context}

Respond with ONLY the Python function code, starting with 'def compute_factor'. No explanation, no markdown, no backticks."""

        logger.info(f"    Calling Claude API for code generation: {idea.get('name', '?')}")
        response = self.client._call_api(system_prompt, user_prompt, max_tokens=2048)

        return self._clean_code(response)

    def _clean_code(self, response: str) -> str:
        """Clean up generated code."""
        text = response.strip()

        # Remove markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]

        text = text.strip()

        # Ensure it starts with import or def
        lines = text.split("\n")
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(("import ", "from ", "def ")):
                start_idx = i
                break

        code = "\n".join(lines[start_idx:])

        # Add imports if not present
        if "import pandas" not in code and "pd." in code:
            code = "import pandas as pd\nimport numpy as np\n\n" + code

        return code
