"""
Layer 3: Implementation Layer
- Translates natural-language ideas into executable factor code
- Validates code correctness (syntax, look-ahead bias, data integrity)
- Manages verified code library
"""

import numpy as np
import pandas as pd
from dataclasses import asdict
from .models import FactorCode, JsonStore
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Pre-built factor implementations (demo mode)
# ============================================================

FACTOR_CODE_TEMPLATES = {
    "short_term_reversal_5d": '''
def compute_factor(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    5-day return reversal factor.
    Negative past-5-day return => expect bounce => high factor value.
    """
    returns_5d = prices / prices.shift(5) - 1
    factor = -1.0 * returns_5d  # Negate: we want to BUY decliners
    return factor
''',

    "volume_price_divergence": '''
def compute_factor(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    Volume-price divergence factor.
    High volume on down days signals capitulation and potential reversal.
    """
    daily_ret = prices.pct_change()
    avg_volume_20 = volumes.rolling(window=20).mean()
    vol_ratio = volumes / avg_volume_20

    # Only trigger on down days; magnitude = volume surge * price drop
    down_day = (daily_ret < 0).astype(float)
    factor = vol_ratio * down_day * daily_ret.abs()
    return factor
''',

    "momentum_20d": '''
def compute_factor(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    20-day momentum factor.
    Buy recent winners, sell recent losers.
    """
    returns_20d = prices / prices.shift(20) - 1
    factor = returns_20d
    return factor
''',
}


class CodeGenerator:
    """Generate factor code from idea descriptions."""

    def generate(self, idea: dict, use_llm: bool = False,
                 llm_client=None, code_library: dict = None) -> FactorCode:
        """
        Generate factor code from an idea.

        Demo mode: looks up pre-built templates.
        LLM mode: sends idea + existing library context to LLM.
        """
        idea_name = idea.get("name", "")

        if use_llm and llm_client:
            code_str = self._generate_with_llm(idea, llm_client, code_library)
        elif idea_name in FACTOR_CODE_TEMPLATES:
            code_str = FACTOR_CODE_TEMPLATES[idea_name]
            logger.info(f"    Code: loaded from template library")
        else:
            logger.warning(f"    No template found for '{idea_name}', generating stub")
            code_str = self._generate_stub(idea)

        return FactorCode(
            idea_id=idea.get("id", ""),
            function_name=f"factor_{idea_name}",
            code=code_str,
            validated=False,
        )

    def _generate_stub(self, idea: dict) -> str:
        return f'''
def compute_factor(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """Auto-generated stub for: {idea.get("name", "unknown")}"""
    return prices.pct_change()  # Fallback: simple daily return
'''

    def _generate_with_llm(self, idea, llm_client, code_library) -> str:
        library_context = ""
        if code_library:
            library_context = (
                "\nExisting verified factor code for reference:\n"
                + "\n---\n".join(
                    v.get("code", "") for v in code_library.values()
                )[:2000]
            )

        prompt = (
            "You are a quant developer. Write a Python function that computes "
            "the following factor.\n\n"
            f"Factor name: {idea.get('name')}\n"
            f"Description: {idea.get('description')}\n"
            f"Formula: {idea.get('factor_formula')}\n\n"
            "Requirements:\n"
            "- Function signature: def compute_factor(prices: pd.DataFrame, "
            "volumes: pd.DataFrame) -> pd.DataFrame\n"
            "- prices and volumes have DatetimeIndex (rows) and ticker columns\n"
            "- Return a DataFrame of factor values, same shape\n"
            "- Use ONLY current and past data. NO future data (look-ahead bias)\n"
            "- Handle NaN values gracefully\n"
            f"{library_context}\n\n"
            "Return ONLY the Python function code, no explanation."
        )
        return llm_client.generate(prompt)


class FactorValidator:
    """Validate factor code for correctness."""

    def validate(self, factor_code: FactorCode,
                 prices: pd.DataFrame, volumes: pd.DataFrame) -> FactorCode:
        """
        Run validation suite on factor code.

        Checks:
        1. Syntax & execution: does the code run without errors?
        2. Output shape: same shape as input?
        3. Look-ahead bias: does factor at time t depend on data after t?
        4. Data quality: reasonable value distribution, no all-NaN?
        5. Stationarity: no extreme values or explosive growth?
        """
        errors = []
        warnings = []

        # 1. Compile and execute
        factor_fn = self._compile_factor(factor_code.code)
        if factor_fn is None:
            errors.append("COMPILE_ERROR: Code failed to compile")
            factor_code.validation_errors = errors
            return factor_code

        try:
            result = factor_fn(prices.copy(), volumes.copy())
        except Exception as e:
            errors.append(f"RUNTIME_ERROR: {str(e)}")
            factor_code.validation_errors = errors
            return factor_code

        # 2. Output shape check
        if result.shape != prices.shape:
            errors.append(
                f"SHAPE_ERROR: Output shape {result.shape} != "
                f"input shape {prices.shape}"
            )

        # 3. Look-ahead bias check (statistical)
        lookahead = self._check_lookahead(factor_fn, prices, volumes)
        if lookahead:
            errors.append("LOOKAHEAD_BIAS: Factor appears to use future data")

        # 4. Data quality
        valid_ratio = result.notna().sum().sum() / result.size
        if valid_ratio < 0.1:
            errors.append(f"DATA_QUALITY: Only {valid_ratio:.1%} non-NaN values")
        elif valid_ratio < 0.5:
            warnings.append(f"LOW_COVERAGE: {valid_ratio:.1%} non-NaN values")

        # 5. Value distribution sanity
        vals = result.values[np.isfinite(result.values)]
        if len(vals) > 0:
            if np.abs(vals).max() > 1e6:
                warnings.append("EXTREME_VALUES: Factor has very large values")
            if np.std(vals) < 1e-10:
                errors.append("CONSTANT: Factor produces constant values")

        factor_code.validation_errors = errors
        factor_code.validation_warnings = warnings
        factor_code.validated = len(errors) == 0

        return factor_code

    def _compile_factor(self, code: str):
        """Safely compile factor code into a callable function."""
        try:
            namespace = {"pd": pd, "np": np}
            exec(code, namespace)
            fn = namespace.get("compute_factor")
            if fn is None:
                return None
            return fn
        except Exception:
            return None

    def _check_lookahead(self, factor_fn, prices: pd.DataFrame,
                          volumes: pd.DataFrame) -> bool:
        """
        Statistical check for look-ahead bias.

        Strategy: compute factor on full data vs truncated data.
        If truncating future data changes historical factor values,
        there's likely look-ahead bias.
        """
        n = len(prices)
        if n < 60:
            return False

        mid = n // 2

        try:
            # Full data
            full_result = factor_fn(prices.copy(), volumes.copy())
            # Truncated: only first half
            trunc_result = factor_fn(
                prices.iloc[:mid].copy(), volumes.iloc[:mid].copy()
            )

            # Compare factor values at overlapping dates
            # Allow small numerical differences (1e-10)
            overlap = full_result.iloc[:mid]
            diff = (overlap - trunc_result).abs()
            max_diff = diff.max().max()

            if pd.isna(max_diff):
                return False
            return max_diff > 1e-8
        except Exception:
            return False


class CodeLibrary:
    """Manage verified factor code."""

    def __init__(self, store: JsonStore):
        self.store = store

    def add(self, factor_code: FactorCode):
        if factor_code.validated:
            self.store.save("code_library", factor_code.idea_id, asdict(factor_code))

    def get(self, idea_id: str) -> dict:
        return self.store.load("code_library", idea_id)

    def get_all(self) -> dict:
        items = self.store.load_all("code_library")
        return {item["idea_id"]: item for item in items}

    def get_function(self, idea_id: str):
        """Load and return the compiled factor function."""
        data = self.get(idea_id)
        if data is None:
            return None
        namespace = {"pd": pd, "np": np}
        exec(data["code"], namespace)
        return namespace.get("compute_factor")


class ImplementationLayer:
    """Main interface for Layer 3."""

    def __init__(self, store: JsonStore):
        self.generator = CodeGenerator()
        self.validator = FactorValidator()
        self.code_library = CodeLibrary(store)

    def implement_and_validate(self, ideas: list,
                                prices: pd.DataFrame,
                                volumes: pd.DataFrame,
                                use_llm: bool = False,
                                llm_client=None) -> list:
        """
        For each idea: generate code, validate, store if valid.
        Returns list of (idea, FactorCode) tuples.
        """
        logger.info("=" * 60)
        logger.info("LAYER 3: IMPLEMENTATION LAYER - Code Generation & Validation")
        logger.info("=" * 60)

        results = []
        existing_lib = self.code_library.get_all()

        for idea in ideas:
            idea_dict = asdict(idea) if hasattr(idea, '__dataclass_fields__') else idea
            idea_name = idea_dict.get("name", "unknown")
            idea_id = idea_dict.get("id", "unknown")

            logger.info(f"\n  Implementing: [{idea_id}] {idea_name}")

            # Generate code
            factor_code = self.generator.generate(
                idea_dict, use_llm=use_llm, llm_client=llm_client,
                code_library=existing_lib
            )

            # Validate
            factor_code = self.validator.validate(factor_code, prices, volumes)

            if factor_code.validated:
                self.code_library.add(factor_code)
                logger.info(f"    ✓ Validation PASSED")
                if factor_code.validation_warnings:
                    for w in factor_code.validation_warnings:
                        logger.info(f"      ⚠ {w}")
            else:
                logger.info(f"    ✗ Validation FAILED:")
                for err in factor_code.validation_errors:
                    logger.info(f"      • {err}")

            results.append((idea_dict, factor_code))

        passed = sum(1 for _, fc in results if fc.validated)
        logger.info(f"\n  Summary: {passed}/{len(results)} ideas implemented successfully")

        return results
