"""Shared data models for the StarGraber pipeline."""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid
import json
import os


@dataclass
class KnowledgeItem:
    """A piece of knowledge in the knowledge base (paper, news, insight)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_type: str = ""        # 'arxiv', 'news', 'review_insight'
    title: str = ""
    content: str = ""
    tags: list = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Idea:
    """A factor or investment idea."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    hypothesis: str = ""
    factor_formula: str = ""     # Natural language formula
    source: str = "manual"       # 'llm', 'manual', 'review'
    status: str = "new"          # 'new' -> 'implemented' -> 'validated' -> 'deployed' -> 'retired'
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


@dataclass
class FactorCode:
    """Generated and validated factor implementation."""
    idea_id: str = ""
    function_name: str = ""
    code: str = ""
    validated: bool = False
    validation_errors: list = field(default_factory=list)
    validation_warnings: list = field(default_factory=list)


@dataclass
class BacktestResult:
    """Results from backtesting a factor."""
    idea_id: str = ""
    sharpe_ratio: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    ic_mean: float = 0.0         # Mean information coefficient
    ic_ir: float = 0.0           # IC information ratio
    turnover: float = 0.0
    total_return: float = 0.0
    win_rate: float = 0.0
    equity_curve: list = field(default_factory=list)
    daily_returns: list = field(default_factory=list)
    passed: bool = False         # Whether it passes quality thresholds


@dataclass
class PortfolioState:
    """Current portfolio state."""
    positions: dict = field(default_factory=dict)    # ticker -> quantity
    cash: float = 1_000_000.0
    nav_history: list = field(default_factory=list)  # [(date, nav), ...]
    trade_log: list = field(default_factory=list)     # list of Trade dicts


@dataclass
class ReviewReport:
    """Output from the review layer."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_ideas_tested: int = 0
    passed_ideas: list = field(default_factory=list)
    failed_ideas: list = field(default_factory=list)
    insights: list = field(default_factory=list)       # New knowledge items
    portfolio_metrics: dict = field(default_factory=dict)
    training_data: list = field(default_factory=list)   # For future model training


class JsonStore:
    """Simple JSON-based persistent storage."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save(self, collection: str, key: str, data: dict):
        path = os.path.join(self.base_dir, collection)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"{key}.json"), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, collection: str, key: str) -> Optional[dict]:
        path = os.path.join(self.base_dir, collection, f"{key}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def load_all(self, collection: str) -> list:
        path = os.path.join(self.base_dir, collection)
        if not os.path.exists(path):
            return []
        results = []
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".json"):
                with open(os.path.join(path, fname)) as f:
                    results.append(json.load(f))
        return results

    def list_keys(self, collection: str) -> list:
        path = os.path.join(self.base_dir, collection)
        if not os.path.exists(path):
            return []
        return [f.replace(".json", "") for f in os.listdir(path) if f.endswith(".json")]
