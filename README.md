# StarGraber

**AI-Driven Quantitative Research Pipeline** — A 7-layer autonomous system that discovers, implements, validates, and trades quantitative factors using AI.

🔗 **[Live Demo →](https://YOUR_USERNAME.github.io/stargraber/)**

---

## What is this?

StarGraber automates the full quantitative research lifecycle:

```
Knowledge → Ideas → Code → Backtest → Portfolio → Trading → Review → Knowledge
     ↑                                                              ↓
     └──────────────── Closed-loop learning ←──────────────────────┘
```

| Layer | Role | Output |
|-------|------|--------|
| **Data** | Market data + knowledge base | Prices, volumes, research papers |
| **Research** | LLM reads KB, generates ideas | Factor hypotheses with formulas |
| **Implementation** | Code gen + validation suite | Verified Python functions |
| **Experiment** | Backtesting + quality gate | Sharpe, IC, drawdown metrics |
| **Decision** | Factor selection + portfolio | Target weights with risk limits |
| **Execution** | Simulated paper trading | Trade log, NAV history |
| **Review** | Analysis → insights → KB | Feedback loop closed |

## Demo Results

```
3 ideas generated → 3 validated → 2 passed quality gate
✓ short_term_reversal_5d:  Sharpe 4.20, Return +165%
✓ volume_price_divergence: Sharpe 0.65, Return +14%
✗ momentum_20d:            Sharpe -0.55 (rejected)

Paper trading (60 days): $999,800 → $1,077,332 (+7.75%, Sharpe 2.90)
Knowledge base: 3 → 6 items (self-growing)
```

## Quick Start

### Run the pipeline (Python)

```bash
pip install numpy pandas
python run_pipeline.py --clean      # Full pipeline
python tests/test_pipeline.py       # 12 tests
```

### View the dashboard

Open `index.html` in any browser — fully standalone, no server needed.

## Deploy to GitHub Pages

### Option A: GitHub Pages (free)

```bash
# 1. Create repo on GitHub
gh repo create stargraber --public

# 2. Push code
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/stargraber.git
git push -u origin main

# 3. Enable GitHub Pages
#    Go to: Settings → Pages → Source: "main" branch, folder: "/ (root)"
#    Your site will be live at: https://YOUR_USERNAME.github.io/stargraber/
```

### Option B: Custom domain

```bash
# After enabling GitHub Pages:
# 1. Add a CNAME file with your domain
echo "stargraber.yourdomain.com" > CNAME
git add CNAME && git commit -m "Add custom domain" && git push

# 2. Configure DNS
#    Add CNAME record: stargraber.yourdomain.com → YOUR_USERNAME.github.io
#    GitHub will auto-provision HTTPS via Let's Encrypt
```

## Project Structure

```
stargraber/
├── index.html                   # Dashboard (GitHub Pages entry point)
├── .nojekyll                    # Skip Jekyll processing
├── run_pipeline.py              # Python pipeline entry point
├── stargraber_core/
│   ├── models.py                # Shared data models
│   ├── data_layer.py            # Layer 1: Data + Knowledge Base
│   ├── research_layer.py        # Layer 2: Idea Generation
│   ├── implementation_layer.py  # Layer 3: Code Gen + Validation
│   ├── experiment_layer.py      # Layer 4: Backtesting
│   ├── decision_layer.py        # Layer 5: Portfolio + Risk
│   ├── execution_layer.py       # Layer 6: Simulated Trading
│   ├── review_layer.py          # Layer 7: Analysis + Feedback
│   └── pipeline.py              # Orchestrator
├── tests/
│   └── test_pipeline.py         # 12 tests covering all layers
└── requirements.txt
```

## Roadmap

- [ ] Real market data (Yahoo Finance / Alpha Vantage)
- [ ] LLM integration via `--use-llm` flag (Anthropic API)
- [ ] arXiv / news API for live knowledge ingestion
- [ ] Broker API (Alpaca / IBKR) for live trading
- [ ] World model training on accumulated idea→outcome data
- [ ] Strategy decay detection + automated alerts
- [ ] Multi-cycle continuous operation mode

## License

MIT
