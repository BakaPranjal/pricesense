# PriceSense — marketplace pricing prototype (local, synthetic data)

Answers: *How competitive is this price? What range should I consider? What deductions and ad-spend levels should I expect?*
**Illustrative synthetic data for demonstration purposes.** Fees are indicative assumptions, not official Meesho tariffs.
The Competitive Visibility Index is a modelled proxy — not Meesho's ranking score.

## Run (Python 3.10+)
    python -m venv .venv
    .venv\Scripts\Activate.ps1          # Windows PowerShell   (Mac/Linux: source .venv/bin/activate)
    pip install -r requirements.txt
    uvicorn backend.app:app --port 8000
    # open http://localhost:8000   (Chart.js loads from a CDN, so the browser needs internet)
Tests: `pytest -q`. Reset data: delete `data/pricesense.db` or use Admin → Synthetic Data Generator.

## Layout
- `backend/engines.py` fee/shipping, returns/RTO, penalties, unit economics, benchmark, seller quality, CVI, rule demand, optimizer
- `backend/seed.py` seeded generator (fee rules, 120+ listings); `backend/db.py` SQLite schema
- `backend/service.py` orchestration; `backend/app.py` FastAPI; `frontend/index.html` Seller + Admin UI; `tests/`

## Key formulas
- Delivered payout = P − commission − platform fee − shipping − GST on applicable fees − TCS/TDS holdbacks
- Return impact = event probability × return-related shipping/recovery assumptions; RTO impact uses the corresponding forward/reverse/handling assumptions
- Competitive position combines price competitiveness with seller-quality, review, return, RTO and fulfilment signals
- Ad benchmark uses synthetic historical observations from the same category/subcategory to show 25th/50th/75th percentile monthly spend
- Adjusted rating = v/(v+m)·R + m/(v+m)·C; review strength = log(1+reviews) normalised; health = 1 − rate/(2·category median)
- Rule demand: Orders(P) = base × (P/median)^−e(P) × quality × ads × season, calibrated to your expected orders at current price
- Price scenarios: lower-entry, recommended and upper-band prices are shown without exposing a profit calculation to the seller

## Known limitations
Meesho's ranking formula and settlement data are unknown; fee rules are editable assumptions (Fee Rules tab; seller_reported rules override). Demand is hypothetical until trained on real data.
Reverse-shipping, TCS/TDS rates and penalties are model assumptions — verify in the Supplier Panel. Not implemented: auth, real integrations, retraining, A/B tests (architect-only).
