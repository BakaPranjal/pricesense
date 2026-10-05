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

## Known limitations
Meesho's ranking formula and settlement data are unknown; fee rules are editable assumptions (Fee Rules tab; seller_reported rules override). Demand is hypothetical until trained on real data.
Reverse-shipping, TCS/TDS rates and penalties are model assumptions — verify in the Supplier Panel. Not implemented: auth, real integrations, retraining, A/B tests (architect-only).
