import pytest
from fastapi.testclient import TestClient
import backend.engines as E, backend.seed as S
from backend.app import app

@pytest.fixture(scope="module")
def c():
    with TestClient(app) as cl: yield cl
DEMO = dict(expected_monthly_orders=220, return_rate=.08, rto_rate=.07, rating=4.3, review_count=1240, fulfilment_rate=.97, cancellation_rate=.015, late_dispatch_rate=.02, ad_spend_monthly=8000)

def _econ():
    rules, cfg = S.default_rules(), E.make_cfg()
    s = dict(product_cost=0.0, packaging_cost=0.0, weight_g=350, zone="Regional", gst_registered=True, return_rate=.08, rto_rate=.07,
             cancel_rate=.015, late_rate=.02, ad_per_order=3.6)
    return lambda p, o=None: E.unit_economics(p, {**s, **(o or {})}, rules, cfg)

def test_floor_is_true_break_even_and_parts_sum():
    econ = _econ(); f = E.cost_free_floor(econ, 349)
    assert abs(econ(349, {"product_cost": f["max_cost"]})["expected_contribution"]) < 0.01
    assert abs(sum(f["parts_per100"].values()) - f["leak_per100"]) < 1e-6
    assert E.cost_free_floor(econ, 399)["max_cost"] > f["max_cost"]

def test_api_returns_floor_settlement_bridge(c):
    j = c.post("/api/full-analysis", json={**DEMO, "shop_price": 320, "unit_cost": 200}).json()
    assert set(j["price_floor"]["scenarios"]) == {"ENTRY", "GROWTH", "BALANCED", "PROFIT"} and j["settlement"]["BALANCED"]["delivered"] > 0
    b = j["offline_bridge"]; assert b["payout_at_shop_price"] > 0 and b["break_even_price"] > 200

def test_ledger_adds_up(c):
    L = c.post("/api/full-analysis", json={**DEMO, "unit_cost": 200}).json()["offline_bridge"]["ledger"]
    parts = L["delivered_total"] + L["return_total"] + L["rto_total"] + L["penalties"] + L["ads"]
    assert abs(parts - L["total"]) < 0.01 and abs(L["total"] / 100 - L["per_order"]) < 1e-6
