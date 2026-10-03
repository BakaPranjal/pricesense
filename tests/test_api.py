import pytest
from fastapi.testclient import TestClient
from backend.app import app

DEMO = dict(expected_monthly_orders=220, return_rate=.08, rto_rate=.07, rating=4.3, review_count=1240, fulfilment_rate=.97, cancellation_rate=.015, late_dispatch_rate=.02, ad_spend_monthly=8000)
@pytest.fixture(scope="module")
def c():
    with TestClient(app) as cl: yield cl

def test_demo_end_to_end(c):
    d = c.get("/api/demo-data").json(); assert d["counts"]["competitors"] >= 100 and d["counts"]["observations"] >= 500 and "synthetic" in d["notice"]
    r = c.post("/api/full-analysis", json=DEMO); assert r.status_code == 200; j = r.json()
    assert j["benchmark"]["n"] >= 20 and j["economics"]["estimated_settlement"] > 0
    o = j["optimization"]; assert set(o["scenarios"]) == {"ENTRY", "GROWTH", "BALANCED", "PROFIT"} and len(o["candidates"]) > 5
    assert j["sensitivity"]["return_sweep"][0]["monthly_contribution"] > j["sensitivity"]["return_sweep"][-1]["monthly_contribution"]
    assert j["explanations"] and "synthetic" in j["data_notice"].lower()

def test_inputs_change_outputs(c):
    base = c.post("/api/full-analysis", json=DEMO).json()
    hi_cost = c.post("/api/full-analysis", json={**DEMO, "total_cost": 210}).json()
    hi_ret = c.post("/api/full-analysis", json={**DEMO, "return_rate": .20}).json()
    low_px = c.post("/api/full-analysis", json={**DEMO, "current_price": 299}).json()
    assert hi_cost["scenarios"] == base["scenarios"] and hi_cost["economics"]["expected_contribution"] == base["economics"]["expected_contribution"]
    assert hi_ret["economics"]["expected_contribution"] < base["economics"]["expected_contribution"]
    assert low_px["benchmark"]["price_percentile"] < base["benchmark"]["price_percentile"]

def test_admin_fee_edit_changes_result(c):
    base = c.post("/api/price-analysis", json=DEMO).json()["expected_contribution"]
    rules = c.get("/api/fee-rules").json(); plat = next(r for r in rules if r["fee_type"] == "platform_fee")
    c.post("/api/admin/fee-rules", json={"id": plat["id"], "amount": 0})
    assert c.post("/api/price-analysis", json=DEMO).json()["expected_contribution"] > base
    c.post("/api/admin/fee-rules", json={"id": plat["id"], "amount": 25})

def test_missing_data_and_new_seller(c):
    j = c.post("/api/full-analysis", json={"current_price": 349, "total_cost": 190, "weight_g": 350}).json()
    assert "return_rate" in j["missing_inputs"] and j["seller_quality"]["new_seller"] and not j["demand_calibrated"]
    j = c.post("/api/full-analysis", json={**DEMO, "category": "Nonexistent", "subcategory": "x", "product_type": "y"}); assert j.status_code == 200

def test_other_endpoints(c):
    assert c.get("/api/market-benchmark").json()["stats"]["median"] > 0
    assert c.get("/api/competitors", params={"category": "Electronics"}).json()["rows"]
    assert c.post("/api/seller-quality", json=DEMO).json()["score"] > 0
    assert c.post("/api/demand-prediction", json={**DEMO, "price": 339}).json()["estimated_orders"] > 0
    assert c.post("/api/sensitivity-analysis", json=DEMO).json()["volume"]
    assert "rule_model_form" in c.get("/api/model-metrics").json()
    assert c.post("/api/admin/model-config", json={"key": "quality_weights", "value": {"adjusted_rating": .25}}).status_code == 200
    assert c.post("/api/products", json=DEMO).json()["id"]

def test_multiple_products_switch_save_delete(c):
    ps = c.get("/api/products").json(); names = [p["name"] for p in ps]
    assert {"Printed Cotton Kurti", "Men's Cotton T-Shirt", "Steel Water Bottle"} <= set(names)
    for p in ps[:3]:
        j = c.post("/api/full-analysis", json=p["payload"]).json()
        assert j["benchmark"]["n"] >= 5 and j["product"]["name"] == p["name"] and len(j["optimization"]["candidates"]) > 5
    pid = c.post("/api/products", json={**ps[0]["payload"], "product_name": "Temp item", "current_price": 399}).json()["id"]
    assert any(p["id"] == pid and p["payload"]["current_price"] == 399 for p in c.get("/api/products").json())
    c.delete(f"/api/products/{pid}"); assert all(p["id"] != pid for p in c.get("/api/products").json())
