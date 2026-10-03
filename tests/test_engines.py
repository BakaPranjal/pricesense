import copy, pytest
from backend import engines as E, seed

RULES = seed.default_rules(); CFG = E.make_cfg()
S = dict(product_cost=180, packaging_cost=10, weight_g=350, zone="Regional", gst_registered=True, return_rate=.08, rto_rate=.07, cancel_rate=.015, late_rate=.02, ad_per_order=0.0)
def rules_with(**kw):
    r = copy.deepcopy(RULES)
    for x in r:
        for fee, upd in kw.items():
            if x["fee_type"] == fee: x.update(upd)
    return r

@pytest.mark.parametrize("w,exp", [(499, 60), (500, 60), (501, 95), (999, 95), (1000, 95), (1001, 140)])
def test_shipping_slab_boundaries(w, exp):
    assert E.shipping_fee(RULES, "forward_shipping", w, "Regional")[0] == exp

def test_gst_on_shipping_only():
    r = rules_with(platform_fee={"enabled": 0})
    e = E.unit_economics(349, S, r, CFG)
    assert e["gst_on_fees"] == pytest.approx(10.8)          # 18% of Rs60

def test_settlement_and_probabilities():
    e = E.unit_economics(349, S, RULES, CFG)
    assert e["p_delivered"] + e["p_return"] + e["p_rto"] == pytest.approx(1) and e["p_delivered"] == pytest.approx(.85)
    assert e["estimated_settlement"] == pytest.approx(349 - 25 - 60 - 0.18 * 85 - 3.49 - 0.349)
    assert e["naive_profit"] == 159

def test_return_and_rto_economics():
    e = E.unit_economics(349, S, RULES, CFG); g = .18
    assert e["contribution_return"] == pytest.approx(-(60 + 60 + g * 120) - 10 - .02 * 180)   # recovery 98%
    assert e["contribution_rto"] == pytest.approx(-(60 + g * 60) - .03 * 190)                  # RTO reverse = 0 demo
    assert e["expected_return_cost"] == pytest.approx(.08 * -e["contribution_return"])

def test_contribution_and_margin():
    e = E.unit_economics(349, S, RULES, CFG)
    c = .85 * e["contribution_delivered"] - e["expected_return_cost"] - e["expected_rto_cost"] - e["expected_penalties"]
    assert e["expected_contribution"] == pytest.approx(c) and e["contribution_margin"] == pytest.approx(c / 349 * 100)

def test_recoverable_tax_is_cash_not_economic():
    e = E.unit_economics(349, S, RULES, CFG)
    assert e["tax_deductions"] > 0 and e["tax_economic_cost"] == 0
    e2 = E.unit_economics(349, {**S, "gst_registered": False}, RULES, CFG)
    assert e2["tax_economic_cost"] == pytest.approx(3.49)       # TCS credit unavailable without GST registration

def test_fee_changes_flow_through():
    base = E.unit_economics(349, S, RULES, CFG)["expected_contribution"]
    no_plat = E.unit_economics(349, S, rules_with(platform_fee={"enabled": 0}), CFG)["expected_contribution"]
    assert no_plat - base == pytest.approx(.85 * 25 * 1.18)
    assert E.unit_economics(349, S, rules_with(commission={"rate": .05}), CFG)["expected_contribution"] < base
    ship80 = [dict(r, amount=80) if r["fee_type"] == "forward_shipping" and r["zone"] == "Regional" and r["weight_max"] == 500 else r for r in RULES]
    assert E.unit_economics(349, S, ship80, CFG)["expected_contribution"] < base

def test_seller_reported_overrides_third_party():
    r = RULES + [dict(fee_type="forward_shipping", amount=70, rate=None, calculation_method="flat", weight_min=0, weight_max=500, zone="Regional", gst_applicable=1, source_type="seller_reported", enabled=1)]
    assert E.shipping_fee(r, "forward_shipping", 350, "Regional")[0] == 70

def test_edge_cases():
    e = E.unit_economics(100, {**S, "product_cost": 150}, RULES, CFG); assert e["expected_contribution"] < 0 and e["warnings"]
    e = E.unit_economics(349, {**S, "return_rate": 1.0, "rto_rate": 0}, RULES, CFG); assert e["p_delivered"] == 0 and e["expected_contribution"] < 0
    e = E.unit_economics(349, {**S, "return_rate": .7, "rto_rate": .6}, RULES, CFG); assert e["p_delivered"] == pytest.approx(0, abs=1e-9)
    assert E.unit_economics(0.0, S, RULES, CFG)["contribution_margin"] == 0
    e = E.unit_economics(349, {**S, "weight_g": 9000}, RULES, CFG); assert any("slab" in w for w in e["warnings"])

def test_bayesian_rating():
    m, C = 200, 4.1
    assert E.bayes_rating(5.0, 5, m, C) < E.bayes_rating(4.5, 5000, m, C)
    assert E.bayes_rating(4.3, 0, m, C) == C

GROUP = [dict(rating=4.0 + i * .05, review_count=100 * (i + 1), return_rate=.12 - i * .005, rto_rate=.10 - i * .004, fulfilment_rate=.94 + i * .004, cancellation_rate=.03 - i * .001) for i in range(10)]
def test_seller_quality_new_seller_not_perfect():
    new = E.seller_quality({"rating": None, "review_count": 0}, GROUP, CFG)
    strong = E.seller_quality(dict(rating=4.7, review_count=9000, return_rate=.03, rto_rate=.02, fulfilment_rate=.99, cancellation_rate=.005), GROUP, CFG)
    fake = E.seller_quality(dict(rating=5.0, review_count=0, return_rate=0, rto_rate=0, fulfilment_rate=1, cancellation_rate=0), GROUP, CFG)
    assert new["new_seller"] and "New Seller" in new["label"] and new["score"] < strong["score"] and fake["score"] == pytest.approx(new["score"])

def test_quality_weights_configurable():
    sel = dict(rating=4.5, review_count=1000, return_rate=.05, rto_rate=.04, fulfilment_rate=.98, cancellation_rate=.01)
    a = E.seller_quality(sel, GROUP, CFG)["score"]
    cfg2 = E.make_cfg({"quality_weights": {"adjusted_rating": 0.9}})
    assert E.seller_quality(sel, GROUP, cfg2)["score"] != pytest.approx(a)

def test_market_percentile_and_stats():
    assert E.price_percentile([100, 200, 300, 400], 250) == 50
    s = E.price_stats([100, 200, 300, 400, 500]); assert s["median"] == 300 and s["min"] == 100 and s["max"] == 500
    assert E.price_stats([]) is None and E.price_stats([300])["median"] == 300

def test_price_competitiveness_smooth_and_monotone():
    v = [E.price_competitiveness(p, 340, 6) for p in range(200, 500, 10)]
    assert all(a >= b for a, b in zip(v, v[1:])) and E.price_competitiveness(340, 340, 6) == pytest.approx(.5)

def test_optimizer_is_computed_not_hardcoded():
    def run(cost):
        s = {**S, "product_cost": cost}; stats = dict(min=239, p10=279, p25=299, median=339, p50=339, p75=380, p90=417, max=519, n=30)
        ctx = dict(sqs=70, ad_per_order=0, median=339, base_orders=220)
        q = E.seller_quality({}, GROUP, CFG)
        return E.optimize(E.candidate_prices(stats, 349, 10), lambda p: E.unit_economics(p, s, RULES, CFG), lambda p: E.rule_orders(p, ctx, CFG),
                          lambda p: E.cvi(p, 339, q, CFG), stats, 30, CFG)
    a, b = run(180), run(260)
    assert a["scenarios"] == b["scenarios"]  # seller sourcing cost is intentionally not an input to the market-pricing model
    assert a["recommended_range"][0] <= a["recommended_range"][1] and len(a["candidates"]) > 5
    assert set(a["scenarios"]) == {"ENTRY", "GROWTH", "BALANCED", "PROFIT"}
    assert all(c["cvi"] >= 0 for c in a["candidates"])

def test_cost_plus_vs_fee_aware():
    p = E.solve_cost_plus(lambda x: E.unit_economics(x, S, RULES, CFG), 100)
    assert p > 190 + 100 and E.unit_economics(p, S, RULES, CFG)["expected_contribution"] == pytest.approx(100, abs=.01)

def test_breakdown_lines_add_up():
    e = E.unit_economics(349, S, RULES, CFG)
    assert e["contribution_return"] == pytest.approx(-(e["forward_shipping"] + e["reverse_shipping"] + e["ret_gst"] + e["packaging_cost"] + e["ret_inventory_loss"]))
    assert e["contribution_rto"] == pytest.approx(-(e["forward_shipping"] + e["rto_reverse_charge"] + e["rto_handling_charge"] + e["rto_gst"] + e["rto_inventory_loss"]))
    assert e["contribution_delivered"] == pytest.approx(e["estimated_settlement"] + e["tax_deductions"] - e["tax_economic_cost"] - e["product_cost"] - e["packaging_cost"])
