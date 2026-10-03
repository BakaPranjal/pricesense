"""Orchestration layer: wires DB data + engines + ML into API-ready results."""
import json, math
import numpy as np, pandas as pd
from . import db, engines as E

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if (isinstance(o, float) and (math.isnan(o) or math.isinf(o))) else float(o)
    if isinstance(o, np.integer): return int(o)
    return o

def build_context(inp):
    rules, cfg = db.load_rules(), db.load_cfg(); P = cfg["params"]
    cur = float(inp["current_price"])
    prod = {k: inp.get(k) for k in ("category", "subcategory", "product_type", "attributes")}
    # Market recommendation should be driven by the product/category and comparable sellers,
    # not by the seller's currently listed price. The current price is used only to show
    # where the seller sits on the market today.
    all_comps = db.rows("SELECT * FROM competitors")
    scoped_prices = [
        float(r["price"]) for r in all_comps
        if r.get("category") == prod.get("category") and r.get("subcategory") == prod.get("subcategory")
    ]
    if not scoped_prices:
        scoped_prices = [float(r["price"]) for r in all_comps if r.get("category") == prod.get("category")]
    reference_price = float(np.median(scoped_prices)) if scoped_prices else cur
    comps, level = E.select_comparables(all_comps, prod, reference_price, cfg)
    stats = E.price_stats([c["price"] for c in comps])
    fallback = stats is None
    if fallback:
        stats = {k: reference_price * f for k, f in dict(min=.8, p10=.85, p25=.93, median=1, p50=1, p75=1.07, p90=1.15, max=1.2).items()}; stats["n"] = 0
    med = stats["median"]
    sel = {k: inp.get(k) for k in ("rating", "review_count", "return_rate", "rto_rate", "fulfilment_rate", "cancellation_rate")}
    q = E.seller_quality(sel, comps, cfg); base = q["baseline"]
    missing = [k for k in ("return_rate", "rto_rate", "fulfilment_rate") if inp.get(k) is None]
    exp_orders = inp.get("expected_monthly_orders") or 0.0
    ad_po = (inp.get("ad_spend_monthly") or 0) / exp_orders if exp_orders > 0 else 0.0
    s = dict(product_cost=0.0, packaging_cost=0.0, weight_g=inp["weight_g"], zone=inp.get("zone", "Regional"),
             gst_registered=inp.get("gst_registered", True), return_rate=inp["return_rate"] if inp.get("return_rate") is not None else base["return_rate"],
             rto_rate=inp["rto_rate"] if inp.get("rto_rate") is not None else base["rto_rate"],
             cancel_rate=inp["cancellation_rate"] if inp.get("cancellation_rate") is not None else base["cancellation_rate"],
             late_rate=inp.get("late_dispatch_rate") if inp.get("late_dispatch_rate") is not None else .02, ad_per_order=ad_po)
    mix = inp.get("zone_mix") or {s["zone"]: 1.0}
    def econ(price, s_over=None, rules_over=None):
        ss = {**s, **(s_over or {})}
        return E.blend([(w, E.unit_economics(price, {**ss, "zone": z}, rules_over or rules, cfg)) for z, w in mix.items()])
    rctx = dict(sqs=q["score"], ad_per_order=ad_po, median=med, base_orders=float(np.median([c["est_monthly_orders"] for c in comps])) if comps else 150.0)
    # Calibrate demand at a fixed market reference price so changing the seller's
    # current listed price does not move the recommended price ladder.
    k_rule = exp_orders / E.rule_orders(med, rctx, cfg) if exp_orders > 0 else 1.0
    rule_fn = lambda p, qd=0: k_rule * E.rule_orders(p, {**rctx, "sqs": clip100(rctx["sqs"] + qd)}, cfg)
    cvi_fn = lambda p: E.cvi(p, med, q, cfg)
    return dict(inp=inp, rules=rules, cfg=cfg, comps=comps, level=level, stats=stats, fallback=fallback, q=q, s=s, econ=econ, mix=mix,
                rule_fn=rule_fn, cvi_fn=cvi_fn, missing=missing, cur=cur, exp_orders=exp_orders,
                calibrated=exp_orders > 0, k_rule=k_rule, rctx=rctx)

def clip100(x): return max(0.0, min(100.0, x))

def confidence(ctx):
    sc = {"high": 1, "medium": .6, "low": .3}
    metas = ctx["econ"](ctx["cur"])["fee_meta"]
    vals = [1.0 if m["source_type"] == "seller_reported" else sc.get(m["confidence"], .3) for m in metas] or [.3]
    avg = sum(vals) / len(vals); n = ctx["stats"]["n"]
    return {"economic": {"level": "High" if avg >= .8 else "Medium" if avg >= .5 else "Low", "why": f"Average fee-input confidence {avg:.2f}; most fee inputs are indicative, not confirmed from seller settlements."},
            "market": {"level": "High" if n >= 20 else "Medium" if n >= 8 else "Low", "why": f"{n} synthetic comparable listings ({ctx['level']})."},
            "demand": {"level": "Low", "why": "Rule-based elasticity assumptions; hypothetical until trained on real seller/listing data."}}

def sensitivity(ctx, price, orders, target=0.0):
    econ, s = ctx["econ"], ctx["s"]
    mo = lambda **kw: orders * econ(price, kw.get("s"), kw.get("rules"))["expected_contribution"]
    rows_r = [{"rate": r, "monthly_contribution": mo(s={"return_rate": r})} for r in (.05, .10, .15, .20, .25)]
    rows_t = [{"rate": r, "monthly_contribution": mo(s={"rto_rate": r})} for r in (.03, .07, .10, .15, .20, .25)]
    def max_rate(key, other):
        f = lambda r: mo(s={key: r}) - target
        hi = 1 - other
        if f(0) < 0: return {"value": None, "note": "Below target even at 0%."}
        if f(hi) >= 0: return {"value": hi, "note": "Target met at any rate."}
        lo = 0.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if f(mid) >= 0: lo = mid
            else: hi = mid
        return {"value": lo, "note": f"Monthly contribution stays >= ₹{target:,.0f} up to this rate."}
    e = econ(price); n_ad = s["ad_per_order"]; vol = []
    for n in (100, 250, 500, 1000, 2500):
        d = n * e["p_delivered"]
        vol.append(dict(orders=n, revenue=d * price, marketplace_deductions=d * (e["commission"] + e["platform_fee"] + e["forward_shipping"] + e["gst_on_fees"] + e["tax_economic_cost"]),
                        product_cost=d * e["product_cost"], packaging=d * e["packaging_cost"], return_cost=n * e["expected_return_cost"], rto_cost=n * e["expected_rto_cost"],
                        penalties=n * e["expected_penalties"], advertising=n * n_ad, contribution=n * e["expected_contribution"]))
    base = mo()
    scale = lambda types: [dict(r, amount=(r["amount"] or 0) * 1.1) if r["fee_type"] in types else r for r in ctx["rules"]]
    D = [("Shipping cost (+10%)", mo(rules=scale(("forward_shipping", "reverse_shipping")))),
         ("Return rate (+10% relative)", mo(s={"return_rate": s["return_rate"] * 1.1})), ("RTO rate (+10% relative)", mo(s={"rto_rate": s["rto_rate"] * 1.1})),
         ("Product cost (+10%)", mo(s={"product_cost": s["product_cost"] * 1.1})), ("Packaging cost (+10%)", mo(s={"packaging_cost": s["packaging_cost"] * 1.1})),
         ("Platform fee (+10%)", mo(rules=scale(("platform_fee",)))),
         ("Price competitiveness (price −10%)", ctx["rule_fn"](price * .9) * econ(price * .9)["expected_contribution"] - ctx["rule_fn"](price) * econ(price)["expected_contribution"] + base),
         ("Seller quality (+10 points)", ctx["rule_fn"](price, 10) * e["expected_contribution"])]
    drivers = [{"factor": n, "delta": v - base} for n, v in D]; mx = max(abs(d["delta"]) for d in drivers) or 1
    for d in drivers: d["impact"] = "High" if abs(d["delta"]) >= .6 * mx else "Medium" if abs(d["delta"]) >= .25 * mx else "Low"
    drivers.sort(key=lambda d: -abs(d["delta"]))
    return {"return_sweep": rows_r, "rto_sweep": rows_t, "max_return_rate": max_rate("return_rate", s["rto_rate"]), "max_rto_rate": max_rate("rto_rate", s["return_rate"]),
            "volume": vol, "drivers": drivers, "biggest_lever": drivers[0]["factor"], "base_monthly_contribution": base, "target": target}

def explain(ctx, opt, cur_e):
    cur, st, c = ctx["cur"], ctx["stats"], opt["candidates"]; med = st["median"]; out = []
    pct = (cur / med - 1) * 100
    out.append(f"Your current price ₹{cur:,.0f} is {abs(pct):.0f}% {'above' if pct > 0 else 'below'} the synthetic comparable median ₹{med:,.0f} (based on {st['n']} comparables).")
    out.append(f"At ₹{cur:,.0f}, the modelled competitive position is {ctx['cvi_fn'](cur)[0]:.0f}/100 and expected monthly orders are about {ctx['rule_fn'](cur):.0f} under the demo demand assumptions.")
    sc = opt["scenario_rows"]; bal, gro = sc["BALANCED"], sc["GROWTH"]
    cr = min(c, key=lambda r: abs(r["price"] - cur))
    if bal["price"] != cr["price"]:
        out.append(f"Recommended ₹{bal['price']:,.0f} vs ₹{cr['price']:,.0f}: modelled orders change by {((bal['expected_orders']/max(cr['expected_orders'],1e-9))-1)*100:+.0f}% while competitive position moves from {cr['cvi']:.0f} to {bal['cvi']:.0f}/100 (demo assumptions).")
    higher = [r for r in c if r["price"] > max(bal["price"], gro["price"])]
    if higher:
        r = higher[min(2, len(higher) - 1)]
        out.append(f"Moving up to ₹{r['price']:,.0f} is associated with a lower modelled competitive position of {r['cvi']:.0f}/100 and lower modelled orders of {r['expected_orders']:.0f}.")
    out.append(f"The growth scenario ₹{gro['price']:,.0f} emphasises expected orders and visibility, while the balanced scenario ₹{bal['price']:,.0f} also accounts for fee-aware economics, including weight-dependent shipping costs. The profit scenario uses the same weight-aware economics more heavily.")
    return out


def _score_comparable(row, stats):
    med = max(float(stats.get('median') or row.get('price') or 1), 1e-9)
    price_comp = E.clip(1 - abs(math.log(max(row.get('price', med), 1e-9) / med)) / .55, 0, 1)
    rating = E.clip((float(row.get('rating') or 3.0) - 3.0) / 1.9, 0, 1)
    revs = E.clip(math.log1p(float(row.get('review_count') or 0)) / 9.0, 0, 1)
    ret_med = max(float(stats.get('avg_return') or .10), .03)
    rto_med = max(float(stats.get('avg_rto') or .10), .03)
    return_h = E.clip(1 - float(row.get('return_rate') or ret_med) / (2 * ret_med), 0, 1)
    rto_h = E.clip(1 - float(row.get('rto_rate') or rto_med) / (2 * rto_med), 0, 1)
    fulf = E.clip((float(row.get('fulfilment_rate') or .90) - .80) / .20, 0, 1)
    seller_q = .28*rating + .18*revs + .22*return_h + .17*rto_h + .15*fulf
    score = 100*(.30*price_comp + .30*seller_q + .10*revs + .10*return_h + .10*rto_h + .10*fulf)
    return score

def _competitive_comparables(ctx, user_score):
    out=[]
    for row in ctx['comps']:
        score=_score_comparable(row, ctx['stats'])
        reasons=[]
        if row.get('price',0) < ctx['cur']*0.97: reasons.append('lower price')
        if row.get('rating') is not None and ctx['inp'].get('rating') is not None and row['rating'] > ctx['inp']['rating']+.15: reasons.append('higher rating')
        if row.get('review_count') is not None:
            user_reviews=ctx['inp'].get('review_count') or 0
            if row['review_count'] > max(user_reviews*1.4, user_reviews+100): reasons.append('stronger review proof')
        base_ret=ctx['s']['return_rate']
        if row.get('return_rate') is not None and row['return_rate'] < base_ret-.025: reasons.append('lower returns')
        base_rto=ctx['s']['rto_rate']
        if row.get('rto_rate') is not None and row['rto_rate'] < base_rto-.02: reasons.append('lower RTO')
        if row.get('fulfilment_rate') is not None and row['fulfilment_rate'] > .97: reasons.append('strong fulfilment')
        out.append(dict(seller_name=row.get('seller_name','Synthetic Seller'),price=row.get('price'),score=score,rating=row.get('rating'),review_count=row.get('review_count'),return_rate=row.get('return_rate'),rto_rate=row.get('rto_rate'),fulfilment_rate=row.get('fulfilment_rate'),reasons=reasons))
    return out

def _ad_benchmark(inp):
    cat, sub = inp.get('category'), inp.get('subcategory')
    rows=db.rows("SELECT ad_spend, orders, price, category, subcategory FROM orders WHERE category=? AND subcategory=? AND ad_spend>0", (cat, sub))
    if len(rows)<10:
        rows=db.rows("SELECT ad_spend, orders, price, category, subcategory FROM orders WHERE category=? AND ad_spend>0", (cat,))
    if not rows: return {'n':0,'category_label':f'{cat} · {sub}'}
    ads=np.array([float(x['ad_spend']) for x in rows],dtype=float)
    p25,p50,p75=[float(np.percentile(ads,p)) for p in (25,50,75)]
    cpos=[float(x['ad_spend'])/max(float(x['orders'] or 1),1) for x in rows]
    return {'n':len(rows),'category_label':f'{cat} · {sub}','p25':p25,'p50':p50,'p75':p75,
            'ad_per_order_median':float(np.median(cpos)),'current':inp.get('ad_spend_monthly') if inp.get('ad_spend_monthly') is not None else None,
            'note':'Synthetic historical observations; not an official marketplace benchmark.'}

def analyze(inp):
    ctx = build_context(inp); cfg, cur = ctx["cfg"], ctx["cur"]
    econ = ctx["econ"]; cur_e = econ(cur)
    # Candidate price pool is generated from the fixed market statistics, not the
    # seller's current listed price.
    cands = E.candidate_prices(ctx["stats"], ctx["stats"]["median"], cfg["params"]["price_step"])
    sel = E.optimize(cands, econ, ctx["rule_fn"], ctx["cvi_fn"], ctx["stats"], ctx["stats"]["n"], cfg, inp.get("constraints")); orders_cur = ctx["rule_fn"](cur)
    cv, parts = ctx["cvi_fn"](cur)
    sens = sensitivity(ctx, cur, ctx["rule_fn"](cur), inp.get("target_monthly_contribution", 0.0))
    scen = dict(sel["scenarios"]); allp = {**scen, "CURRENT": cur}
    price_floor = dict(current=E.cost_free_floor(econ, cur), scenarios={k: E.cost_free_floor(econ, v) for k, v in scen.items()})
    settlement = {k: dict(delivered=econ(v)["estimated_settlement"], expected=econ(v)["expected_settlement"]) for k, v in allp.items()}
    st, rr, shop, uc = ctx["stats"], sel["recommended_range"], inp.get("shop_price"), inp.get("unit_cost")
    bridge = dict(online_low=st["p25"], online_high=st["p75"], median=st["median"], n=st["n"], rec_low=rr[0], rec_high=rr[1],
                  payout_at_range=[econ(rr[0])["estimated_settlement"], econ(rr[1])["estimated_settlement"]])
    if shop:
        bridge.update(shop_price=shop, shop_vs_median_pct=(shop / st["median"] - 1) * 100, payout_at_shop_price=econ(shop)["estimated_settlement"])
    if uc:
        ce = lambda p, c=uc: econ(p, {"product_cost": c})
        bridge.update(unit_cost=uc, break_even_price=E.solve_cost_plus(ce, 0.0),
                      profit_per_order={k: ce(v)["expected_contribution"] for k, v in scen.items()})
        ee, n = ce(cur), 100
        bridge["ledger"] = dict(price=cur, unit_cost=uc, n_delivered=ee["p_delivered"] * n, n_return=ee["p_return"] * n, n_rto=ee["p_rto"] * n,
            delivered_unit=ee["contribution_delivered"], delivered_total=ee["p_delivered"] * ee["contribution_delivered"] * n,
            return_total=ee["p_return"] * ee["contribution_return"] * n, rto_total=ee["p_rto"] * ee["contribution_rto"] * n,
            penalties=-ee["expected_penalties"] * n, ads=-ee["advertising_cost"] * n, total=ee["expected_contribution"] * n,
            per_order=ee["expected_contribution"], break_even_price=bridge["break_even_price"],
            tax_held=ee["tax_deductions"] - ee["tax_economic_cost"], cash_unit=ee["estimated_settlement"] - uc)
    res = dict(product=dict(name=inp.get("product_name"), category=inp["category"], subcategory=inp["subcategory"], product_type=inp["product_type"], current_price=cur),
        economics=cur_e, benchmark=dict(stats=ctx["stats"], level=ctx["level"], n=ctx["stats"]["n"], fallback=ctx["fallback"],
            price_percentile=E.price_percentile([c["price"] for c in ctx["comps"]], cur), price_index=cur / ctx["stats"]["median"], label="Synthetic competitors — illustrative synthetic data for demonstration purposes."),
        seller_quality=ctx["q"], cvi=dict(score=cv, parts=parts, label="Competitive Positioning (modelled proxy — not Meesho's ranking score)"),
        expected_orders_current=orders_cur, expected_monthly_contribution=orders_cur * cur_e["expected_contribution"], demand_calibrated=ctx["calibrated"],
        optimization=sel, scenarios=sel["scenarios"], recommended_range=sel["recommended_range"], reasonable_range=sel["reasonable_range"],
        competitive_comparables=_competitive_comparables(ctx, cv), ad_benchmark=_ad_benchmark(inp),
        price_floor=price_floor, settlement=settlement, offline_bridge=bridge,
        confidence=confidence(ctx), sensitivity=sens, explanations=explain(ctx, sel, cur_e),
        missing_inputs=ctx["missing"], warnings=cur_e["warnings"], data_notice="Illustrative synthetic data for demonstration purposes.")
    res = clean(res)
    with db.conn() as c:
        c.execute("INSERT INTO price_simulations(product_id,inputs,outputs) VALUES(1,?,?)", (json.dumps(inp), json.dumps({"scenarios": res["scenarios"], "range": res["recommended_range"]})))
        c.execute("INSERT INTO market_benchmarks(category,subcategory,product_type,n,stats) VALUES(?,?,?,?,?)", (inp["category"], inp["subcategory"], inp["product_type"], ctx["stats"]["n"], json.dumps(clean(ctx["stats"]))))
    return res
