"""Pure calculation engines (no DB, no web): fees, shipping, returns/RTO, unit economics,
market benchmark, seller quality, CVI, rule-based demand, optimizer.
All fee values come from rule dicts (data), never from code constants."""
import copy, math
import numpy as np

QUALITY_W = {"adjusted_rating": .25, "review_strength": .15, "return_health": .20,
             "rto_health": .15, "fulfilment": .15, "cancellation": .10}
CVI_W = {"price_competitiveness": .30, "seller_quality": .30, "review_strength": .10,
         "return_health": .10, "rto_health": .10, "fulfilment": .10}
PARAMS = dict(
    return_recovery_rate=0.98, rto_recovery_rate=0.97, weight_discrepancy_rate=0.03,
    quality_penalty_rate=0.01, bayes_m=200, return_health_method="ratio", price_k=6.0,
    elasticity_below=1.6, elasticity_above=3.4, quality_effect_floor=0.6, quality_effect_span=0.8,
    ad_gain=0.15, seasonality=1.0, similarity_thresholds=[0.75, 0.6, 0.45, 0.3], min_comparables=5,
    min_margin_pct=5.0, min_cvi=35.0, max_price=None, min_orders=0.0,
    growth_w_orders=0.6, balanced_w_contribution=0.5,
    balanced_w_cvi=0.50, balanced_w_orders=0.25, balanced_w_economics=0.15, balanced_w_weight_efficiency=0.10,
    profit_w_economics=0.50, profit_w_margin=0.20, profit_w_cvi=0.20, profit_w_weight_efficiency=0.10,
    max_shipping_share_balanced=0.25, max_shipping_share_profit=0.20, price_step=10, default_rating=4.1,
    sim_weights={"category": .2, "subcategory": .2, "product_type": .3, "attributes": .2, "price_band": .1},
)

def make_cfg(stored=None):
    cfg = {"params": copy.deepcopy(PARAMS), "quality_weights": dict(QUALITY_W), "cvi_weights": dict(CVI_W)}
    for k, v in (stored or {}).items():
        if k in cfg and isinstance(v, dict): cfg[k].update(v)
    return cfg

def sigmoid(x): return 1 / (1 + math.exp(-max(-50, min(50, x))))
def clip(x, lo, hi): return max(lo, min(hi, x))

# ---------------- Fee / shipping engine ----------------
def find_rule(rules, fee_type, zone=None, weight=None):
    c = [r for r in rules if r["fee_type"] == fee_type and r.get("enabled", 1)]
    c.sort(key=lambda r: 0 if r.get("source_type") == "seller_reported" else 1)  # seller actuals win
    for r in c:
        if r.get("zone") not in (None, "", "any", zone): continue
        lo, hi = r.get("weight_min"), r.get("weight_max")
        if weight is not None and lo is not None and hi is not None:
            if not ((lo < weight <= hi) or (weight <= 0 and lo == 0)): continue
        return r
    return None

def rule_value(r, price):
    if not r: return 0.0
    if r["calculation_method"] == "percent_of_price": return price * (r.get("rate") or 0.0)
    return float(r.get("amount") or 0.0)

def shipping_fee(rules, fee_type, weight_g, zone):
    """Slabs are (min, max]: 500g -> 0-500 slab, 501g -> next slab."""
    r, warn = find_rule(rules, fee_type, zone, weight_g), []
    if r is None:
        c = [x for x in rules if x["fee_type"] == fee_type and x.get("enabled", 1)
             and x.get("zone") in (zone, "any", None) and x.get("weight_max")]
        if c:
            r = max(c, key=lambda x: x["weight_max"])
            warn.append(f"Weight {weight_g}g exceeds highest configured slab; using top slab rate.")
    return rule_value(r, 0), r, warn

def gst_rate(rules):
    r = find_rule(rules, "gst")
    return float(r["rate"]) if r else 0.0

def _meta(r):
    if not r: return None
    return {k: r.get(k) for k in ("id", "fee_type", "source", "source_type", "confidence", "status", "notes")}

# ---------------- Unit economics (returns / RTO / penalties inside) ----------------
def unit_economics(price, s, rules, cfg):
    P, warn = cfg["params"], []
    ret, rto = clip(s.get("return_rate", 0.0), 0, 1), clip(s.get("rto_rate", 0.0), 0, 1)
    if ret + rto > 1:
        t = ret + rto; ret, rto = ret / t, rto / t; warn.append("Return + RTO exceeded 100%; scaled proportionally.")
    dlv = 1 - ret - rto
    zone, w = s.get("zone", "Regional"), s.get("weight_g", 350)
    prod, pack = s.get("product_cost", 0.0), s.get("packaging_cost", 0.0)
    g = gst_rate(rules)
    def fee(t):
        r = find_rule(rules, t); return rule_value(r, price), bool(r and r.get("gst_applicable")), r
    comm, comm_g, rc = fee("commission"); plat, plat_g, rp = fee("platform_fee")
    fwd, fr, w1 = shipping_fee(rules, "forward_shipping", w, zone)
    rev, rr, w2 = shipping_fee(rules, "reverse_shipping", w, zone)
    rtor, rtor_g, rrt = fee("rto_reverse"); rtoh, rtoh_g, rrh = fee("rto_handling")
    warn += w1 + w2
    fwd_g, rev_g = bool(fr and fr.get("gst_applicable")), bool(rr and rr.get("gst_applicable"))
    gst_fees = g * ((comm if comm_g else 0) + (plat if plat_g else 0) + (fwd if fwd_g else 0))
    tcs_r, tds_r = find_rule(rules, "tcs"), find_rule(rules, "tds")
    tcs, tds = rule_value(tcs_r, price), rule_value(tds_r, price)
    tcs_rec = bool(tcs_r and tcs_r.get("recoverable") and s.get("gst_registered", True))
    tds_rec = bool(tds_r and tds_r.get("recoverable"))
    tax_econ = (0 if tcs_rec else tcs) + (0 if tds_rec else tds)
    settle_d = price - comm - plat - fwd - gst_fees - tcs - tds             # cash from Meesho, delivered order
    c_d = price - comm - plat - fwd - gst_fees - tax_econ - prod - pack     # business contribution, delivered order
    ret_gst = g * ((fwd if fwd_g else 0) + (rev if rev_g else 0))
    c_r = -(fwd + rev + ret_gst) - pack - (1 - P["return_recovery_rate"]) * prod
    rto_gst = g * ((fwd if fwd_g else 0) + (rtor if rtor_g else 0) + (rtoh if rtoh_g else 0))
    c_t = -(fwd + rtor + rtoh + rto_gst) - (1 - P["rto_recovery_rate"]) * (prod + pack)
    pens = {"late_dispatch": (s.get("late_rate", 0.02), "late_dispatch_penalty"),
            "cancellation": (s.get("cancel_rate", 0.015), "cancellation_penalty"),
            "weight_discrepancy": (P["weight_discrepancy_rate"], "weight_discrepancy_charge"),
            "quality": (P["quality_penalty_rate"], "quality_penalty")}
    pen_detail = {k: p * rule_value(find_rule(rules, t), price) for k, (p, t) in pens.items()}
    pen = sum(pen_detail.values()); ad = s.get("ad_per_order", 0.0)
    ret_cost, rto_cost = -ret * c_r, -rto * c_t
    contrib = dlv * c_d - ret_cost - rto_cost - pen - ad
    exp_settle = dlv * settle_d - ret * (fwd + rev + ret_gst) - rto * (fwd + rtor + rtoh + rto_gst)
    if prod > price: warn.append("Product cost exceeds selling price.")
    if pack > price: warn.append("Packaging cost exceeds selling price.")
    if contrib < 0: warn.append("Expected contribution is negative at this price.")
    used = [_meta(x) for x in (rc, rp, fr, rr, rrt, rrh, tcs_r, tds_r) if x]
    return dict(price=price, p_delivered=dlv, p_return=ret, p_rto=rto, commission=comm, platform_fee=plat,
        forward_shipping=fwd, reverse_shipping=rev, gst_on_fees=gst_fees, tcs=tcs, tds=tds,
        tax_deductions=tcs + tds, tax_economic_cost=tax_econ, estimated_settlement=settle_d,
        expected_settlement=exp_settle, product_cost=prod, packaging_cost=pack,
        contribution_delivered=c_d, contribution_return=c_r, contribution_rto=c_t,
        ret_gst=ret_gst, ret_inventory_loss=(1 - P["return_recovery_rate"]) * prod, rto_gst=rto_gst, rto_reverse_charge=rtor,
        rto_handling_charge=rtoh, rto_inventory_loss=(1 - P["rto_recovery_rate"]) * (prod + pack),
        expected_return_cost=ret_cost, expected_rto_cost=rto_cost, expected_penalties=pen,
        penalty_detail=pen_detail, advertising_cost=ad, expected_contribution=contrib,
        contribution_margin=(contrib / price * 100) if price > 0 else 0.0,
        naive_profit=price - prod - pack, fee_meta=used, warnings=warn)

def blend(items):
    tot = sum(w for w, _ in items) or 1
    out = dict(items[0][1])
    for k, v in out.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = sum(w * d[k] for w, d in items) / tot
    return out

# ---------------- Market benchmark ----------------
def similarity(row, prod, price, w):
    a = {t.strip().lower() for t in (prod.get("attributes") or "").split(",") if t.strip()}
    b = {t.strip().lower() for t in (row.get("attributes") or "").split(",") if t.strip()}
    attr = len(a & b) / len(a | b) if (a | b) else 1.0
    pb = clip(1 - abs(math.log(row["price"] / price)) / 0.5, 0, 1) if price > 0 and row["price"] > 0 else 0
    s = (w["category"] * (row["category"] == prod["category"]) + w["subcategory"] * (row["subcategory"] == prod["subcategory"])
         + w["product_type"] * (row["product_type"] == prod["product_type"]) + w["attributes"] * attr + w["price_band"] * pb)
    return s / sum(w.values())

def select_comparables(rows, prod, price, cfg):
    P = cfg["params"]
    scored = [dict(r, sim=similarity(r, prod, price, P["sim_weights"])) for r in rows]
    for t in P["similarity_thresholds"]:
        sel = [r for r in scored if r["sim"] >= t]
        if len(sel) >= P["min_comparables"]:
            med = float(np.median([r["price"] for r in sel]))
            return [r for r in sel if .5 * med <= r["price"] <= 2 * med], f"similarity >= {t}"
    t = P["similarity_thresholds"][-1]
    sel = [r for r in scored if r["sim"] >= t]
    return sel, (f"similarity >= {t} (few comparables)" if sel else "none")

def price_stats(prices):
    a = np.asarray(prices, float)
    if a.size == 0: return None
    q = np.percentile(a, [10, 25, 50, 75, 90]).tolist()
    return {"min": float(a.min()), "p10": q[0], "p25": q[1], "median": q[2], "p50": q[2], "p75": q[3],
            "p90": q[4], "max": float(a.max()), "n": int(a.size)}

def price_percentile(prices, price):
    a = np.asarray(prices, float)
    return float(100 * np.mean(a <= price)) if a.size else None

# ---------------- Seller quality ----------------
def bayes_rating(R, v, m, C):
    return (v / (v + m)) * R + (m / (v + m)) * C if (v + m) > 0 else C

def seller_quality(sel, group, cfg):
    P, W = cfg["params"], cfg["quality_weights"]; m = P["bayes_m"]
    def col(k): return [g[k] for g in group if g.get(k) is not None]
    def med(k, d):
        vals = col(k); return float(np.median(vals)) if vals else d
    ratings = col("rating")
    base = {"rating": float(np.mean(ratings)) if ratings else P["default_rating"],
            "return_rate": med("return_rate", .09), "rto_rate": med("rto_rate", .08),
            "fulfilment_rate": med("fulfilment_rate", .96), "cancellation_rate": med("cancellation_rate", .02)}
    v = max(0, sel.get("review_count") or 0)
    w = v / (v + m) if v > 0 else 0.0                      # evidence weight; new seller -> pure priors
    R = sel.get("rating") if sel.get("rating") else None
    adj = bayes_rating(R, v, m, base["rating"]) if R else base["rating"]
    def eff(k):
        o = sel.get(k); return base[k] if o is None else w * o + (1 - w) * base[k]
    def health(k):
        e = eff(k)
        if P["return_health_method"] == "percentile":
            g = col(k); return float(np.mean([x >= e for x in g])) if g else .5
        return clip(1 - e / (2 * base[k]), 0, 1) if base[k] > 0 else .5
    maxlog = max([math.log1p(x) for x in col("review_count")] + [math.log1p(v), 1e-9])
    s = {"adjusted_rating": clip((adj - 1) / 4, 0, 1), "review_strength": math.log1p(v) / maxlog,
         "return_health": health("return_rate"), "rto_health": health("rto_rate"),
         "fulfilment": clip((eff("fulfilment_rate") - .80) / .20, 0, 1),
         "cancellation": clip(1 - eff("cancellation_rate") / (2 * base["cancellation_rate"]), 0, 1) if base["cancellation_rate"] > 0 else .5}
    vals = {"adjusted_rating": adj, "review_strength": v, "return_health": eff("return_rate"), "rto_health": eff("rto_rate"),
            "fulfilment": eff("fulfilment_rate"), "cancellation": eff("cancellation_rate")}
    tw = sum(W.values()) or 1
    comps = [{"name": k, "weight": W[k] / tw, "value": vals[k], "score": s[k], "points": 100 * W[k] / tw * s[k]} for k in W]
    return {"score": sum(c["points"] for c in comps), "components": comps, "baseline": base, "new_seller": v == 0,
            "evidence_weight": w,
            "label": "New Seller — estimated from category benchmarks" if v == 0 else "Seller history blended with category priors",
            "note": "Modelled score — NOT Meesho's ranking score."}

# ---------------- Competitive Visibility Index (modelled proxy) ----------------
def price_competitiveness(price, median, k):
    return sigmoid(-k * (price / median - 1)) if median > 0 else .5

def cvi(price, median, q, cfg):
    W = cfg["cvi_weights"]; sc = {c["name"]: c["score"] for c in q["components"]}
    parts = {"price_competitiveness": price_competitiveness(price, median, cfg["params"]["price_k"]),
             "seller_quality": q["score"] / 100, "review_strength": sc["review_strength"],
             "return_health": sc["return_health"], "rto_health": sc["rto_health"], "fulfilment": sc["fulfilment"]}
    tw = sum(W.get(k, 0) for k in parts) or 1
    return 100 * sum(W.get(k, 0) * v for k, v in parts.items()) / tw, parts

# ---------------- Rule-based demand ----------------
def price_effect(price, median, cfg):
    P = cfg["params"]; idx = max(price, 1e-6) / median
    e = P["elasticity_below"] + (P["elasticity_above"] - P["elasticity_below"]) * sigmoid(10 * (idx - 1))
    return idx ** (-e)

def rule_orders(price, ctx, cfg):
    P = cfg["params"]
    qe = P["quality_effect_floor"] + P["quality_effect_span"] * ctx["sqs"] / 100
    ae = 1 + P["ad_gain"] * math.log1p(ctx["ad_per_order"] / 10)
    return ctx["base_orders"] * price_effect(price, ctx["median"], cfg) * qe * ae * P["seasonality"]

# ---------------- Optimizer ----------------
def candidate_prices(stats, current, step):
    lo = min(stats["p10"] * .92, current * .85); hi = max(stats["p90"] * 1.4, current * 1.25)
    ks = range(int(lo // step) + 1, int(hi // step) + 2)
    c = sorted({k * step - 1 for k in ks if k * step - 1 > 0} | {round(current)})
    return c[:60]

def norm(x, xs):
    lo, hi = min(xs), max(xs); return 1.0 if hi == lo else (x - lo) / (hi - lo)

def optimize(cands, econ_fn, orders_fn, cvi_fn, stats, n_comp, cfg, constraints=None):
    P = {**cfg["params"], **{k: v for k, v in (constraints or {}).items() if v is not None}}
    rows = []
    for p in cands:
        e = econ_fn(p); o = max(0.0, orders_fn(p)); cv, _ = cvi_fn(p)
        inside = stats and stats["p10"] <= p <= stats["p90"]
        conf = "High" if inside and n_comp >= 20 else ("Medium" if (stats and stats["min"] <= p <= stats["max"]) or n_comp >= 8 else "Low")
        shipping_cost = (e["forward_shipping"] + e["p_return"] * e["reverse_shipping"]
                         + e["p_rto"] * (e["rto_reverse_charge"] + e["rto_handling_charge"]))
        weight_efficiency = clip(1.0 - shipping_cost / max(float(p), 1e-9), 0.0, 1.0)
        rows.append(dict(price=p, expected_orders=o, expected_contribution=e["expected_contribution"], contribution_margin=e["contribution_margin"],
                         shipping_cost=shipping_cost, weight_efficiency=weight_efficiency, monthly_contribution=o * e["expected_contribution"],
                         return_rate=e["p_return"], rto_rate=e["p_rto"], cvi=cv, confidence=conf))
    def ok(r): return (r["cvi"] >= P["min_cvi"] and r["expected_orders"] >= P["min_orders"] and (P["max_price"] is None or r["price"] <= P["max_price"]))
    feas = [r for r in rows if ok(r)]; relaxed = not feas
    pool = feas or [max(rows, key=lambda r: r["cvi"]) ]
    O, C = [r["expected_orders"] for r in pool], [r["cvi"] for r in pool]
    E, M = [r["expected_contribution"] for r in pool], [r["contribution_margin"] for r in pool]
    gw = P["growth_w_orders"]

    # Entry remains the low-price/traction endpoint. Weight enters the economics
    # through the configured shipping slabs used by unit_economics().
    entry_pick = min(pool, key=lambda r: (r["price"], -r["cvi"]))
    entry_price = entry_pick["price"]

    growth_score = lambda r: gw * norm(r["expected_orders"], O) + (1 - gw) * norm(r["cvi"], C)
    growth_pool = [r for r in pool if r["price"] != entry_price] or list(pool)
    growth_pick = max(growth_pool, key=growth_score)

    def weight_efficiency(r, max_share):
        # A transparent weight-aware viability signal: keep expected weight-driven
        # shipping below the selected share of the selling price.
        burden = r["shipping_cost"] / max(float(r["price"]), 1e-9)
        return clip(1.0 - burden / max_share, 0.0, 1.0)

    # Balanced pricing only considers candidates where the product's weight-driven
    # shipping burden is not excessive relative to selling price. This makes a heavier
    # product naturally need a higher viable price even without asking for product cost.
    balanced_pool = [
        r for r in pool
        if r["price"] not in {entry_price, growth_pick["price"]}
        and r["price"] >= r["shipping_cost"] / max(P["max_shipping_share_balanced"], 1e-9)
    ]
    if not balanced_pool:
        balanced_pool = [r for r in pool if r["price"] not in {entry_price, growth_pick["price"]}] or list(pool)
    balanced_score = lambda r: (
        P["balanced_w_cvi"] * norm(r["cvi"], C)
        + P["balanced_w_orders"] * norm(r["expected_orders"], O)
        + P["balanced_w_economics"] * norm(r["expected_contribution"], E)
        + P["balanced_w_weight_efficiency"] * weight_efficiency(r, P["max_shipping_share_balanced"])
    )
    balanced_pick = max(balanced_pool, key=balanced_score)

    # Profit pricing uses fee-aware contribution more heavily, but only among prices
    # that also pass the stricter weight-aware shipping viability floor.
    profit_pool = [
        r for r in pool
        if r["price"] > balanced_pick["price"]
        and r["price"] >= r["shipping_cost"] / max(P["max_shipping_share_profit"], 1e-9)
    ]
    if not profit_pool:
        profit_pool = [r for r in pool if r["price"] != balanced_pick["price"]] or list(pool)
    profit_score = lambda r: (
        P["profit_w_economics"] * norm(r["expected_contribution"], E)
        + P["profit_w_margin"] * norm(r["contribution_margin"], M)
        + P["profit_w_cvi"] * norm(r["cvi"], C)
        + P["profit_w_weight_efficiency"] * weight_efficiency(r, P["max_shipping_share_profit"])
    )
    profit_pick = max(profit_pool, key=profit_score)
    profit_price = profit_pick["price"]

    pick = {
        "ENTRY": entry_pick,
        "GROWTH": growth_pick,
        "BALANCED": balanced_pick,
        "PROFIT": profit_pick
    }
    # The recommended range remains the Growth–Balanced band; Entry and Profit
    # are shown as outer scenarios for sellers who want to explore the ladder.
    ps = [r["price"] for r in (growth_pick, balanced_pick)]; lo, hi = min(ps), max(ps)
    step = P["price_step"]
    if lo == hi: lo, hi = lo - step, hi + step
    best_score = max((0.5 * norm(r["expected_orders"], O) + 0.5 * norm(r["cvi"], C)) for r in pool)
    near = [r["price"] for r in pool if (0.5 * norm(r["expected_orders"], O) + 0.5 * norm(r["cvi"], C)) >= .95 * best_score] if pool else ps
    return {"candidates": rows, "scenarios": {k: v["price"] for k, v in pick.items()}, "scenario_rows": pick,
            "recommended_range": [lo, hi], "reasonable_range": [min(near), max(near)] if near else [lo, hi],
            "edge_scenarios": [k for k, v in pick.items() if v["price"] in (max(cands), min(cands))],
            "constraints_relaxed": relaxed, "constraints": {k: P[k] for k in ("min_cvi", "max_price", "min_orders")}}

def solve_cost_plus(econ_fn, target, hi=20000):
    """Price at which fee-aware expected contribution/order equals target (bisection)."""
    lo = 0.0
    if econ_fn(hi)["expected_contribution"] < target: return None
    for _ in range(80):
        mid = (lo + hi) / 2
        if econ_fn(mid)["expected_contribution"] < target: lo = mid
        else: hi = mid
    return hi


def cost_free_floor(econ_fn, price):
    """Max all-in unit cost (product+packing) at which expected profit/order is 0, needing no seller cost input.
    Expected profit is linear in cost: A - B*cost, so break-even cost = A/B. Parts are per Rs100 of price."""
    e0 = econ_fn(price, {"product_cost": 0.0, "packaging_cost": 0.0})
    e1 = econ_fn(price, {"product_cost": 1.0, "packaging_cost": 0.0})
    A = e0["expected_contribution"]; B = A - e1["expected_contribution"]
    if B <= 1e-9 or price <= 0: return None
    Y = max(0.0, A / B); leak = price - Y; d = e0["p_delivered"]
    fees = d * (e0["commission"] + e0["platform_fee"] + e0["gst_on_fees"] + e0["tax_economic_cost"]) / B
    ship = d * e0["forward_shipping"] / B
    adp = (e0["expected_penalties"] + e0["advertising_cost"]) / B
    pc = lambda x: 100 * x / price
    return dict(price=price, max_cost=Y, leak=leak, leak_per100=pc(leak),
                parts_per100=dict(fees_taxes=pc(fees), shipping=pc(ship), returns_rto=pc(leak - fees - ship - adp), ads_penalties=pc(adp)))
