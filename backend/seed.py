"""Reproducible synthetic data (fixed seed). Illustrative synthetic data for demonstration purposes."""
import math
import numpy as np, pandas as pd

SEED = 42
# category -> subcategory -> (product types, median price, base monthly orders)
CATS = {
 "Women's Fashion": {"Kurtis": (["Cotton Kurti", "Rayon Kurti", "Anarkali Kurti"], 339, 220), "Sarees": (["Cotton Saree", "Silk Saree"], 450, 160)},
 "Men's Fashion": {"T-Shirts": (["Cotton T-Shirt", "Polo T-Shirt"], 299, 260), "Shirts": (["Casual Shirt", "Formal Shirt"], 420, 180)},
 "Home & Kitchen": {"Kitchen Storage": (["Container Set", "Steel Bottle"], 249, 200), "Home Decor": (["Wall Hanging", "Cushion Cover"], 279, 150)},
 "Beauty & Personal Care": {"Skincare": (["Face Serum", "Face Wash"], 249, 240), "Haircare": (["Hair Oil", "Hair Mask"], 229, 210)},
 "Electronics": {"Accessories": (["Phone Cover", "Earphones", "Charger"], 399, 170)},
}
ZONES = ["Local", "Regional", "National"]

def _fee(ft, amount=None, rate=None, method="flat", wmin=None, wmax=None, zone=None, event="delivered", gst=0,
         st="third_party", conf="medium", status="indicative", notes="", rec=0,
         src="Third-party reference guides + user-supplied screenshots (indicative demo value)"):
    return dict(fee_type=ft, amount=amount, rate=rate, calculation_method=method, weight_min=wmin, weight_max=wmax, zone=zone,
                applicable_event=event, gst_applicable=gst, recoverable=rec, source=src, source_type=st, confidence=conf,
                status=status, effective_date="2026-01-01", notes=notes, enabled=1)

SHIP = {  # (min, max] grams -> {zone: rupees}  (illustrative)
 (0, 500): {"Local": 45, "Regional": 60, "National": 75}, (500, 1000): {"Local": 75, "Regional": 95, "National": 115},
 (1000, 2000): {"Local": 110, "Regional": 140, "National": 170}, (2000, 5000): {"Local": 190, "Regional": 240, "National": 290}}

def default_rule_specs():
    r = [_fee("commission", rate=0.0, method="percent_of_price", gst=1, notes="Demo assumption 0%; configurable. Verify in Supplier Panel."),
         _fee("platform_fee", amount=25, gst=1, st="model_assumption", conf="low", status="unconfirmed",
              src="Illustrative platform/service fee assumption", notes="Not treated as universally confirmed. Admin can disable.")]
    for (lo, hi), z in SHIP.items():
        for zone, amt in z.items():
            r.append(_fee("forward_shipping", amount=amt, wmin=lo, wmax=hi, zone=zone, gst=1, notes="Illustrative third-party/reference assumption — not an official universal tariff."))
            r.append(_fee("reverse_shipping", amount=amt, wmin=lo, wmax=hi, zone=zone, event="customer_return", gst=1, st="model_assumption", conf="low",
                          src="Model assumption (= forward slab)", notes="Verify against actual seller settlement."))
    r += [_fee("rto_reverse", amount=0, event="rto", gst=1, st="model_assumption", conf="low", status="unconfirmed",
               src="Illustrative assumption", notes="Illustrative assumption — verify against actual seller settlement."),
          _fee("rto_handling", amount=0, event="rto", gst=1, st="model_assumption", conf="low", status="unconfirmed", src="Model assumption"),
          _fee("gst", rate=0.18, method="percent_of_fee", st="third_party", conf="medium", notes="GST on applicable service/logistics fees only."),
          _fee("tcs", rate=0.01, method="percent_of_price", rec=1, st="model_assumption", conf="low", status="unconfirmed",
               src="Model assumption", notes="Rate has changed historically — verify current rate. Recoverable only if GST-registered."),
          _fee("tds", rate=0.001, method="percent_of_price", rec=1, st="model_assumption", conf="low", status="unconfirmed",
               src="Model assumption", notes="Verify current rate. Treated as recoverable (credit)."),
          _fee("late_dispatch_penalty", amount=50, event="late_dispatch", st="model_assumption", conf="low", src="Model assumption"),
          _fee("cancellation_penalty", amount=100, event="seller_cancellation", st="model_assumption", conf="low", src="Model assumption"),
          _fee("weight_discrepancy_charge", amount=40, event="weight_discrepancy", st="model_assumption", conf="low", src="Model assumption"),
          _fee("quality_penalty", amount=30, event="quality_issue", st="model_assumption", conf="low", src="Model assumption")]
    return r

def default_rules():
    return [dict(x, id=i + 1) for i, x in enumerate(default_rule_specs())]

FIXED_KURTI = [("Seller B", 299, 4.0, 430, .12, .11, .94), ("Seller C", 319, 4.2, 780, .09, .08, .96),
 ("Seller D", 329, 4.4, 2100, .07, .06, .97), ("Seller E", 339, 4.5, 5400, .06, .05, .98),
 ("Seller F", 349, 4.6, 9800, .05, .04, .99), ("Seller G", 359, 4.3, 3100, .07, .06, .98),
 ("Seller H", 379, 4.7, 14200, .04, .03, .99), ("Seller I", 399, 4.4, 6700, .06, .05, .98),
 ("Seller J", 289, 3.8, 210, .15, .14, .92), ("Seller K", 309, 4.1, 620, .11, .10, .95)]

def _orders(rng, base, q, price, med):
    idx = price / med
    e = 1.4 + 1.8 / (1 + math.exp(-8 * (idx - 1)))
    return max(1, int(base * (0.5 + q) * idx ** -e * rng.lognormal(0, .25)))

def _listing(rng, i, cat, sub, ptype, attrs, med, base, name, seller, q=None, fixed=None):
    q = rng.beta(2.6, 2.6) if q is None else q
    if fixed:
        price, rating, rev, ret, rto, ful = fixed
        q = float(np.clip((rating - 3.4) / 1.4, 0, 1))
    else:
        price = int(round(med * math.exp(rng.normal(.12 * (q - .5), .16)) / 10) * 10 - 1)
        rating = float(np.clip(3.4 + 1.4 * q + rng.normal(0, .2), 3.0, 4.9)); rev = int(math.exp(4.5 + 4.5 * q + rng.normal(0, .8)))
        ret = float(np.clip(.16 - .12 * q + rng.normal(0, .015), .02, .3)); rto = float(np.clip(.15 - .12 * q + rng.normal(0, .015), .02, .3))
        ful = float(np.clip(.90 + .09 * q + rng.normal(0, .01), .85, .995))
    return dict(competitor_id=f"C{i:04d}", seller_name=seller, product_name=name, category=cat, subcategory=sub, product_type=ptype,
                attributes=attrs, price=float(price), rating=round(rating, 1), review_count=int(rev), return_rate=round(ret, 3),
                rto_rate=round(rto, 3), fulfilment_rate=round(ful, 3),
                cancellation_rate=round(float(np.clip(.04 - .03 * q + rng.normal(0, .005), .002, .08)), 3),
                listing_age_days=int(rng.integers(30, 900)), est_monthly_orders=_orders(rng, base, q, price, med),
                delivery_zone=str(rng.choice(ZONES)), sponsored=int(rng.random() < .25), similarity=round(float(rng.uniform(.85, .99)), 2), is_synthetic=1)

def gen_competitors(seed=SEED):
    rng = np.random.default_rng(seed); out, i = [], 1
    types, med, base = CATS["Women's Fashion"]["Kurtis"]
    for s, p, r, rv, ret, rto, ful in FIXED_KURTI:
        out.append(_listing(rng, i, "Women's Fashion", "Kurtis", "Cotton Kurti", "Printed, Regular fit", med, base, "Printed Cotton Kurti", f"{s} (synthetic)", fixed=(p, r, rv, ret, rto, ful))); i += 1
    for k in range(22):
        out.append(_listing(rng, i, "Women's Fashion", "Kurtis", "Cotton Kurti", rng.choice(["Printed, Regular fit", "Printed, Straight fit", "Solid, Regular fit"]), med, base, "Cotton Kurti", f"Synthetic Seller K{k+1}")); i += 1
    for cat, subs in CATS.items():
        for sub, (types, med, base) in subs.items():
            for pt in types:
                if (cat, sub, pt) == ("Women's Fashion", "Kurtis", "Cotton Kurti"): continue
                for k in range(5):
                    out.append(_listing(rng, i, cat, sub, pt, rng.choice(["Printed", "Solid", "Premium", "Combo"]), med, base, pt, f"Synthetic Seller {i}")); i += 1
    return pd.DataFrame(out)

def gen_observations(n=900, seed=SEED):
    rng = np.random.default_rng(seed + 1); rows = []
    flat = [(c, s, v) for c, subs in CATS.items() for s, v in subs.items()]
    for _ in range(n):
        cat, sub, (types, med, base) = flat[rng.integers(len(flat))]
        q = rng.beta(2.6, 2.6); idx = math.exp(rng.normal(.1 * (q - .5), .18)); price = round(med * idx)
        month = int(rng.integers(1, 13)); season = 1 + .25 * math.cos((month - 10.5) / 12 * 2 * math.pi)
        ad = float(math.exp(rng.normal(7.5, .8))) if rng.random() < .7 else 0.0
        age = int(rng.integers(20, 900)); e = 1.4 + 1.8 / (1 + math.exp(-8 * (idx - 1))) + rng.normal(0, .2)
        orders = max(1, base * (.5 + q) * idx ** -e * (1 + .12 * math.log1p(ad / 1000)) * (.8 + .2 * min(age / 365, 1.5)) * season * rng.lognormal(0, .25))
        ctr = float(np.clip(rng.normal(.035 * (.7 + .6 * q), .005), .005, .1)); conv = float(np.clip(.05 * idx ** -.8 * (.7 + .6 * q) * rng.lognormal(0, .15), .005, .3))
        clicks = orders / conv; imp = clicks / ctr
        rows.append(dict(seller_id=int(rng.integers(1, 60)), product_id=int(rng.integers(1, 400)), date=f"2026-{month:02d}-{int(rng.integers(1, 28)):02d}",
            price=float(price), impressions=int(imp), clicks=int(clicks), orders=int(round(orders)),
            rating=round(float(np.clip(3.4 + 1.4 * q + rng.normal(0, .2), 3, 4.9)), 1), reviews=int(math.exp(4.5 + 4.5 * q + rng.normal(0, .8))),
            return_rate=float(np.clip(.16 - .12 * q + rng.normal(0, .02), .02, .3)), rto_rate=float(np.clip(.15 - .12 * q + rng.normal(0, .02), .02, .3)),
            fulfilment=float(np.clip(.90 + .09 * q + rng.normal(0, .01), .85, .995)), ad_spend=ad, category=cat, subcategory=sub,
            seasonality=season, price_index=idx, similarity=float(rng.uniform(.6, .99)), listing_age=age, ctr=ctr, conversion_rate=conv,
            orders_per_1000_impr=1000 * orders / max(imp, 1), ad_cost_per_order=ad / max(orders, 1)))
    return pd.DataFrame(rows)

_H = dict(expected_monthly_orders=None, return_rate=None, rto_rate=None, rating=None, review_count=None, fulfilment_rate=None,
          cancellation_rate=None, late_dispatch_rate=None, ad_spend_monthly=None)
DEMO_PRODUCTS = [  # synthetic example products for Demo Seller A (new-seller mode: no history)
    dict(product_name="Printed Cotton Kurti", total_cost=190, weight_g=350, current_price=349, zone="Regional", category="Women's Fashion",
         subcategory="Kurtis", product_type="Cotton Kurti", attributes="Printed, Regular fit", gst_registered=True, who="new", shop_price=320, unit_cost=None, **_H),
    dict(product_name="Men's Cotton T-Shirt", total_cost=150, weight_g=300, current_price=299, zone="Regional", category="Men's Fashion",
         subcategory="T-Shirts", product_type="Cotton T-Shirt", attributes="Solid", gst_registered=True, who="new", **_H),
    dict(product_name="Steel Water Bottle", total_cost=130, weight_g=450, current_price=249, zone="Regional", category="Home & Kitchen",
         subcategory="Kitchen Storage", product_type="Steel Bottle", attributes="Premium", gst_registered=True, who="new", **_H),
]
