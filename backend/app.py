import json, os
from typing import Optional, Dict, Any
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from . import db, service, seed, engines as E

app = FastAPI(title="PriceSense — Marketplace pricing prototype (synthetic data)")
FRONT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

class ProductIn(BaseModel):
    product_name: str = "Printed Cotton Kurti"
    category: str = "Women's Fashion"; subcategory: str = "Kurtis"; product_type: str = "Cotton Kurti"; attributes: str = "Printed, Regular fit"
    weight_g: float = Field(350, ge=0)
    current_price: float = Field(349, gt=0); gst_registered: bool = True; zone: str = "Regional"; zone_mix: Optional[Dict[str, float]] = None
    expected_monthly_orders: Optional[float] = None; return_rate: Optional[float] = None; rto_rate: Optional[float] = None
    rating: Optional[float] = None; review_count: Optional[int] = None; fulfilment_rate: Optional[float] = None
    cancellation_rate: Optional[float] = None; late_dispatch_rate: Optional[float] = None; ad_spend_monthly: Optional[float] = None
    constraints: Optional[Dict[str, Any]] = None
    shop_price: Optional[float] = None; unit_cost: Optional[float] = None  # optional offline bridge inputs (never stored in fee logic)

class FeeRuleIn(BaseModel):
    id: Optional[int] = None; fee_type: Optional[str] = None; amount: Optional[float] = None; rate: Optional[float] = None
    calculation_method: Optional[str] = None; weight_min: Optional[float] = None; weight_max: Optional[float] = None; zone: Optional[str] = None
    applicable_event: Optional[str] = None; gst_applicable: Optional[int] = None; recoverable: Optional[int] = None; source: Optional[str] = None
    source_type: Optional[str] = None; confidence: Optional[str] = None; status: Optional[str] = None; effective_date: Optional[str] = None
    notes: Optional[str] = None; enabled: Optional[int] = None

class ConfigIn(BaseModel):
    key: str; value: Dict[str, Any]

class PriceReq(ProductIn):
    price: Optional[float] = None

@app.on_event("startup")
def _start(): db.init()

def _inp(p: ProductIn):
    d = p.model_dump()
    d['product_cost'] = 0.0  # intentionally not requested from sellers in this prototype
    d['packaging_cost'] = 0.0  # packaging is not requested from sellers
    return d

@app.get("/api/demo-data")
def demo():
    return service.clean({"seller": db.rows("SELECT * FROM sellers")[0], "product": db.rows("SELECT * FROM products WHERE id=1")[0],
        "performance": db.rows("SELECT * FROM seller_performance WHERE seller_id=1")[0], "sources": db.rows("SELECT * FROM data_sources"),
        "counts": {"competitors": db.rows("SELECT COUNT(*) n FROM competitors")[0]["n"], "observations": db.rows("SELECT COUNT(*) n FROM orders")[0]["n"]},
        "notice": "Illustrative synthetic data for demonstration purposes."})

@app.get("/api/products")
def list_products():
    out = []
    for r in db.rows("SELECT id,name,payload FROM products WHERE seller_id=1 ORDER BY id"):
        try: pl = json.loads(r["payload"]) if r["payload"] else None
        except Exception: pl = None
        out.append({"id": r["id"], "name": r["name"], "payload": pl})
    return out

@app.post("/api/products")
def save_product(body: Dict[str, Any]):
    pid = body.get("id"); payload = json.dumps({k: v for k, v in body.items() if k != "id"})
    cols = (body.get("product_name") or "Untitled product", body.get("category"), body.get("subcategory"), body.get("product_type"), body.get("attributes"),
            body.get("total_cost"), 0.0, body.get("weight_g"), body.get("current_price"), payload)
    with db.conn() as c:
        if pid:
            c.execute("UPDATE products SET name=?,category=?,subcategory=?,product_type=?,attributes=?,product_cost=?,packaging_cost=?,weight_g=?,current_price=?,payload=? WHERE id=?", (*cols, pid))
            return {"id": pid}
        return {"id": c.execute("INSERT INTO products(seller_id,name,category,subcategory,product_type,attributes,product_cost,packaging_cost,weight_g,current_price,payload) VALUES(1,?,?,?,?,?,?,?,?,?,?)", cols).lastrowid}

@app.delete("/api/products/{pid}")
def delete_product(pid: int):
    with db.conn() as c: c.execute("DELETE FROM products WHERE id=? AND seller_id=1", (pid,))
    return {"deleted": pid}

@app.post("/api/price-analysis")
def price_analysis(p: ProductIn):
    r = service.analyze(_inp(p)); e = r["economics"]
    keys = ["estimated_settlement", "product_cost", "packaging_cost", "forward_shipping", "platform_fee", "commission", "gst_on_fees", "tax_deductions",
            "expected_return_cost", "expected_rto_cost", "expected_penalties", "advertising_cost", "expected_contribution", "contribution_margin"]
    return {"product": r["product"], "current_price": p.current_price, **{k: e[k] for k in keys}, "market_median": r["benchmark"]["stats"]["median"],
            "price_percentile": r["benchmark"]["price_percentile"], "seller_quality_score": r["seller_quality"]["score"],
            "competitive_visibility_index": r["cvi"]["score"], "confidence": r["confidence"], "economics": e, "warnings": r["warnings"]}

@app.post("/api/optimize-price")
def optimize_price(p: ProductIn): return service.analyze(_inp(p))["optimization"]

@app.post("/api/full-analysis")
def full(p: ProductIn): return service.analyze(_inp(p))

@app.post("/api/sensitivity-analysis")
def sens(p: ProductIn): return service.analyze(_inp(p))["sensitivity"]

@app.post("/api/seller-quality")
def sq(p: ProductIn): return service.analyze(_inp(p))["seller_quality"]

@app.post("/api/demand-prediction")
def demand(p: PriceReq):
    ctx = service.build_context(_inp(p)); price = p.price or p.current_price
    return service.clean({"price": price, "estimated_orders": ctx["rule_fn"](price), "calibrated_to_expected_orders": ctx["calibrated"],
                          "note": "Hypothetical demand from rule-based assumptions."})

@app.get("/api/market-benchmark")
def bench(category: str = "Women's Fashion", subcategory: str = "Kurtis", product_type: str = "Cotton Kurti", price: float = 349, attributes: str = "Printed, Regular fit"):
    cfg = db.load_cfg(); comps, lvl = E.select_comparables(db.rows("SELECT * FROM competitors"), dict(category=category, subcategory=subcategory, product_type=product_type, attributes=attributes), price, cfg)
    return service.clean({"stats": E.price_stats([c["price"] for c in comps]), "level": lvl, "price_percentile": E.price_percentile([c["price"] for c in comps], price), "price": price,
                          "label": "Synthetic competitors — illustrative synthetic data."})

@app.get("/api/competitors")
def competitors(category: Optional[str] = None, subcategory: Optional[str] = None, min_price: float = 0, max_price: float = 1e9, min_rating: float = 0,
                min_reviews: int = 0, max_return: float = 1, max_rto: float = 1):
    q, a = "SELECT * FROM competitors WHERE price BETWEEN ? AND ? AND rating>=? AND review_count>=? AND return_rate<=? AND rto_rate<=?", [min_price, max_price, min_rating, min_reviews, max_return, max_rto]
    for k, v in (("category", category), ("subcategory", subcategory)):
        if v: q += f" AND {k}=?"; a.append(v)
    return {"notice": "Illustrative synthetic data for demonstration purposes.", "rows": db.rows(q, a)}

@app.get("/api/dataset-summary")
def summary():
    r = db.rows("SELECT category, COUNT(*) n, AVG(price) avg_price, AVG(rating) avg_rating, AVG(review_count) avg_reviews, AVG(return_rate) avg_return, AVG(rto_rate) avg_rto FROM competitors GROUP BY category")
    return service.clean({"by_category": r, "total": sum(x["n"] for x in r), "notice": "Illustrative synthetic data for demonstration purposes."})

@app.get("/api/fee-rules")
def fee_rules(): return db.load_rules()

@app.post("/api/admin/fee-rules")
def upsert_rule(r: FeeRuleIn):
    d = {k: v for k, v in r.model_dump().items() if v is not None and k != "id"}
    with db.conn() as c:
        if r.id:
            if d: c.execute(f"UPDATE fee_rules SET {','.join(k + '=?' for k in d)}, updated_at=CURRENT_TIMESTAMP WHERE id=?", [*d.values(), r.id])
            return {"id": r.id}
        d.setdefault("calculation_method", "flat"); d.setdefault("source_type", "seller_reported"); d.setdefault("status", "seller_actual")
        return {"id": c.execute(f"INSERT INTO fee_rules({','.join(d)}) VALUES({','.join('?'*len(d))})", list(d.values())).lastrowid}

@app.get("/api/model-config")
def get_cfg(): return db.load_cfg()

@app.post("/api/admin/model-config")
def set_cfg(c: ConfigIn): db.save_cfg(c.key, c.value); return db.load_cfg()

@app.get("/api/model-metrics")
def metrics():
    P = db.load_cfg()["params"]
    return {"rule_model_form": "Orders(P) = BaseOrders x (P/Median)^-e(P) x QualityEffect x AdEffect x Seasonality; calibrated to your orders if you give them, otherwise estimated from similar listings. e moves from elasticity_below to elasticity_above around the median.",
            "params": {k: P[k] for k in ("elasticity_below", "elasticity_above", "quality_effect_floor", "quality_effect_span", "ad_gain", "seasonality")},
            "note": "Hypothetical demand model - not trained on real data."}

@app.post("/api/admin/regenerate-data")
def regen(seed_value: int = seed.SEED): db.regenerate(seed_value); return demo()

SELLER_ONLY = os.environ.get("SELLER_ONLY") == "1"  # set on public hosting: blocks admin + product writes

@app.middleware("http")
async def seller_only_guard(request, call_next):
    from fastapi.responses import JSONResponse
    pth = request.url.path
    if SELLER_ONLY and (pth.startswith("/api/admin/") or (pth.startswith("/api/products") and request.method != "GET")):
        return JSONResponse({"detail": "Disabled on the public seller demo"}, status_code=403)
    return await call_next(request)

@app.get("/api/mode")
def mode(): return {"seller_only": SELLER_ONLY}

app.mount("/static", StaticFiles(directory=FRONT), name="static")
@app.get("/")
def index(): return FileResponse(os.path.join(FRONT, "index.html"))
