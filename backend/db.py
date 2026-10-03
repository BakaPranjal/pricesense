import json, os, sqlite3
from . import engines, seed
DB_PATH = os.environ.get("PRICESENSE_DB", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pricesense.db"))
TS = "created_at TEXT DEFAULT CURRENT_TIMESTAMP"
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS sellers(id INTEGER PRIMARY KEY, name TEXT, is_synthetic INTEGER DEFAULT 1, {TS});
CREATE TABLE IF NOT EXISTS products(id INTEGER PRIMARY KEY, seller_id INTEGER, name TEXT, category TEXT, subcategory TEXT, product_type TEXT, attributes TEXT,
  product_cost REAL, packaging_cost REAL, weight_g REAL, current_price REAL, payload TEXT, {TS});
CREATE TABLE IF NOT EXISTS listings(id INTEGER PRIMARY KEY, seller_id INTEGER, product_id INTEGER, price REAL, {TS});
CREATE TABLE IF NOT EXISTS competitors(id INTEGER PRIMARY KEY AUTOINCREMENT, competitor_id TEXT, seller_name TEXT, product_name TEXT, category TEXT, subcategory TEXT,
  product_type TEXT, attributes TEXT, price REAL, rating REAL, review_count INTEGER, return_rate REAL, rto_rate REAL, fulfilment_rate REAL, cancellation_rate REAL,
  listing_age_days INTEGER, est_monthly_orders INTEGER, delivery_zone TEXT, sponsored INTEGER, similarity REAL, is_synthetic INTEGER, {TS});
CREATE TABLE IF NOT EXISTS fee_rules(id INTEGER PRIMARY KEY AUTOINCREMENT, fee_type TEXT, amount REAL, rate REAL, calculation_method TEXT, weight_min REAL, weight_max REAL,
  zone TEXT, applicable_event TEXT, gst_applicable INTEGER, recoverable INTEGER DEFAULT 0, source TEXT, source_type TEXT, confidence TEXT, status TEXT, effective_date TEXT,
  notes TEXT, enabled INTEGER DEFAULT 1, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY AUTOINCREMENT, seller_id INTEGER, product_id INTEGER, date TEXT, price REAL, impressions INTEGER, clicks INTEGER,
  orders INTEGER, rating REAL, reviews INTEGER, return_rate REAL, rto_rate REAL, fulfilment REAL, ad_spend REAL, category TEXT, subcategory TEXT, seasonality REAL,
  price_index REAL, similarity REAL, listing_age INTEGER, ctr REAL, conversion_rate REAL, orders_per_1000_impr REAL, ad_cost_per_order REAL);
CREATE TABLE IF NOT EXISTS seller_performance(id INTEGER PRIMARY KEY, seller_id INTEGER, rating REAL, review_count INTEGER, return_rate REAL, rto_rate REAL,
  cancellation_rate REAL, late_dispatch_rate REAL, fulfilment_rate REAL, monthly_orders REAL, ad_spend_monthly REAL, is_synthetic INTEGER DEFAULT 1, {TS});
CREATE TABLE IF NOT EXISTS market_benchmarks(id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT, subcategory TEXT, product_type TEXT, n INTEGER, stats TEXT, {TS});
CREATE TABLE IF NOT EXISTS model_configs(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS price_simulations(id INTEGER PRIMARY KEY AUTOINCREMENT, product_id INTEGER, inputs TEXT, outputs TEXT, {TS});
CREATE TABLE IF NOT EXISTS data_sources(id INTEGER PRIMARY KEY, name TEXT, source_type TEXT, description TEXT, {TS});
"""
def conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH); c.row_factory = sqlite3.Row; return c

def rows(sql, args=()):
    with conn() as c: return [dict(r) for r in c.execute(sql, args).fetchall()]

def init():
    with conn() as c:
        c.executescript(SCHEMA)
        if not c.execute("SELECT 1 FROM sellers").fetchone():
            c.execute("INSERT INTO sellers(id,name) VALUES(1,'Demo Seller A')")
            c.execute("INSERT INTO products(id,seller_id,name,category,subcategory,product_type,attributes,product_cost,packaging_cost,weight_g,current_price) VALUES(1,1,'Printed Cotton Kurti',?,?,?,?,180,10,350,349)",
                      ("Women's Fashion", "Kurtis", "Cotton Kurti", "Printed, Regular fit"))
            c.execute("INSERT INTO listings(seller_id,product_id,price) VALUES(1,1,349)")
            c.execute("INSERT INTO seller_performance(seller_id,rating,review_count,return_rate,rto_rate,cancellation_rate,late_dispatch_rate,fulfilment_rate,monthly_orders,ad_spend_monthly) VALUES(1,4.3,1240,.08,.07,.015,.02,.97,220,8000)")
            for n, t, d in [("Shiprocket Meesho shipping-fee guide", "third_party", "Reference for initial shipping assumptions (values here are illustrative, not copied as official)."),
                            ("Blooprint Meesho seller-fee guide", "third_party", "Reference for initial fee assumptions."), ("User-supplied screenshots", "seller_reported", "Reference material for fee/penalty categories."),
                            ("Synthetic generator (seed 42)", "model_assumption", "Illustrative synthetic data for demonstration purposes.")]:
                c.execute("INSERT INTO data_sources(name,source_type,description) VALUES(?,?,?)", (n, t, d))
        if not c.execute("SELECT 1 FROM fee_rules").fetchone():
            for r in seed.default_rule_specs():
                c.execute(f"INSERT INTO fee_rules({','.join(r)}) VALUES({','.join('?' * len(r))})", list(r.values()))
        for k in ("params", "quality_weights", "cvi_weights"):
            if not c.execute("SELECT 1 FROM model_configs WHERE key=?", (k,)).fetchone():
                c.execute("INSERT INTO model_configs(key,value) VALUES(?,?)", (k, json.dumps(engines.make_cfg()[k])))
    with conn() as c:
        for d in seed.DEMO_PRODUCTS:
            r = c.execute("SELECT id FROM products WHERE seller_id=1 AND name=?", (d["product_name"],)).fetchone()
            pl = json.dumps(d)
            if r: c.execute("UPDATE products SET payload=? WHERE id=? AND payload IS NULL", (pl, r[0]))
            else: c.execute("INSERT INTO products(seller_id,name,category,subcategory,product_type,attributes,product_cost,packaging_cost,weight_g,current_price,payload) VALUES(1,?,?,?,?,?,?,0,?,?,?)",
                            (d["product_name"], d["category"], d["subcategory"], d["product_type"], d["attributes"], d["total_cost"], d["weight_g"], d["current_price"], pl))
    if not rows("SELECT 1 FROM competitors LIMIT 1"): regenerate(seed.SEED)

def regenerate(sd):
    with conn() as c:
        c.execute("DELETE FROM competitors"); c.execute("DELETE FROM orders")
        seed.gen_competitors(sd).to_sql("competitors", c, if_exists="append", index=False)
        seed.gen_observations(900, sd).to_sql("orders", c, if_exists="append", index=False)

def load_rules(): return rows("SELECT * FROM fee_rules")
def load_cfg():
    return engines.make_cfg({r["key"]: json.loads(r["value"]) for r in rows("SELECT key,value FROM model_configs")})
def save_cfg(key, value):
    cur = json.loads((rows("SELECT value FROM model_configs WHERE key=?", (key,)) or [{"value": "{}"}])[0]["value"]); cur.update(value)
    with conn() as c: c.execute("INSERT OR REPLACE INTO model_configs(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP)", (key, json.dumps(cur)))
