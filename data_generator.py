"""
E-Commerce Business Data Generator
Generates realistic synthetic data for comprehensive BI analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

# ── Constants ────────────────────────────────────────────────────────────────
START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2024, 12, 31)
N_CUSTOMERS = 5_000
N_PRODUCTS  = 500
N_ORDERS    = 50_000

CATEGORIES = {
    "Electronics":      {"price_range": (50, 2000),  "cost_mult": 0.55, "weight": 0.20},
    "Fashion":          {"price_range": (15, 400),   "cost_mult": 0.35, "weight": 0.25},
    "Home & Garden":    {"price_range": (10, 800),   "cost_mult": 0.45, "weight": 0.15},
    "Sports":           {"price_range": (20, 600),   "cost_mult": 0.40, "weight": 0.12},
    "Beauty":           {"price_range": (8, 200),    "cost_mult": 0.30, "weight": 0.13},
    "Books & Media":    {"price_range": (5, 80),     "cost_mult": 0.25, "weight": 0.08},
    "Toys & Games":     {"price_range": (10, 300),   "cost_mult": 0.38, "weight": 0.07},
}

REGIONS = {
    "Lagos":    0.30, "Abuja": 0.15, "Kano": 0.10, "Port Harcourt": 0.10,
    "Ibadan":   0.08, "Enugu": 0.06, "Kaduna": 0.06, "Benin": 0.05,
    "Owerri":   0.05, "Warri": 0.05,
}

PAYMENT_METHODS = {
    "Card": 0.40, "Bank Transfer": 0.25, "USSD": 0.15,
    "Wallet": 0.12, "PayOnDelivery": 0.08,
}

ORDER_STATUSES = {
    "Delivered": 0.72, "Processing": 0.10, "Shipped": 0.08,
    "Cancelled": 0.06, "Returned": 0.04,
}

MARKETING_CHANNELS = {
    "Organic Search": 0.30, "Social Media": 0.25, "Email": 0.18,
    "Paid Search": 0.15, "Referral": 0.07, "Direct": 0.05,
}

CAMPAIGNS = [
    "Black Friday 2022", "Christmas 2022", "Valentine 2023",
    "Easter 2023", "Mid-Year Sale 2023", "Black Friday 2023",
    "Christmas 2023", "Valentine 2024", "Easter 2024",
    "Mid-Year Sale 2024", "Black Friday 2024", "Christmas 2024",
]


def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def weighted_choice(mapping: dict):
    keys   = list(mapping.keys())
    weights = list(mapping.values())
    return random.choices(keys, weights=weights, k=1)[0]


# ── Customers ────────────────────────────────────────────────────────────────
def generate_customers() -> pd.DataFrame:
    records = []
    for i in range(1, N_CUSTOMERS + 1):
        reg_date = random_date(START_DATE, END_DATE - timedelta(days=30))
        region   = weighted_choice(REGIONS)
        age      = int(np.random.normal(32, 10))
        age      = max(18, min(70, age))
        records.append({
            "customer_id":       f"C{i:05d}",
            "age":               age,
            "gender":            random.choice(["Male", "Female", "Other"]),
            "region":            region,
            "registration_date": reg_date,
            "channel":           weighted_choice(MARKETING_CHANNELS),
            "loyalty_tier":      random.choices(
                ["Bronze", "Silver", "Gold", "Platinum"],
                weights=[0.50, 0.30, 0.15, 0.05]
            )[0],
        })
    return pd.DataFrame(records)


# ── Products ─────────────────────────────────────────────────────────────────
def generate_products() -> pd.DataFrame:
    records = []
    cat_keys = list(CATEGORIES.keys())
    cat_wts  = [CATEGORIES[c]["weight"] for c in cat_keys]
    for i in range(1, N_PRODUCTS + 1):
        cat     = random.choices(cat_keys, weights=cat_wts, k=1)[0]
        lo, hi  = CATEGORIES[cat]["price_range"]
        price   = round(random.uniform(lo, hi), 2)
        cost    = round(price * CATEGORIES[cat]["cost_mult"] * random.uniform(0.85, 1.15), 2)
        records.append({
            "product_id":    f"P{i:04d}",
            "product_name":  f"{cat} Product {i}",
            "category":      cat,
            "price":         price,
            "cost":          cost,
            "stock":         random.randint(0, 500),
            "launch_date":   random_date(START_DATE - timedelta(days=365), END_DATE),
            "rating":        round(random.uniform(1.0, 5.0), 1),
            "review_count":  random.randint(0, 2000),
            "is_featured":   random.random() < 0.10,
        })
    return pd.DataFrame(records)


# ── Orders & Order Items ──────────────────────────────────────────────────────
def generate_orders_and_items(customers: pd.DataFrame, products: pd.DataFrame):
    customer_ids = customers["customer_id"].tolist()
    product_ids  = products["product_id"].tolist()
    product_map  = products.set_index("product_id")

    # Simulate seasonality — higher order volume in Q4
    def seasonal_weight(dt: datetime) -> float:
        month = dt.month
        if month == 11: return 2.5
        if month == 12: return 2.2
        if month in (1, 2): return 0.7
        return 1.0

    orders, items = [], []
    order_id = 1

    for _ in range(N_ORDERS):
        order_date = random_date(START_DATE, END_DATE)
        customer   = random.choice(customer_ids)
        status     = weighted_choice(ORDER_STATUSES)
        payment    = weighted_choice(PAYMENT_METHODS)
        discount   = round(random.choices([0, 5, 10, 15, 20, 25], weights=[0.50, 0.15, 0.15, 0.10, 0.06, 0.04])[0], 2)
        campaign   = random.choices([None] + CAMPAIGNS, weights=[0.60] + [0.4/len(CAMPAIGNS)]*len(CAMPAIGNS))[0]
        ship_days  = random.randint(1, 14) if status not in ("Processing",) else None
        delivery_date = (order_date + timedelta(days=ship_days)) if ship_days else None

        n_items   = random.choices([1, 2, 3, 4, 5], weights=[0.45, 0.28, 0.15, 0.08, 0.04])[0]
        chosen    = random.sample(product_ids, min(n_items, len(product_ids)))

        order_total = 0
        for pid in chosen:
            qty    = random.randint(1, 5)
            price  = product_map.loc[pid, "price"]
            subtotal = round(price * qty * (1 - discount / 100), 2)
            order_total += subtotal
            items.append({
                "item_id":    f"I{order_id:06d}{pid}",
                "order_id":   f"O{order_id:06d}",
                "product_id": pid,
                "quantity":   qty,
                "unit_price": price,
                "discount_pct": discount,
                "subtotal":   subtotal,
            })

        orders.append({
            "order_id":       f"O{order_id:06d}",
            "customer_id":    customer,
            "order_date":     order_date,
            "delivery_date":  delivery_date,
            "status":         status,
            "payment_method": payment,
            "discount_pct":   discount,
            "order_total":    round(order_total, 2),
            "campaign":       campaign,
            "shipping_region": random.choices(list(REGIONS.keys()), weights=list(REGIONS.values()))[0],
        })
        order_id += 1

    return pd.DataFrame(orders), pd.DataFrame(items)


# ── Reviews ───────────────────────────────────────────────────────────────────
def generate_reviews(orders: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    delivered = orders[orders["status"] == "Delivered"]["order_id"].tolist()
    delivered_items = items[items["order_id"].isin(delivered)]
    sample = delivered_items.sample(frac=0.40, random_state=42)

    reviews = []
    for i, row in enumerate(sample.itertuples(), 1):
        order = orders[orders["order_id"] == row.order_id].iloc[0]
        base  = random.gauss(4.0, 0.8)
        rating = max(1, min(5, round(base)))
        reviews.append({
            "review_id":   f"R{i:06d}",
            "order_id":    row.order_id,
            "product_id":  row.product_id,
            "customer_id": order["customer_id"],
            "rating":      rating,
            "sentiment":   "positive" if rating >= 4 else ("neutral" if rating == 3 else "negative"),
            "review_date": order["order_date"] + timedelta(days=random.randint(1, 14)),
        })
    return pd.DataFrame(reviews)


# ── Marketing Campaigns ────────────────────────────────────────────────────────
def generate_campaign_stats() -> pd.DataFrame:
    records = []
    for camp in CAMPAIGNS:
        year = int(camp.split()[-1])
        records.append({
            "campaign":       camp,
            "year":           year,
            "spend":          round(random.uniform(50_000, 500_000), 2),
            "impressions":    random.randint(500_000, 5_000_000),
            "clicks":         random.randint(10_000, 200_000),
            "conversions":    random.randint(500, 10_000),
            "revenue_attr":   round(random.uniform(500_000, 5_000_000), 2),
            "channel":        weighted_choice(MARKETING_CHANNELS),
        })
    return pd.DataFrame(records)


# ── Web Sessions ───────────────────────────────────────────────────────────────
def generate_sessions() -> pd.DataFrame:
    records = []
    for i in range(1, 200_001):
        dt = random_date(START_DATE, END_DATE)
        records.append({
            "session_id":    f"S{i:07d}",
            "session_date":  dt,
            "channel":       weighted_choice(MARKETING_CHANNELS),
            "device":        random.choices(["Mobile", "Desktop", "Tablet"], weights=[0.55, 0.38, 0.07])[0],
            "page_views":    random.randint(1, 20),
            "duration_sec":  random.randint(5, 1800),
            "bounced":       random.random() < 0.40,
            "converted":     random.random() < 0.08,
        })
    return pd.DataFrame(records)


# ── Inventory Snapshots ────────────────────────────────────────────────────────
def generate_inventory(products: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    sold = items.groupby("product_id")["quantity"].sum().reset_index()
    sold.columns = ["product_id", "units_sold"]
    inv = products[["product_id", "category", "stock", "cost"]].merge(sold, on="product_id", how="left")
    inv["units_sold"] = inv["units_sold"].fillna(0).astype(int)
    inv["ending_stock"] = (inv["stock"] - inv["units_sold"]).clip(lower=0)
    inv["inventory_value"] = inv["ending_stock"] * inv["cost"]
    return inv


# ── Main ───────────────────────────────────────────────────────────────────────
def generate_all(output_dir: str = "data") -> dict:
    os.makedirs(output_dir, exist_ok=True)
    print("Generating customers …")
    customers = generate_customers()

    print("Generating products …")
    products = generate_products()

    print("Generating orders & order items …")
    orders, items = generate_orders_and_items(customers, products)

    print("Generating reviews …")
    reviews = generate_reviews(orders, items)

    print("Generating campaign stats …")
    campaigns = generate_campaign_stats()

    print("Generating web sessions …")
    sessions = generate_sessions()

    print("Generating inventory …")
    inventory = generate_inventory(products, items)

    datasets = {
        "customers":  customers,
        "products":   products,
        "orders":     orders,
        "order_items": items,
        "reviews":    reviews,
        "campaigns":  campaigns,
        "sessions":   sessions,
        "inventory":  inventory,
    }

    for name, df in datasets.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved {path}  ({len(df):,} rows)")

    print("\nAll datasets generated successfully.")
    return datasets


if __name__ == "__main__":
    generate_all()
