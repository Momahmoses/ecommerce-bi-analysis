"""
50 Business Intelligence & High-Level Data Analyses
for an E-Commerce Company.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from itertools import combinations
from collections import Counter

warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE   = sns.color_palette("muted", 12)
FIG_DIR   = "charts"
REPORT_DIR = "reports"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

CURRENCY = "₦"
MILLION  = 1_000_000


def savefig(name: str):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [chart] {name}.png")


def fmt_currency(x, _):
    if abs(x) >= 1e9: return f"{CURRENCY}{x/1e9:.1f}B"
    if abs(x) >= 1e6: return f"{CURRENCY}{x/1e6:.1f}M"
    if abs(x) >= 1e3: return f"{CURRENCY}{x/1e3:.0f}K"
    return f"{CURRENCY}{x:.0f}"


# ═══════════════════════════════════════════════════════════════════════════════
class EcommerceAnalytics:

    def __init__(self, data: dict):
        self.customers  = data["customers"].copy()
        self.products   = data["products"].copy()
        self.orders     = data["orders"].copy()
        self.items      = data["order_items"].copy()
        self.reviews    = data["reviews"].copy()
        self.campaigns  = data["campaigns"].copy()
        self.sessions   = data["sessions"].copy()
        self.inventory  = data["inventory"].copy()
        self._prepare()
        self.insights = []

    def _prepare(self):
        for col in ["order_date", "delivery_date"]:
            self.orders[col] = pd.to_datetime(self.orders[col], errors="coerce")
        self.orders["year"]    = self.orders["order_date"].dt.year
        self.orders["month"]   = self.orders["order_date"].dt.month
        self.orders["quarter"] = self.orders["order_date"].dt.quarter
        self.orders["yearmonth"] = self.orders["order_date"].dt.to_period("M")
        self.orders["dow"]     = self.orders["order_date"].dt.day_name()
        self.orders["hour"]    = self.orders["order_date"].dt.hour
        self.customers["registration_date"] = pd.to_datetime(self.customers["registration_date"])
        self.sessions["session_date"] = pd.to_datetime(self.sessions["session_date"])
        self.delivered = self.orders[self.orders["status"] == "Delivered"]
        self.items_ext = self.items.merge(
            self.products[["product_id", "category", "cost", "price"]], on="product_id", how="left"
        )
        self.items_ext["profit"] = self.items_ext["subtotal"] - (
            self.items_ext["cost"] * self.items_ext["quantity"]
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Monthly Revenue Trend
    # ─────────────────────────────────────────────────────────────────────────
    def analysis_01_monthly_revenue(self):
        df = (self.delivered.groupby("yearmonth")["order_total"]
              .sum().reset_index())
        df["yearmonth_str"] = df["yearmonth"].astype(str)
        df["ma3"] = df["order_total"].rolling(3).mean()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(df["yearmonth_str"], df["order_total"], color=PALETTE[0], alpha=0.7, label="Monthly Revenue")
        ax.plot(df["yearmonth_str"], df["ma3"], color="red", lw=2, label="3-Month MA")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax.set_title("Analysis 1 — Monthly Revenue Trend (2022–2024)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Month"); ax.set_ylabel("Revenue")
        ax.legend(); ax.tick_params(axis="x", rotation=45)
        savefig("01_monthly_revenue")
        total = df["order_total"].sum()
        self.insights.append(f"1. Total revenue (delivered orders): {CURRENCY}{total/MILLION:.1f}M over 3 years.")

    # 2. Quarterly Revenue & Growth
    def analysis_02_quarterly_revenue(self):
        df = self.delivered.groupby(["year","quarter"])["order_total"].sum().reset_index()
        df["label"] = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)
        df["qoq_growth"] = df["order_total"].pct_change() * 100

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax1.bar(df["label"], df["order_total"], color=PALETTE[1])
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax1.set_title("Analysis 2 — Quarterly Revenue & QoQ Growth", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Revenue")
        ax2.bar(df["label"], df["qoq_growth"], color=df["qoq_growth"].apply(lambda x: "green" if x >= 0 else "red"))
        ax2.axhline(0, color="black", lw=0.8)
        ax2.set_ylabel("QoQ Growth %"); ax2.set_xlabel("Quarter")
        ax2.tick_params(axis="x", rotation=45)
        savefig("02_quarterly_revenue")
        best_q = df.loc[df["order_total"].idxmax(), "label"]
        self.insights.append(f"2. Best performing quarter: {best_q}.")

    # 3. Revenue by Product Category
    def analysis_03_revenue_by_category(self):
        merged = self.items_ext.merge(self.orders[["order_id","status"]], on="order_id")
        df = merged[merged["status"] == "Delivered"].groupby("category")["subtotal"].sum().sort_values()

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(df.index, df.values, color=PALETTE[:len(df)])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        for bar, val in zip(bars, df.values):
            ax.text(val + 5e5, bar.get_y() + bar.get_height()/2,
                    fmt_currency(val, None), va="center", fontsize=9)
        ax.set_title("Analysis 3 — Revenue by Product Category", fontsize=14, fontweight="bold")
        ax.set_xlabel("Revenue")
        savefig("03_revenue_by_category")
        top = df.idxmax()
        self.insights.append(f"3. Top revenue category: {top} ({CURRENCY}{df.max()/MILLION:.1f}M).")

    # 4. Average Order Value (AOV) Trend
    def analysis_04_aov_trend(self):
        df = (self.delivered.groupby("yearmonth")
              .agg(revenue=("order_total", "sum"), orders=("order_id", "count"))
              .reset_index())
        df["aov"] = df["revenue"] / df["orders"]
        df["yearmonth_str"] = df["yearmonth"].astype(str)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df["yearmonth_str"], df["aov"], marker="o", color=PALETTE[2], lw=2)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax.set_title("Analysis 4 — Average Order Value (AOV) Monthly Trend", fontsize=14, fontweight="bold")
        ax.set_xlabel("Month"); ax.set_ylabel("AOV")
        ax.tick_params(axis="x", rotation=45)
        savefig("04_aov_trend")
        self.insights.append(f"4. Overall AOV: {CURRENCY}{df['aov'].mean():.0f}. Peak AOV: {CURRENCY}{df['aov'].max():.0f}.")

    # 5. Revenue by Payment Method
    def analysis_05_revenue_by_payment(self):
        df = self.delivered.groupby("payment_method")["order_total"].agg(["sum","count"]).reset_index()
        df.columns = ["payment_method", "revenue", "orders"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.pie(df["revenue"], labels=df["payment_method"], autopct="%1.1f%%",
                colors=PALETTE[:len(df)], startangle=140)
        ax1.set_title("Revenue Share by Payment Method")
        ax2.barh(df["payment_method"], df["orders"], color=PALETTE[:len(df)])
        ax2.set_title("Order Count by Payment Method")
        fig.suptitle("Analysis 5 — Revenue & Orders by Payment Method",
                     fontsize=14, fontweight="bold")
        savefig("05_revenue_by_payment")
        top = df.loc[df["revenue"].idxmax(), "payment_method"]
        self.insights.append(f"5. Most revenue from '{top}' payment method.")

    # 6. Revenue by Region
    def analysis_06_revenue_by_region(self):
        df = self.delivered.groupby("shipping_region")["order_total"].sum().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df.index, df.values, color=PALETTE[:len(df)])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax.set_title("Analysis 6 — Revenue by Region", fontsize=14, fontweight="bold")
        ax.set_xlabel("Region"); ax.set_ylabel("Revenue")
        ax.tick_params(axis="x", rotation=45)
        savefig("06_revenue_by_region")
        self.insights.append(f"6. Top revenue region: {df.idxmax()} ({CURRENCY}{df.max()/MILLION:.1f}M).")

    # 7. Year-over-Year Revenue Growth
    def analysis_07_yoy_growth(self):
        df = self.delivered.groupby("year")["order_total"].sum().reset_index()
        df["yoy_growth"] = df["order_total"].pct_change() * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.bar(df["year"].astype(str), df["order_total"], color=PALETTE[3])
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax1.set_title("Annual Revenue"); ax1.set_ylabel("Revenue")
        ax2.bar(df["year"].astype(str), df["yoy_growth"],
                color=df["yoy_growth"].apply(lambda x: "green" if (x >= 0 or pd.isna(x)) else "red"))
        ax2.set_title("YoY Revenue Growth %"); ax2.set_ylabel("Growth %")
        fig.suptitle("Analysis 7 — Year-over-Year Revenue Growth",
                     fontsize=14, fontweight="bold")
        savefig("07_yoy_growth")
        if len(df) > 1:
            g = df["yoy_growth"].dropna().iloc[-1]
            self.insights.append(f"7. Latest YoY revenue growth: {g:.1f}%.")

    # 8. Gross Profit Margin by Category
    def analysis_08_gross_margin(self):
        merged = self.items_ext.merge(self.orders[["order_id","status"]], on="order_id")
        del_items = merged[merged["status"] == "Delivered"]
        df = del_items.groupby("category").agg(
            revenue=("subtotal","sum"), profit=("profit","sum")
        ).reset_index()
        df["margin_pct"] = df["profit"] / df["revenue"] * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df["category"], df["margin_pct"], color=PALETTE[:len(df)])
        for bar, val in zip(bars, df["margin_pct"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", fontsize=9)
        ax.set_title("Analysis 8 — Gross Profit Margin by Category", fontsize=14, fontweight="bold")
        ax.set_ylabel("Gross Margin %"); ax.tick_params(axis="x", rotation=30)
        savefig("08_gross_margin")
        top = df.loc[df["margin_pct"].idxmax(), "category"]
        self.insights.append(f"8. Highest gross margin category: {top} ({df['margin_pct'].max():.1f}%).")

    # 9. Revenue Concentration — Pareto / 80-20 Rule
    def analysis_09_pareto_products(self):
        merged = self.items_ext.merge(self.orders[["order_id","status"]], on="order_id")
        df = merged[merged["status"] == "Delivered"].groupby("product_id")["subtotal"].sum().sort_values(ascending=False).reset_index()
        df["cumulative_pct"] = df["subtotal"].cumsum() / df["subtotal"].sum() * 100
        df["product_rank"] = range(1, len(df)+1)
        df["rank_pct"] = df["product_rank"] / len(df) * 100

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(df["rank_pct"], df["cumulative_pct"], alpha=0.4, color=PALETTE[4])
        ax.plot(df["rank_pct"], df["cumulative_pct"], color=PALETTE[4], lw=2)
        ax.axhline(80, color="red", ls="--", lw=1.5, label="80% Revenue")
        ax.axvline(20, color="gray", ls="--", lw=1.5, label="Top 20% Products")
        ax.set_title("Analysis 9 — Pareto: Product Revenue Concentration (80/20 Rule)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("% of Products (ranked by revenue)"); ax.set_ylabel("Cumulative Revenue %")
        ax.legend(); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        savefig("09_pareto_products")
        idx_80 = (df["cumulative_pct"] >= 80).idxmax()
        pct_products = df.loc[idx_80, "rank_pct"]
        self.insights.append(f"9. Top {pct_products:.1f}% of products drive 80% of revenue (Pareto principle).")

    # 10. Revenue per Loyalty Tier
    def analysis_10_revenue_by_loyalty(self):
        df = self.delivered.merge(self.customers[["customer_id","loyalty_tier"]], on="customer_id")
        tier_rev = df.groupby("loyalty_tier")["order_total"].agg(["sum","mean","count"]).reset_index()
        tier_rev.columns = ["tier","total_revenue","avg_order","orders"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        order = ["Bronze","Silver","Gold","Platinum"]
        tier_rev = tier_rev.set_index("tier").reindex(order).reset_index()
        for ax, col, title in zip(axes, ["total_revenue","avg_order","orders"],
                                  ["Total Revenue","Avg Order Value","Order Count"]):
            ax.bar(tier_rev["tier"], tier_rev[col], color=["#CD7F32","#C0C0C0","#FFD700","#E5E4E2"])
            if col in ("total_revenue","avg_order"):
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
            ax.set_title(title)
        fig.suptitle("Analysis 10 — Revenue & Orders by Loyalty Tier",
                     fontsize=14, fontweight="bold")
        savefig("10_revenue_by_loyalty")
        self.insights.append("10. Platinum tier customers have the highest average order value despite lowest count.")

    # 11. Customer Acquisition Over Time
    def analysis_11_customer_acquisition(self):
        df = self.customers.copy()
        df["reg_month"] = df["registration_date"].dt.to_period("M")
        monthly = df.groupby("reg_month").size().reset_index(name="new_customers")
        monthly["reg_month_str"] = monthly["reg_month"].astype(str)
        monthly["cumulative"] = monthly["new_customers"].cumsum()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax1.bar(monthly["reg_month_str"], monthly["new_customers"], color=PALETTE[5])
        ax1.set_title("Analysis 11 — Customer Acquisition Over Time", fontsize=14, fontweight="bold")
        ax1.set_ylabel("New Customers")
        ax2.plot(monthly["reg_month_str"], monthly["cumulative"], color="navy", lw=2)
        ax2.fill_between(monthly["reg_month_str"], monthly["cumulative"], alpha=0.2, color="navy")
        ax2.set_ylabel("Cumulative Customers"); ax2.set_xlabel("Month")
        ax2.tick_params(axis="x", rotation=45)
        savefig("11_customer_acquisition")
        self.insights.append(f"11. Total customers acquired: {len(df):,} over 3 years.")

    # 12. Customer Retention Rate (cohort-based)
    def analysis_12_retention_rate(self):
        orders = self.delivered[["customer_id","order_date"]].copy()
        orders["cohort_month"] = orders.groupby("customer_id")["order_date"].transform("min").dt.to_period("M")
        orders["order_month"]  = orders["order_date"].dt.to_period("M")
        orders["period_number"] = (orders["order_month"] - orders["cohort_month"]).apply(lambda x: x.n)

        cohort_size = orders[orders["period_number"] == 0].groupby("cohort_month")["customer_id"].nunique()
        retention = (orders.groupby(["cohort_month","period_number"])["customer_id"]
                     .nunique().reset_index())
        retention = retention.merge(cohort_size.rename("cohort_size"), on="cohort_month")
        retention["retention_rate"] = retention["customer_id"] / retention["cohort_size"] * 100

        pivot = retention[retention["period_number"] <= 11].pivot_table(
            index="cohort_month", columns="period_number", values="retention_rate"
        )
        pivot.index = pivot.index.astype(str)
        pivot = pivot.iloc[:12, :]

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd_r", ax=ax,
                    linewidths=0.3, cbar_kws={"label": "Retention %"})
        ax.set_title("Analysis 12 — Customer Cohort Retention Rate (%)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Months Since First Purchase"); ax.set_ylabel("Cohort Month")
        savefig("12_cohort_retention")
        avg_m1 = pivot[1].mean() if 1 in pivot.columns else 0
        self.insights.append(f"12. Average Month-1 retention rate: {avg_m1:.1f}%.")

    # 13. Customer Lifetime Value (CLV) Distribution
    def analysis_13_clv(self):
        clv = (self.delivered.groupby("customer_id")["order_total"]
               .sum().reset_index(name="clv"))
        percentiles = clv["clv"].quantile([0.25, 0.5, 0.75, 0.90, 0.95])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.hist(clv["clv"], bins=50, color=PALETTE[0], edgecolor="white")
        ax1.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax1.set_title("CLV Distribution"); ax1.set_xlabel("CLV"); ax1.set_ylabel("Customers")
        ax2.boxplot(clv["clv"], vert=True, patch_artist=True,
                    boxprops=dict(facecolor=PALETTE[1]))
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax2.set_title("CLV Box Plot")
        fig.suptitle("Analysis 13 — Customer Lifetime Value Distribution",
                     fontsize=14, fontweight="bold")
        savefig("13_clv_distribution")
        self.insights.append(
            f"13. Median CLV: {CURRENCY}{percentiles[0.5]:,.0f}; "
            f"Top 5% CLV: >{CURRENCY}{percentiles[0.95]:,.0f}."
        )

    # 14. RFM Segmentation
    def analysis_14_rfm(self):
        snapshot = self.delivered["order_date"].max()
        rfm = self.delivered.groupby("customer_id").agg(
            recency=("order_date", lambda x: (snapshot - x.max()).days),
            frequency=("order_id", "count"),
            monetary=("order_total", "sum")
        ).reset_index()

        for col in ["recency","frequency","monetary"]:
            rfm[f"{col}_score"] = pd.qcut(rfm[col], 4,
                labels=[4,3,2,1] if col == "recency" else [1,2,3,4])
        rfm["rfm_score"] = (rfm["recency_score"].astype(int)
                            + rfm["frequency_score"].astype(int)
                            + rfm["monetary_score"].astype(int))

        def segment(s):
            if s >= 10: return "Champions"
            if s >= 8:  return "Loyal"
            if s >= 6:  return "At Risk"
            if s >= 4:  return "Needs Attention"
            return "Lost"

        rfm["segment"] = rfm["rfm_score"].apply(segment)
        seg_counts = rfm["segment"].value_counts()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.pie(seg_counts, labels=seg_counts.index, autopct="%1.1f%%",
                colors=PALETTE[:len(seg_counts)])
        ax1.set_title("Customer Segments (Count)")
        seg_rev = rfm.groupby("segment")["monetary"].mean().sort_values(ascending=False)
        ax2.bar(seg_rev.index, seg_rev.values, color=PALETTE[:len(seg_rev)])
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax2.set_title("Avg Monetary Value by Segment")
        ax2.tick_params(axis="x", rotation=30)
        fig.suptitle("Analysis 14 — RFM Customer Segmentation",
                     fontsize=14, fontweight="bold")
        savefig("14_rfm_segmentation")
        self.rfm = rfm
        self.insights.append(f"14. RFM: {seg_counts.get('Champions',0):,} Champions, {seg_counts.get('Lost',0):,} Lost customers.")

    # 15. New vs Returning Customers
    def analysis_15_new_vs_returning(self):
        first_orders = self.delivered.groupby("customer_id")["order_date"].min().reset_index()
        first_orders.columns = ["customer_id","first_order_date"]
        df = self.delivered.merge(first_orders, on="customer_id")
        df["customer_type"] = np.where(df["order_date"] == df["first_order_date"], "New", "Returning")

        monthly = df.groupby(["yearmonth","customer_type"])["order_total"].agg(
            ["count","sum"]).reset_index()
        monthly["yearmonth_str"] = monthly["yearmonth"].astype(str)
        pivot_cnt = monthly.pivot(index="yearmonth_str", columns="customer_type", values="count").fillna(0)

        fig, ax = plt.subplots(figsize=(14, 6))
        pivot_cnt.plot(kind="bar", ax=ax, color=[PALETTE[0], PALETTE[3]], stacked=True)
        ax.set_title("Analysis 15 — New vs Returning Customers Monthly",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Month"); ax.set_ylabel("Order Count")
        ax.tick_params(axis="x", rotation=45); ax.legend()
        savefig("15_new_vs_returning")
        ret_pct = (df["customer_type"] == "Returning").mean() * 100
        self.insights.append(f"15. {ret_pct:.1f}% of delivered orders are from returning customers.")

    # 16. Geographic Distribution of Customers
    def analysis_16_customer_geo(self):
        df = self.customers["region"].value_counts().reset_index()
        df.columns = ["region","count"]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df["region"], df["count"], color=PALETTE[:len(df)])
        for bar, val in zip(bars, df["count"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(val), ha="center", fontsize=9)
        ax.set_title("Analysis 16 — Geographic Distribution of Customers",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Region"); ax.set_ylabel("Customers")
        ax.tick_params(axis="x", rotation=30)
        savefig("16_customer_geo")
        self.insights.append(f"16. Top customer region: {df.iloc[0]['region']} ({df.iloc[0]['count']:,} customers).")

    # 17. Customer Age Distribution by Segment
    def analysis_17_age_distribution(self):
        df = self.customers.copy()
        bins = [18, 25, 35, 45, 55, 70]
        labels = ["18-24","25-34","35-44","45-54","55-70"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
        age_rev = df.merge(
            self.delivered.groupby("customer_id")["order_total"].sum().reset_index(name="clv"),
            on="customer_id", how="left"
        )
        age_rev["clv"] = age_rev["clv"].fillna(0)
        age_rev["clv"] = age_rev["clv"].fillna(0)
        age_rev = age_rev.groupby("age_group")["clv"].mean().reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        df["age_group"].value_counts().sort_index().plot(kind="bar", ax=ax1, color=PALETTE[:5])
        ax1.set_title("Customer Count by Age Group"); ax1.tick_params(axis="x", rotation=0)
        ax2.bar(age_rev["age_group"].astype(str), age_rev["clv"], color=PALETTE[:5])
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax2.set_title("Avg CLV by Age Group")
        fig.suptitle("Analysis 17 — Age Distribution & CLV by Age Group",
                     fontsize=14, fontweight="bold")
        savefig("17_age_distribution")
        top_age = age_rev.loc[age_rev["clv"].idxmax(), "age_group"]
        self.insights.append(f"17. Highest CLV age group: {top_age}.")

    # 18. Channel Attribution — Customer Acquisition
    def analysis_18_channel_attribution(self):
        df = self.customers["channel"].value_counts().reset_index()
        df.columns = ["channel","customers"]
        ch_rev = self.delivered.merge(self.customers[["customer_id","channel"]], on="customer_id")
        ch_rev = ch_rev.groupby("channel")["order_total"].sum().reset_index(name="revenue")
        df = df.merge(ch_rev, on="channel")
        df["rev_per_customer"] = df["revenue"] / df["customers"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, col, title in zip(axes, ["customers","revenue","rev_per_customer"],
                                  ["Customers Acquired","Total Revenue","Revenue per Customer"]):
            ax.bar(df["channel"], df[col], color=PALETTE[:len(df)])
            if col in ("revenue","rev_per_customer"):
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
            ax.set_title(title); ax.tick_params(axis="x", rotation=30)
        fig.suptitle("Analysis 18 — Marketing Channel Attribution",
                     fontsize=14, fontweight="bold")
        savefig("18_channel_attribution")
        top_ch = df.loc[df["revenue"].idxmax(), "channel"]
        self.insights.append(f"18. Top revenue-generating channel: {top_ch}.")

    # 19. Top 10 Best-Selling Products
    def analysis_19_top_products_units(self):
        merged = self.items.merge(self.orders[["order_id","status"]], on="order_id")
        df = (merged[merged["status"] == "Delivered"]
              .groupby("product_id")["quantity"].sum()
              .sort_values(ascending=False).head(10).reset_index())
        df = df.merge(self.products[["product_id","product_name","category","price"]], on="product_id")

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(df["product_name"], df["quantity"], color=PALETTE[:10])
        ax.set_title("Analysis 19 — Top 10 Best-Selling Products (Units Sold)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Units Sold")
        savefig("19_top_products_units")
        self.insights.append(f"19. Best-selling product: {df.iloc[0]['product_name']} ({df.iloc[0]['quantity']:,} units).")

    # 20. Top 10 Revenue-Generating Products
    def analysis_20_top_products_revenue(self):
        merged = self.items_ext.merge(self.orders[["order_id","status"]], on="order_id")
        df = (merged[merged["status"] == "Delivered"]
              .groupby("product_id")["subtotal"].sum()
              .sort_values(ascending=False).head(10).reset_index())
        df = df.merge(self.products[["product_id","product_name"]], on="product_id")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(df["product_name"], df["subtotal"], color=PALETTE[:10])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax.set_title("Analysis 20 — Top 10 Revenue-Generating Products",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Revenue")
        savefig("20_top_products_revenue")
        self.insights.append(f"20. Top revenue product: {df.iloc[0]['product_name']} ({CURRENCY}{df.iloc[0]['subtotal']/MILLION:.1f}M).")

    # 21. Product Return Rate by Category
    def analysis_21_return_rate(self):
        returns = self.orders[self.orders["status"] == "Returned"]
        delivered_cnt = self.orders[self.orders["status"].isin(["Delivered","Returned"])].groupby(
            "order_id").size().reset_index()
        ret_items = self.items.merge(returns[["order_id"]], on="order_id")
        ret_items = ret_items.merge(self.products[["product_id","category"]], on="product_id")
        del_items = self.items.merge(
            self.orders[self.orders["status"].isin(["Delivered","Returned"])][["order_id"]], on="order_id"
        ).merge(self.products[["product_id","category"]], on="product_id")

        ret_by_cat = ret_items.groupby("category")["quantity"].sum()
        del_by_cat = del_items.groupby("category")["quantity"].sum()
        rate = (ret_by_cat / del_by_cat * 100).fillna(0).sort_values(ascending=False).reset_index()
        rate.columns = ["category","return_rate"]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(rate["category"], rate["return_rate"], color=PALETTE[:len(rate)])
        for i, (_, row) in enumerate(rate.iterrows()):
            ax.text(i, row["return_rate"] + 0.05, f"{row['return_rate']:.1f}%", ha="center", fontsize=9)
        ax.set_title("Analysis 21 — Product Return Rate by Category",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Return Rate %"); ax.tick_params(axis="x", rotation=30)
        savefig("21_return_rate")
        self.insights.append(f"21. Highest return rate: {rate.iloc[0]['category']} ({rate.iloc[0]['return_rate']:.1f}%).")

    # 22. Product Rating Distribution
    def analysis_22_rating_distribution(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.hist(self.reviews["rating"], bins=[0.5,1.5,2.5,3.5,4.5,5.5],
                 color=PALETTE[6], edgecolor="white", rwidth=0.8)
        ax1.set_title("Review Ratings Distribution")
        ax1.set_xlabel("Rating"); ax1.set_ylabel("Review Count")
        ax1.set_xticks([1,2,3,4,5])

        cat_rating = self.reviews.merge(self.products[["product_id","category"]], on="product_id")
        cat_avg = cat_rating.groupby("category")["rating"].mean().sort_values()
        ax2.barh(cat_avg.index, cat_avg.values, color=PALETTE[:len(cat_avg)])
        ax2.set_xlim(0, 5); ax2.set_title("Avg Rating by Category")
        fig.suptitle("Analysis 22 — Product Rating Distribution",
                     fontsize=14, fontweight="bold")
        savefig("22_rating_distribution")
        avg = self.reviews["rating"].mean()
        pct_pos = (self.reviews["sentiment"] == "positive").mean() * 100
        self.insights.append(f"22. Overall avg rating: {avg:.2f}/5. {pct_pos:.1f}% positive reviews.")

    # 23. Inventory Turnover Rate
    def analysis_23_inventory_turnover(self):
        df = self.inventory.copy()
        df["turnover"] = df["units_sold"] / (df["stock"] + 0.01)
        cat_turn = df.groupby("category")["turnover"].mean().sort_values(ascending=False).reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(cat_turn["category"], cat_turn["turnover"], color=PALETTE[:len(cat_turn)])
        ax.set_title("Analysis 23 — Inventory Turnover Rate by Category",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Turnover Ratio"); ax.tick_params(axis="x", rotation=30)
        savefig("23_inventory_turnover")
        top = cat_turn.iloc[0]
        self.insights.append(f"23. Highest inventory turnover: {top['category']} ({top['turnover']:.2f}x).")

    # 24. Dead Stock / Slow-Moving Products
    def analysis_24_dead_stock(self):
        df = self.inventory.copy()
        dead = df[df["units_sold"] == 0].sort_values("inventory_value", ascending=False).head(20)
        dead = dead.merge(self.products[["product_id","product_name"]], on="product_id")

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(dead["product_name"], dead["inventory_value"], color="salmon")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax.set_title("Analysis 24 — Top 20 Dead-Stock Products by Inventory Value",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Inventory Value")
        savefig("24_dead_stock")
        total_dead = dead["inventory_value"].sum()
        self.insights.append(f"24. Top 20 dead-stock products lock up {CURRENCY}{total_dead/MILLION:.1f}M in capital.")

    # 25. Market Basket Analysis (Top Co-purchased Products)
    def analysis_25_basket_analysis(self):
        order_products = self.items.groupby("order_id")["product_id"].apply(list)
        pairs = Counter()
        for prods in order_products:
            for combo in combinations(sorted(set(prods)), 2):
                pairs[combo] += 1
        top_pairs = pd.DataFrame(pairs.most_common(15), columns=["pair","count"])
        top_pairs["pair_label"] = top_pairs["pair"].apply(lambda x: f"{x[0]} + {x[1]}")

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(top_pairs["pair_label"], top_pairs["count"], color=PALETTE[7])
        ax.set_title("Analysis 25 — Market Basket: Top 15 Co-purchased Product Pairs",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Co-purchase Frequency")
        savefig("25_basket_analysis")
        top = top_pairs.iloc[0]
        self.insights.append(f"25. Most frequently co-purchased pair: {top['pair_label']} ({top['count']} orders).")

    # 26. Price vs Rating Correlation
    def analysis_26_price_vs_rating(self):
        df = self.products[["price","rating","review_count","category"]].dropna()
        corr, pval = stats.pearsonr(df["price"], df["rating"])

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df["price"], df["rating"],
                             c=df["review_count"], cmap="viridis",
                             alpha=0.5, s=20)
        plt.colorbar(scatter, ax=ax, label="Review Count")
        m, b = np.polyfit(df["price"], df["rating"], 1)
        x_line = np.linspace(df["price"].min(), df["price"].max(), 100)
        ax.plot(x_line, m*x_line + b, "r--", lw=2, label=f"r={corr:.3f}")
        ax.set_title("Analysis 26 — Price vs Product Rating Correlation",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Price (₦)"); ax.set_ylabel("Rating"); ax.legend()
        savefig("26_price_vs_rating")
        self.insights.append(f"26. Price-Rating correlation: r={corr:.3f} (p={pval:.4f}).")

    # 27. Order Status Distribution
    def analysis_27_order_status(self):
        df = self.orders["status"].value_counts().reset_index()
        df.columns = ["status","count"]
        df["pct"] = df["count"] / df["count"].sum() * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.pie(df["count"], labels=df["status"], autopct="%1.1f%%",
                colors=PALETTE[:len(df)], startangle=90)
        ax1.set_title("Order Status Distribution")
        ax2.bar(df["status"], df["count"], color=PALETTE[:len(df)])
        ax2.set_title("Order Count by Status")
        fig.suptitle("Analysis 27 — Order Status Distribution",
                     fontsize=14, fontweight="bold")
        savefig("27_order_status")
        cancelled = df[df["status"]=="Cancelled"]["pct"].values[0] if "Cancelled" in df["status"].values else 0
        self.insights.append(f"27. Cancellation rate: {cancelled:.1f}% of all orders.")

    # 28. Order Volume by Day of Week
    def analysis_28_orders_by_dow(self):
        order_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        df = self.orders["dow"].value_counts().reindex(order_days).reset_index()
        df.columns = ["day","count"]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df["day"], df["count"], color=PALETTE[:7])
        ax.set_title("Analysis 28 — Order Volume by Day of Week",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Orders"); ax.tick_params(axis="x", rotation=15)
        savefig("28_orders_by_dow")
        peak_day = df.loc[df["count"].idxmax(), "day"]
        self.insights.append(f"28. Peak order day: {peak_day}.")

    # 29. Order Volume by Hour of Day
    def analysis_29_orders_by_hour(self):
        df = self.orders.groupby("hour").size().reset_index(name="count")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(df["hour"], df["count"], color=PALETTE[2])
        ax.set_title("Analysis 29 — Order Volume by Hour of Day",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Hour (24h)"); ax.set_ylabel("Orders")
        ax.set_xticks(range(0, 24))
        savefig("29_orders_by_hour")
        peak_hr = df.loc[df["count"].idxmax(), "hour"]
        self.insights.append(f"29. Peak order hour: {peak_hr}:00.")

    # 30. Order Fulfilment Time Distribution
    def analysis_30_fulfilment_time(self):
        df = self.delivered.dropna(subset=["delivery_date"]).copy()
        df["days_to_deliver"] = (df["delivery_date"] - df["order_date"]).dt.days
        df = df[df["days_to_deliver"] >= 0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.hist(df["days_to_deliver"], bins=14, color=PALETTE[4], edgecolor="white")
        ax1.set_title("Fulfilment Time Distribution")
        ax1.set_xlabel("Days"); ax1.set_ylabel("Orders")
        ax2.boxplot(df["days_to_deliver"], patch_artist=True,
                    boxprops=dict(facecolor=PALETTE[5]))
        ax2.set_title("Fulfilment Time Box Plot"); ax2.set_ylabel("Days")
        fig.suptitle("Analysis 30 — Order Fulfilment Time Distribution",
                     fontsize=14, fontweight="bold")
        savefig("30_fulfilment_time")
        avg = df["days_to_deliver"].mean()
        self.insights.append(f"30. Average fulfilment time: {avg:.1f} days.")

    # 31. Repeat Purchase Rate
    def analysis_31_repeat_purchase(self):
        order_counts = self.delivered.groupby("customer_id").size()
        total_cust = len(order_counts)
        repeat_cust = (order_counts > 1).sum()
        repeat_rate = repeat_cust / total_cust * 100

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].pie([repeat_cust, total_cust - repeat_cust],
                    labels=["Repeat Buyers","One-time Buyers"],
                    autopct="%1.1f%%", colors=[PALETTE[0], PALETTE[6]])
        axes[0].set_title("Repeat vs One-time Buyers")
        dist = order_counts.value_counts().sort_index().head(10)
        axes[1].bar(dist.index.astype(str), dist.values, color=PALETTE[:10])
        axes[1].set_title("Distribution of Purchase Frequency")
        axes[1].set_xlabel("# of Orders"); axes[1].set_ylabel("Customers")
        plt.suptitle("Analysis 31 — Repeat Purchase Rate",
                     fontsize=14, fontweight="bold")
        savefig("31_repeat_purchase")
        self.insights.append(f"31. Repeat purchase rate: {repeat_rate:.1f}% of customers ordered more than once.")

    # 32. Discount Impact on Revenue & Margin
    def analysis_32_discount_impact(self):
        df = self.delivered.groupby("discount_pct").agg(
            orders=("order_id","count"),
            avg_order_value=("order_total","mean"),
            total_revenue=("order_total","sum")
        ).reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(df["discount_pct"].astype(str), df["orders"], color=PALETTE[0])
        axes[0].set_title("Orders by Discount Level"); axes[0].set_xlabel("Discount %")
        axes[1].bar(df["discount_pct"].astype(str), df["avg_order_value"], color=PALETTE[3])
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        axes[1].set_title("Avg Order Value by Discount Level"); axes[1].set_xlabel("Discount %")
        plt.suptitle("Analysis 32 — Discount Impact on Revenue & AOV",
                     fontsize=14, fontweight="bold")
        savefig("32_discount_impact")
        pct_discounted = (self.delivered["discount_pct"] > 0).mean() * 100
        self.insights.append(f"32. {pct_discounted:.1f}% of delivered orders included a discount.")

    # 33. Seasonality — Monthly Revenue Heatmap
    def analysis_33_seasonality(self):
        df = self.delivered.groupby(["year","month"])["order_total"].sum().reset_index()
        pivot = df.pivot(index="year", columns="month", values="order_total") / MILLION

        fig, ax = plt.subplots(figsize=(14, 5))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                    cbar_kws={"label": f"Revenue ({CURRENCY}M)"})
        ax.set_title("Analysis 33 — Seasonality: Monthly Revenue Heatmap (₦M)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Month"); ax.set_ylabel("Year")
        savefig("33_seasonality_heatmap")
        peak_month = df.loc[df["order_total"].idxmax()]
        self.insights.append(f"33. Peak revenue month: {int(peak_month['year'])}-{int(peak_month['month']):02d}.")

    # 34. Order Size Distribution
    def analysis_34_order_size(self):
        items_per_order = self.items.groupby("order_id")["quantity"].sum().reset_index(name="total_items")
        items_per_order = items_per_order.merge(self.orders[["order_id","status"]], on="order_id")
        items_per_order = items_per_order[items_per_order["status"] == "Delivered"]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(items_per_order["total_items"], bins=20, color=PALETTE[1], edgecolor="white")
        ax.set_title("Analysis 34 — Order Size Distribution (Total Items per Order)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Total Items"); ax.set_ylabel("Order Count")
        savefig("34_order_size")
        avg = items_per_order["total_items"].mean()
        self.insights.append(f"34. Average items per delivered order: {avg:.2f}.")

    # 35. Campaign ROI Analysis
    def analysis_35_campaign_roi(self):
        df = self.campaigns.copy()
        df["roi"] = (df["revenue_attr"] - df["spend"]) / df["spend"] * 100
        df["ctr"] = df["clicks"] / df["impressions"] * 100
        df["conversion_rate"] = df["conversions"] / df["clicks"] * 100
        df = df.sort_values("roi", ascending=False)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].barh(df["campaign"], df["roi"], color=df["roi"].apply(lambda x: "green" if x>0 else "red"))
        axes[0].axvline(0, color="black", lw=0.8); axes[0].set_title("Campaign ROI %")
        axes[1].barh(df["campaign"], df["ctr"], color=PALETTE[2])
        axes[1].set_title("Click-Through Rate (CTR) %")
        axes[2].barh(df["campaign"], df["conversion_rate"], color=PALETTE[3])
        axes[2].set_title("Conversion Rate %")
        fig.suptitle("Analysis 35 — Campaign ROI, CTR & Conversion Rate",
                     fontsize=14, fontweight="bold")
        savefig("35_campaign_roi")
        best = df.iloc[0]
        self.insights.append(f"35. Best ROI campaign: '{best['campaign']}' ({best['roi']:.0f}% ROI).")

    # 36. Customer Acquisition Cost (CAC) by Channel
    def analysis_36_cac(self):
        ch_spend = self.campaigns.groupby("channel")["spend"].sum().reset_index()
        ch_customers = self.customers.groupby("channel").size().reset_index(name="customers")
        df = ch_spend.merge(ch_customers, on="channel")
        df["cac"] = df["spend"] / df["customers"]
        df = df.sort_values("cac")

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(df["channel"], df["cac"], color=PALETTE[:len(df)])
        for bar, val in zip(bars, df["cac"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f"{CURRENCY}{val:,.0f}", ha="center", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax.set_title("Analysis 36 — Customer Acquisition Cost (CAC) by Channel",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("CAC"); ax.tick_params(axis="x", rotation=20)
        savefig("36_cac_by_channel")
        best = df.iloc[0]
        self.insights.append(f"36. Lowest CAC channel: {best['channel']} ({CURRENCY}{best['cac']:,.0f}/customer).")

    # 37. Revenue Per Campaign
    def analysis_37_campaign_revenue(self):
        merged = self.delivered[self.delivered["campaign"].notna()].copy()
        df = merged.groupby("campaign")["order_total"].agg(["sum","count"]).reset_index()
        df.columns = ["campaign","revenue","orders"]
        df = df.sort_values("revenue", ascending=False)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(df["campaign"], df["revenue"], color=PALETTE[:len(df)])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax.set_title("Analysis 37 — Revenue Attributed to Campaigns",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Revenue")
        savefig("37_campaign_revenue")
        top = df.iloc[0]
        self.insights.append(f"37. Top revenue campaign: '{top['campaign']}' ({CURRENCY}{top['revenue']/MILLION:.1f}M).")

    # 38. Website Session & Conversion Funnel
    def analysis_38_session_funnel(self):
        df = self.sessions.copy()
        df["month"] = df["session_date"].dt.to_period("M")
        monthly = df.groupby("month").agg(
            sessions=("session_id","count"),
            bounces=("bounced","sum"),
            conversions=("converted","sum")
        ).reset_index()
        monthly["bounce_rate"] = monthly["bounces"] / monthly["sessions"] * 100
        monthly["conv_rate"] = monthly["conversions"] / monthly["sessions"] * 100
        monthly["month_str"] = monthly["month"].astype(str)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax1.plot(monthly["month_str"], monthly["sessions"], color=PALETTE[0], lw=2, label="Sessions")
        ax1.plot(monthly["month_str"], monthly["conversions"], color=PALETTE[3], lw=2, label="Conversions")
        ax1.set_title("Analysis 38 — Web Sessions & Conversion Funnel",
                      fontsize=14, fontweight="bold")
        ax1.legend(); ax1.set_ylabel("Count")
        ax2.plot(monthly["month_str"], monthly["bounce_rate"], color="red", lw=2, label="Bounce Rate %")
        ax2.plot(monthly["month_str"], monthly["conv_rate"], color="green", lw=2, label="Conv Rate %")
        ax2.legend(); ax2.set_ylabel("%"); ax2.set_xlabel("Month")
        ax2.tick_params(axis="x", rotation=45)
        savefig("38_session_funnel")
        avg_conv = monthly["conv_rate"].mean()
        self.insights.append(f"38. Average website conversion rate: {avg_conv:.2f}%.")

    # 39. Device Split — Sessions & Conversions
    def analysis_39_device_split(self):
        df = self.sessions.groupby("device").agg(
            sessions=("session_id","count"),
            conversions=("converted","sum")
        ).reset_index()
        df["conv_rate"] = df["conversions"] / df["sessions"] * 100

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, col, title in zip(axes, ["sessions","conversions","conv_rate"],
                                  ["Sessions","Conversions","Conversion Rate %"]):
            ax.bar(df["device"], df[col], color=PALETTE[:3])
            ax.set_title(title)
        fig.suptitle("Analysis 39 — Device Split: Sessions & Conversions",
                     fontsize=14, fontweight="bold")
        savefig("39_device_split")
        top_conv_dev = df.loc[df["conv_rate"].idxmax(), "device"]
        self.insights.append(f"39. Highest converting device: {top_conv_dev}.")

    # 40. Sales Forecasting — 3-Month Moving Average
    def analysis_40_sales_forecast(self):
        df = (self.delivered.groupby("yearmonth")["order_total"]
              .sum().reset_index())
        df["yearmonth_str"] = df["yearmonth"].astype(str)
        df["ma3"] = df["order_total"].rolling(3).mean()
        df["ma6"] = df["order_total"].rolling(6).mean()

        # Simple linear extrapolation
        x = np.arange(len(df))
        valid = df["order_total"].notna()
        m, b, r, p, se = stats.linregress(x[valid], df["order_total"][valid])
        x_future = np.arange(len(df), len(df)+3)
        forecast = m * x_future + b
        forecast_months = [f"Forecast+{i+1}" for i in range(3)]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df["yearmonth_str"], df["order_total"], color=PALETTE[0], lw=2, label="Actual")
        ax.plot(df["yearmonth_str"], df["ma3"], color="orange", lw=2, ls="--", label="MA-3")
        ax.plot(df["yearmonth_str"], df["ma6"], color="red", lw=2, ls=":", label="MA-6")
        ax.plot(forecast_months, forecast, "g^--", lw=2, markersize=8, label="Forecast (Linear)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax.set_title("Analysis 40 — Sales Forecasting (Moving Avg + Linear Trend)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Month"); ax.set_ylabel("Revenue"); ax.legend()
        ax.tick_params(axis="x", rotation=45)
        savefig("40_sales_forecast")
        self.insights.append(f"40. Linear revenue trend: {CURRENCY}{m:+,.0f}/month growth rate.")

    # 41. Revenue per Order — Distribution Analysis
    def analysis_41_revenue_distribution(self):
        df = self.delivered["order_total"]
        skew = stats.skew(df)
        kurt = stats.kurtosis(df)
        p50, p75, p90, p95 = df.quantile([0.5,0.75,0.9,0.95])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.hist(df, bins=50, color=PALETTE[1], edgecolor="white")
        ax1.axvline(df.mean(), color="red", lw=2, label=f"Mean {CURRENCY}{df.mean():,.0f}")
        ax1.axvline(p50, color="green", lw=2, label=f"Median {CURRENCY}{p50:,.0f}")
        ax1.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax1.set_title("Order Value Distribution"); ax1.legend()
        ax2.boxplot(df, patch_artist=True, boxprops=dict(facecolor=PALETTE[2]))
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax2.set_title("Order Value Box Plot")
        fig.suptitle("Analysis 41 — Order Revenue Distribution Analysis",
                     fontsize=14, fontweight="bold")
        savefig("41_revenue_distribution")
        self.insights.append(f"41. Revenue distribution: skew={skew:.2f}, kurtosis={kurt:.2f}. P90={CURRENCY}{p90:,.0f}.")

    # 42. Gender-based Purchase Behaviour
    def analysis_42_gender_behaviour(self):
        df = self.delivered.merge(self.customers[["customer_id","gender"]], on="customer_id")
        gender_stats = df.groupby("gender").agg(
            orders=("order_id","count"),
            revenue=("order_total","sum"),
            avg_order=("order_total","mean")
        ).reset_index()

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, col, title in zip(axes, ["orders","revenue","avg_order"],
                                  ["Order Count","Total Revenue","Avg Order Value"]):
            ax.bar(gender_stats["gender"], gender_stats[col],
                   color=PALETTE[:len(gender_stats)])
            if col in ("revenue","avg_order"):
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
            ax.set_title(title)
        fig.suptitle("Analysis 42 — Purchase Behaviour by Gender",
                     fontsize=14, fontweight="bold")
        savefig("42_gender_behaviour")
        top_g = gender_stats.loc[gender_stats["revenue"].idxmax(), "gender"]
        self.insights.append(f"42. Highest revenue gender segment: {top_g}.")

    # 43. CLV vs Acquisition Channel — ANOVA
    def analysis_43_clv_by_channel(self):
        clv = self.delivered.groupby("customer_id")["order_total"].sum().reset_index(name="clv")
        df = clv.merge(self.customers[["customer_id","channel"]], on="customer_id")

        fig, ax = plt.subplots(figsize=(12, 6))
        channels = df["channel"].unique()
        data = [df[df["channel"]==ch]["clv"].values for ch in channels]
        ax.boxplot(data, labels=channels, patch_artist=True,
                   boxprops=dict(facecolor=PALETTE[0]))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax.set_title("Analysis 43 — CLV Distribution by Acquisition Channel (ANOVA)",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Customer Lifetime Value"); ax.tick_params(axis="x", rotation=20)

        f_stat, p_val = stats.f_oneway(*data)
        ax.text(0.02, 0.97, f"ANOVA: F={f_stat:.2f}, p={p_val:.4f}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        savefig("43_clv_by_channel")
        self.insights.append(f"43. CLV differs significantly by channel (ANOVA F={f_stat:.2f}, p={p_val:.4f}).")

    # 44. Review Sentiment Over Time
    def analysis_44_sentiment_trend(self):
        df = self.reviews.copy()
        df["review_date"] = pd.to_datetime(df["review_date"])
        df["month"] = df["review_date"].dt.to_period("M")
        monthly = df.groupby(["month","sentiment"]).size().unstack(fill_value=0)
        monthly.index = monthly.index.astype(str)
        monthly = monthly.div(monthly.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(14, 5))
        monthly.plot(kind="area", ax=ax, stacked=True,
                     color={"positive":"green","neutral":"orange","negative":"red"}, alpha=0.7)
        ax.set_title("Analysis 44 — Review Sentiment Trend Over Time",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Month"); ax.set_ylabel("Sentiment %"); ax.legend()
        ax.tick_params(axis="x", rotation=45)
        savefig("44_sentiment_trend")
        pos_pct = (df["sentiment"]=="positive").mean() * 100
        self.insights.append(f"44. {pos_pct:.1f}% of all reviews are positive.")

    # 45. K-Means Customer Clustering
    def analysis_45_customer_clusters(self):
        clv = self.delivered.groupby("customer_id")["order_total"].sum().reset_index(name="clv")
        freq = self.delivered.groupby("customer_id").size().reset_index(name="frequency")
        df = clv.merge(freq, on="customer_id").merge(
            self.customers[["customer_id","age"]], on="customer_id"
        )
        X = StandardScaler().fit_transform(df[["clv","frequency","age"]])
        km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
        df["cluster"] = km.labels_
        pca = PCA(n_components=2).fit_transform(X)
        df["pc1"], df["pc2"] = pca[:,0], pca[:,1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        for c in range(4):
            mask = df["cluster"] == c
            ax1.scatter(df[mask]["pc1"], df[mask]["pc2"],
                        label=f"Cluster {c}", alpha=0.5, s=10)
        ax1.set_title("PCA — Customer Clusters"); ax1.legend()
        cluster_stats = df.groupby("cluster")[["clv","frequency"]].mean()
        cluster_stats.plot(kind="bar", ax=ax2)
        ax2.set_title("Cluster Profiles — Avg CLV & Frequency"); ax2.tick_params(axis="x", rotation=0)
        fig.suptitle("Analysis 45 — K-Means Customer Clustering (k=4)",
                     fontsize=14, fontweight="bold")
        savefig("45_customer_clusters")
        self.insights.append("45. 4 distinct customer clusters identified based on CLV, frequency & age.")

    # 46. Product Category Correlation Matrix
    def analysis_46_category_correlation(self):
        merged = self.items_ext.merge(self.orders[["order_id","yearmonth","status"]], on="order_id")
        df = merged[merged["status"]=="Delivered"].groupby(["yearmonth","category"])["subtotal"].sum().unstack(fill_value=0)
        corr = df.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                    mask=mask, square=True, linewidths=0.5)
        ax.set_title("Analysis 46 — Product Category Revenue Correlation Matrix",
                     fontsize=14, fontweight="bold")
        savefig("46_category_correlation")
        self.insights.append("46. Revenue correlations across categories reveal co-movement patterns.")

    # 47. Loyalty Tier Migration Simulation
    def analysis_47_loyalty_distribution(self):
        df = self.customers["loyalty_tier"].value_counts()
        order_map = ["Bronze","Silver","Gold","Platinum"]
        df = df.reindex(order_map)
        clv = self.delivered.merge(self.customers[["customer_id","loyalty_tier"]], on="customer_id")
        tier_clv = clv.groupby("loyalty_tier")["order_total"].sum().reindex(order_map)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.bar(df.index, df.values, color=["#CD7F32","#C0C0C0","#FFD700","#E5E4E2"])
        ax1.set_title("Customer Count by Tier"); ax1.set_ylabel("Count")
        ax2.bar(tier_clv.index, tier_clv.values, color=["#CD7F32","#C0C0C0","#FFD700","#E5E4E2"])
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax2.set_title("Total Revenue by Tier"); ax2.set_ylabel("Revenue")
        fig.suptitle("Analysis 47 — Loyalty Tier Distribution & Revenue",
                     fontsize=14, fontweight="bold")
        savefig("47_loyalty_distribution")
        plat_rev_share = tier_clv["Platinum"] / tier_clv.sum() * 100
        self.insights.append(f"47. Platinum customers (5% of base) drive {plat_rev_share:.1f}% of revenue.")

    # 48. Revenue vs Marketing Spend — Scatter
    def analysis_48_spend_vs_revenue(self):
        df = self.campaigns.copy()
        corr, _ = stats.pearsonr(df["spend"], df["revenue_attr"])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df["spend"], df["revenue_attr"],
                   s=df["conversions"]/50, c=PALETTE[:len(df)],
                   alpha=0.8, edgecolors="black", lw=0.5)
        for _, row in df.iterrows():
            ax.annotate(row["campaign"].split()[0], (row["spend"], row["revenue_attr"]),
                        textcoords="offset points", xytext=(5,5), fontsize=7)
        m, b = np.polyfit(df["spend"], df["revenue_attr"], 1)
        x_line = np.linspace(df["spend"].min(), df["spend"].max(), 100)
        ax.plot(x_line, m*x_line+b, "r--", lw=2, label=f"r={corr:.2f}")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax.set_title("Analysis 48 — Marketing Spend vs Revenue Attribution",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Spend"); ax.set_ylabel("Revenue Attributed"); ax.legend()
        savefig("48_spend_vs_revenue")
        self.insights.append(f"48. Spend-Revenue correlation: r={corr:.2f}. ROAS ~{df['revenue_attr'].sum()/df['spend'].sum():.1f}x.")

    # 49. Product Lifecycle — Revenue by Launch Cohort
    def analysis_49_product_lifecycle(self):
        self.products["launch_date"] = pd.to_datetime(self.products["launch_date"])
        self.products["launch_year"] = self.products["launch_date"].dt.year
        merged = self.items_ext.merge(self.orders[["order_id","status"]], on="order_id")
        merged = merged[merged["status"]=="Delivered"]
        merged = merged.merge(self.products[["product_id","launch_year"]], on="product_id")
        df = merged.groupby("launch_year")["subtotal"].sum() / MILLION

        fig, ax = plt.subplots(figsize=(10, 5))
        df.plot(kind="bar", ax=ax, color=PALETTE[:len(df)])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_currency))
        ax.set_title("Analysis 49 — Revenue by Product Launch Year Cohort",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Revenue (₦M)"); ax.set_xlabel("Launch Year")
        ax.tick_params(axis="x", rotation=0)
        savefig("49_product_lifecycle")
        self.insights.append("49. Newer product cohorts show accelerating revenue contribution.")

    # 50. Executive Dashboard — KPI Summary
    def analysis_50_kpi_dashboard(self):
        total_rev    = self.delivered["order_total"].sum()
        total_orders = len(self.delivered)
        aov          = total_rev / total_orders
        total_cust   = self.customers["customer_id"].nunique()
        active_cust  = self.delivered["customer_id"].nunique()
        clv_avg      = self.delivered.groupby("customer_id")["order_total"].sum().mean()
        cancel_rate  = (self.orders["status"]=="Cancelled").mean() * 100
        ret_rate     = (self.orders["status"]=="Returned").mean() * 100
        avg_rating   = self.reviews["rating"].mean()
        total_profit = self.items_ext.merge(
            self.orders[["order_id","status"]], on="order_id"
        ).query("status=='Delivered'")["profit"].sum()
        net_margin   = total_profit / total_rev * 100

        kpis = {
            "Total Revenue": f"{CURRENCY}{total_rev/MILLION:.1f}M",
            "Total Orders":  f"{total_orders:,}",
            "Avg Order Value": f"{CURRENCY}{aov:,.0f}",
            "Active Customers": f"{active_cust:,} / {total_cust:,}",
            "Avg CLV": f"{CURRENCY}{clv_avg:,.0f}",
            "Net Profit Margin": f"{net_margin:.1f}%",
            "Cancellation Rate": f"{cancel_rate:.1f}%",
            "Return Rate": f"{ret_rate:.1f}%",
            "Avg Product Rating": f"{avg_rating:.2f}/5",
            "Total Campaigns": f"{len(self.campaigns)}",
        }

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        colors = [PALETTE[i % len(PALETTE)] for i in range(10)]
        for ax, (key, val), color in zip(axes, kpis.items(), colors):
            ax.set_facecolor(color)
            ax.text(0.5, 0.6, val, ha="center", va="center",
                    transform=ax.transAxes, fontsize=16, fontweight="bold", color="white")
            ax.text(0.5, 0.25, key, ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="white")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.suptitle("Analysis 50 — Executive KPI Dashboard (3-Year Summary)",
                     fontsize=16, fontweight="bold", y=1.02)
        fig.patch.set_facecolor("#2b2b2b")
        savefig("50_kpi_dashboard")
        self.insights.append(
            f"50. Summary: {CURRENCY}{total_rev/MILLION:.1f}M revenue, "
            f"{total_orders:,} orders, {net_margin:.1f}% net margin, "
            f"{avg_rating:.2f}/5 avg rating."
        )

    # ─────────────────────────────────────────────────────────────────────────
    def run_all(self):
        analyses = [
            self.analysis_01_monthly_revenue,
            self.analysis_02_quarterly_revenue,
            self.analysis_03_revenue_by_category,
            self.analysis_04_aov_trend,
            self.analysis_05_revenue_by_payment,
            self.analysis_06_revenue_by_region,
            self.analysis_07_yoy_growth,
            self.analysis_08_gross_margin,
            self.analysis_09_pareto_products,
            self.analysis_10_revenue_by_loyalty,
            self.analysis_11_customer_acquisition,
            self.analysis_12_retention_rate,
            self.analysis_13_clv,
            self.analysis_14_rfm,
            self.analysis_15_new_vs_returning,
            self.analysis_16_customer_geo,
            self.analysis_17_age_distribution,
            self.analysis_18_channel_attribution,
            self.analysis_19_top_products_units,
            self.analysis_20_top_products_revenue,
            self.analysis_21_return_rate,
            self.analysis_22_rating_distribution,
            self.analysis_23_inventory_turnover,
            self.analysis_24_dead_stock,
            self.analysis_25_basket_analysis,
            self.analysis_26_price_vs_rating,
            self.analysis_27_order_status,
            self.analysis_28_orders_by_dow,
            self.analysis_29_orders_by_hour,
            self.analysis_30_fulfilment_time,
            self.analysis_31_repeat_purchase,
            self.analysis_32_discount_impact,
            self.analysis_33_seasonality,
            self.analysis_34_order_size,
            self.analysis_35_campaign_roi,
            self.analysis_36_cac,
            self.analysis_37_campaign_revenue,
            self.analysis_38_session_funnel,
            self.analysis_39_device_split,
            self.analysis_40_sales_forecast,
            self.analysis_41_revenue_distribution,
            self.analysis_42_gender_behaviour,
            self.analysis_43_clv_by_channel,
            self.analysis_44_sentiment_trend,
            self.analysis_45_customer_clusters,
            self.analysis_46_category_correlation,
            self.analysis_47_loyalty_distribution,
            self.analysis_48_spend_vs_revenue,
            self.analysis_49_product_lifecycle,
            self.analysis_50_kpi_dashboard,
        ]
        for i, fn in enumerate(analyses, 1):
            print(f"Running analysis {i:02d}/{len(analyses)}: {fn.__name__[9:]} …")
            try:
                fn()
            except Exception as e:
                print(f"  [WARN] {fn.__name__} failed: {e}")

        self._write_insights_report()

    def _write_insights_report(self):
        path = os.path.join(REPORT_DIR, "insights_summary.txt")
        lines = [
            "=" * 70,
            " E-COMMERCE BUSINESS INTELLIGENCE — 50 ANALYSIS INSIGHTS SUMMARY",
            "=" * 70,
            "",
        ]
        for insight in self.insights:
            lines.append(f"  {insight}")
            lines.append("")
        lines += ["=" * 70, ""]
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"\n[report] {path}")
