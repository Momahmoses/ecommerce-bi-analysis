# E-Commerce Business Intelligence Analysis Suite

A comprehensive Python project that generates realistic synthetic e-commerce data and performs **50 high-standard business intelligence and data analytics** — covering sales, customers, products, operations, marketing, and advanced ML-powered insights.

---

## Project Structure

```
ecommerce-bi-analysis/
├── data_generator.py      # Synthetic data generator (8 datasets)
├── bi_analysis.py         # All 50 BI analyses with visualizations
├── main.py                # Entry point — runs everything
├── requirements.txt
├── data/                  # Generated CSVs (auto-created)
├── charts/                # 50 PNG charts (auto-created)
└── reports/               # Insights summary (auto-created)
```

---

## Datasets Generated

| Dataset | Rows | Description |
|---|---|---|
| customers.csv | 5,000 | Demographics, region, loyalty tier, acquisition channel |
| products.csv | 500 | Category, price, cost, stock, ratings |
| orders.csv | 50,000 | Date, status, payment method, discount, campaign |
| order_items.csv | ~99,000 | Per-item quantities, prices, subtotals |
| reviews.csv | ~28,000 | Ratings, sentiment, review dates |
| campaigns.csv | 12 | Spend, impressions, clicks, conversions, revenue |
| sessions.csv | 200,000 | Device, channel, bounce/conversion flags |
| inventory.csv | 500 | Stock levels, units sold, inventory value |

---

## 50 Business Intelligence Analyses

### Sales & Revenue (1–10)
1. Monthly Revenue Trend with 3-Month Moving Average
2. Quarterly Revenue & QoQ Growth
3. Revenue by Product Category
4. Average Order Value (AOV) Monthly Trend
5. Revenue & Orders by Payment Method
6. Revenue by Region
7. Year-over-Year Revenue Growth
8. Gross Profit Margin by Category
9. Pareto Analysis — Product Revenue Concentration (80/20 Rule)
10. Revenue & Orders by Loyalty Tier

### Customer Analytics (11–18)
11. Customer Acquisition Rate Over Time
12. Cohort Retention Rate Heatmap
13. Customer Lifetime Value (CLV) Distribution
14. RFM Segmentation (Champions / Loyal / At Risk / Lost)
15. New vs Returning Customers Monthly
16. Geographic Distribution of Customers
17. Age Distribution & CLV by Age Group
18. Marketing Channel Attribution

### Product Analytics (19–26)
19. Top 10 Best-Selling Products (Units)
20. Top 10 Revenue-Generating Products
21. Product Return Rate by Category
22. Product Rating Distribution
23. Inventory Turnover Rate by Category
24. Dead Stock / Slow-Moving Products
25. Market Basket Analysis (Co-purchased Pairs)
26. Price vs Rating Correlation (Pearson r)

### Operations & Orders (27–34)
27. Order Status Distribution
28. Order Volume by Day of Week
29. Order Volume by Hour of Day
30. Order Fulfilment Time Distribution
31. Repeat Purchase Rate
32. Discount Impact on Revenue & AOV
33. Seasonality — Monthly Revenue Heatmap
34. Order Size Distribution

### Marketing & Campaigns (35–39)
35. Campaign ROI, CTR & Conversion Rate
36. Customer Acquisition Cost (CAC) by Channel
37. Revenue Attributed to Campaigns
38. Website Session & Conversion Funnel
39. Device Split — Sessions & Conversions

### Advanced Analytics (40–50)
40. Sales Forecasting (MA + Linear Trend)
41. Revenue Distribution — Statistical Analysis
42. Purchase Behaviour by Gender
43. CLV by Acquisition Channel (ANOVA)
44. Review Sentiment Trend Over Time
45. K-Means Customer Clustering (PCA Visualisation)
46. Product Category Revenue Correlation Matrix
47. Loyalty Tier Revenue Distribution
48. Marketing Spend vs Revenue Scatter (ROAS)
49. Product Lifecycle — Revenue by Launch Cohort
50. Executive KPI Dashboard

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Charts are saved to `charts/` and a text insights summary to `reports/insights_summary.txt`.

---

## Tech Stack

- **Python 3.10+**
- **pandas** — data manipulation
- **numpy** — numerical operations
- **matplotlib / seaborn** — visualisations
- **scipy** — statistical tests (Pearson r, ANOVA, skewness)
- **scikit-learn** — K-Means clustering, PCA, StandardScaler
