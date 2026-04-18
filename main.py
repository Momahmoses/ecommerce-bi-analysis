"""
E-Commerce BI Analysis — Entry Point
Generates data and runs all 50 business intelligence analyses.
"""

import os
import time

def main():
    from data_generator import generate_all
    from bi_analysis import EcommerceAnalytics

    print("=" * 60)
    print("  E-COMMERCE BUSINESS INTELLIGENCE ANALYSIS SUITE")
    print("  50 High-Level BI Analyses | 2022–2024 Synthetic Data")
    print("=" * 60)
    print()

    # ── Step 1: Generate data ────────────────────────────────────
    data_dir = "data"
    if all(os.path.exists(os.path.join(data_dir, f"{t}.csv")) for t in
           ["customers","products","orders","order_items","reviews","campaigns","sessions","inventory"]):
        print("Data already exists — loading from CSV …")
        import pandas as pd
        data = {
            name: pd.read_csv(os.path.join(data_dir, f"{name}.csv"))
            for name in ["customers","products","orders","order_items",
                         "reviews","campaigns","sessions","inventory"]
        }
    else:
        print("Step 1 — Generating datasets …\n")
        data = generate_all(data_dir)

    print()

    # ── Step 2: Run all 50 analyses ──────────────────────────────
    print("Step 2 — Running 50 BI analyses …\n")
    t0 = time.time()
    analytics = EcommerceAnalytics(data)
    analytics.run_all()
    elapsed = time.time() - t0

    print(f"\nAll 50 analyses completed in {elapsed:.1f}s.")
    print(f"Charts saved to:  charts/")
    print(f"Report saved to:  reports/insights_summary.txt")
    print()
    print("=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
