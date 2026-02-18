"""
Statistical Analysis Module

- Summary statistics per algorithm
- Wilcoxon signed-rank tests for pairwise comparison
- Generates publication-ready tables
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_raw_results():
    """Load raw experiment results."""
    path = os.path.join(RESULTS_DIR, "tables", "raw_results.csv")
    return pd.read_csv(path)


def summary_table(df=None):
    """
    Generate summary statistics table (Table 3 in paper).

    Returns DataFrame with Best/Mean/Worst/StdDev/MeanTime per algorithm.
    """
    if df is None:
        df = load_raw_results()

    summary = []
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        dists = sub["best_distance_m"].values
        times = sub["time_seconds"].values
        summary.append({
            "Algorithm": algo,
            "Best (m)": f"{dists.min():.2f}",
            "Mean (m)": f"{dists.mean():.2f}",
            "Worst (m)": f"{dists.max():.2f}",
            "Std Dev (m)": f"{dists.std():.2f}",
            "Mean Time (s)": f"{times.mean():.4f}",
            "Runs": len(dists),
        })

    result = pd.DataFrame(summary)
    path = os.path.join(RESULTS_DIR, "tables", "summary_table.csv")
    result.to_csv(path, index=False)
    print(f"Saved summary table to {path}")
    print("\n" + result.to_string(index=False))
    return result


def wilcoxon_tests(df=None, reference_algo="MMAS"):
    """
    Run Wilcoxon signed-rank tests comparing reference algorithm against all others.
    Only applicable for stochastic algorithms with multiple runs.

    Returns DataFrame with test results (Table 5 in paper).
    """
    if df is None:
        df = load_raw_results()

    ref_data = df[df["algorithm"] == reference_algo]
    if len(ref_data) <= 1:
        print(f"Cannot run Wilcoxon: {reference_algo} has only {len(ref_data)} run(s)")
        return None

    ref_dists = ref_data.sort_values("run")["best_distance_m"].values

    results = []
    other_algos = [a for a in df["algorithm"].unique() if a != reference_algo]

    for algo in other_algos:
        other_data = df[df["algorithm"] == algo]
        if len(other_data) <= 1:
            results.append({
                "Comparison": f"{reference_algo} vs {algo}",
                "W statistic": "N/A",
                "p-value": "N/A",
                "Significant (p<0.05)": "N/A (deterministic)",
                "Ref Mean (m)": f"{ref_dists.mean():.2f}",
                "Other Mean (m)": f"{other_data['best_distance_m'].values[0]:.2f}",
            })
            continue

        other_dists = other_data.sort_values("run")["best_distance_m"].values

        # Ensure same number of observations
        min_len = min(len(ref_dists), len(other_dists))
        r = ref_dists[:min_len]
        o = other_dists[:min_len]

        # Check if all differences are zero
        if np.all(r == o):
            results.append({
                "Comparison": f"{reference_algo} vs {algo}",
                "W statistic": "0",
                "p-value": "1.000",
                "Significant (p<0.05)": "No (identical)",
                "Ref Mean (m)": f"{r.mean():.2f}",
                "Other Mean (m)": f"{o.mean():.2f}",
            })
            continue

        try:
            w_stat, p_value = stats.wilcoxon(r, o, alternative="two-sided")
            significant = "Yes" if p_value < 0.05 else "No"
            results.append({
                "Comparison": f"{reference_algo} vs {algo}",
                "W statistic": f"{w_stat:.4f}",
                "p-value": f"{p_value:.6f}",
                "Significant (p<0.05)": significant,
                "Ref Mean (m)": f"{r.mean():.2f}",
                "Other Mean (m)": f"{o.mean():.2f}",
            })
        except ValueError as e:
            results.append({
                "Comparison": f"{reference_algo} vs {algo}",
                "W statistic": "Error",
                "p-value": str(e),
                "Significant (p<0.05)": "N/A",
                "Ref Mean (m)": f"{r.mean():.2f}",
                "Other Mean (m)": f"{o.mean():.2f}",
            })

    result = pd.DataFrame(results)
    path = os.path.join(RESULTS_DIR, "tables", "wilcoxon_tests.csv")
    result.to_csv(path, index=False)
    print(f"\nSaved Wilcoxon test results to {path}")
    print("\n" + result.to_string(index=False))
    return result


def wilcoxon_all_pairwise(df=None):
    """
    Run Wilcoxon signed-rank tests for ALL pairwise comparisons among
    stochastic algorithms (those with > 1 run).

    Returns DataFrame with test results.
    """
    if df is None:
        df = load_raw_results()

    # Identify stochastic algorithms (more than 1 run)
    stochastic = [
        algo for algo in df["algorithm"].unique()
        if len(df[df["algorithm"] == algo]) > 1
    ]

    results = []
    for i in range(len(stochastic)):
        for j in range(i + 1, len(stochastic)):
            a1, a2 = stochastic[i], stochastic[j]
            d1 = df[df["algorithm"] == a1].sort_values("run")["best_distance_m"].values
            d2 = df[df["algorithm"] == a2].sort_values("run")["best_distance_m"].values
            min_len = min(len(d1), len(d2))
            d1, d2 = d1[:min_len], d2[:min_len]

            if np.all(d1 == d2):
                results.append({
                    "Comparison": f"{a1} vs {a2}",
                    "W statistic": "0",
                    "p-value": "1.000000",
                    "Significant (p<0.05)": "No (identical)",
                    "Mean A (m)": f"{d1.mean():.2f}",
                    "Mean B (m)": f"{d2.mean():.2f}",
                })
                continue

            try:
                w_stat, p_value = stats.wilcoxon(d1, d2, alternative="two-sided")
                significant = "Yes" if p_value < 0.05 else "No"
                results.append({
                    "Comparison": f"{a1} vs {a2}",
                    "W statistic": f"{w_stat:.4f}",
                    "p-value": f"{p_value:.6f}",
                    "Significant (p<0.05)": significant,
                    "Mean A (m)": f"{d1.mean():.2f}",
                    "Mean B (m)": f"{d2.mean():.2f}",
                })
            except ValueError as e:
                results.append({
                    "Comparison": f"{a1} vs {a2}",
                    "W statistic": "Error",
                    "p-value": str(e),
                    "Significant (p<0.05)": "N/A",
                    "Mean A (m)": f"{d1.mean():.2f}",
                    "Mean B (m)": f"{d2.mean():.2f}",
                })

    result = pd.DataFrame(results)
    path = os.path.join(RESULTS_DIR, "tables", "wilcoxon_all_pairwise.csv")
    result.to_csv(path, index=False)
    print(f"\nSaved all-pairwise Wilcoxon tests to {path}")
    print("\n" + result.to_string(index=False))
    return result


def optimality_gap(df=None, optimal_distance=None):
    """
    Compute optimality gap for each algorithm (if brute force result available).
    """
    if df is None:
        df = load_raw_results()

    if optimal_distance is None:
        bf = df[df["algorithm"] == "BruteForce"]
        if len(bf) == 0:
            print("No brute force result available for optimality gap computation.")
            return None
        optimal_distance = bf["best_distance_m"].values[0]

    print(f"\nOptimal distance (Brute Force): {optimal_distance:.2f} m\n")
    print(f"{'Algorithm':<15} {'Best (m)':>12} {'Gap (%)':>10}")
    print("-" * 40)

    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        best = sub["best_distance_m"].min()
        gap = ((best - optimal_distance) / optimal_distance) * 100
        print(f"{algo:<15} {best:>12.2f} {gap:>9.2f}%")


def run_all_analysis():
    """Run complete statistical analysis pipeline."""
    print("=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    df = load_raw_results()

    print("\n--- Summary Table (Table 3) ---")
    summary_table(df)

    print("\n--- Wilcoxon Signed-Rank Tests (vs MMAS) ---")
    wilcoxon_tests(df)

    print("\n--- Wilcoxon All-Pairwise Tests ---")
    wilcoxon_all_pairwise(df)

    print("\n--- Optimality Gap ---")
    optimality_gap(df)


if __name__ == "__main__":
    run_all_analysis()
