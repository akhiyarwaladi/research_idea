"""
Experiment Runner

Runs all algorithms multiple times, collects results, and saves to CSV.
Designed for reproducible benchmarking.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from aco_mmas import solve_mmas, MMAS_TSP
from aco_acs import solve_acs, ACS_TSP
from ga_tsp import solve_ga, GA_TSP
from sa_tsp import solve_sa, SA_TSP
from greedy_nn import solve_nn_all_starts
from brute_force import solve_brute_force

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_single_algorithm(name, solve_func, dist_matrix, n_runs=30, base_seed=42, **kwargs):
    """
    Run a single algorithm n_runs times with different seeds.

    Returns
    -------
    results : list of dict
    all_convergence : list of list
    """
    results = []
    all_convergence = []

    for run in range(n_runs):
        seed = base_seed + run
        tour, dist, elapsed, convergence = solve_func(
            dist_matrix, seed=seed, **kwargs
        )
        results.append({
            "algorithm": name,
            "run": run + 1,
            "seed": seed,
            "best_distance": dist,
            "time_seconds": elapsed,
            "tour": tour,
        })
        all_convergence.append(convergence)

    return results, all_convergence


def run_deterministic_algorithm(name, solve_func, dist_matrix, **kwargs):
    """Run a deterministic algorithm once (NN, Brute Force)."""
    tour, dist, elapsed, convergence = solve_func(dist_matrix, **kwargs)
    results = [{
        "algorithm": name,
        "run": 1,
        "seed": None,
        "best_distance": dist,
        "time_seconds": elapsed,
        "tour": tour,
    }]
    return results, [convergence]


def run_full_experiment(dist_matrix, n_runs=30, max_iter=500, run_brute_force=False):
    """
    Run all algorithms and collect results.

    Parameters
    ----------
    dist_matrix : 2D array
        Distance matrix.
    n_runs : int
        Number of independent runs per stochastic algorithm.
    max_iter : int
        Max iterations for iterative algorithms.
    run_brute_force : bool
        Whether to run exact brute force (only if n <= 12).
    """
    n = len(dist_matrix)
    all_results = []
    convergence_data = {}

    # --- MMAS ---
    print(f"\n{'='*60}")
    print(f"Running MMAS ({n_runs} runs, {max_iter} iterations each)...")
    print(f"{'='*60}")
    results, conv = run_single_algorithm(
        "MMAS", solve_mmas, dist_matrix,
        n_runs=n_runs, max_iter=max_iter,
        alpha=1.0, beta=3.0, rho=0.02,
    )
    all_results.extend(results)
    convergence_data["MMAS"] = conv
    _print_summary("MMAS", results)

    # --- ACS ---
    print(f"\n{'='*60}")
    print(f"Running ACS ({n_runs} runs, {max_iter} iterations each)...")
    print(f"{'='*60}")
    results, conv = run_single_algorithm(
        "ACS", solve_acs, dist_matrix,
        n_runs=n_runs, max_iter=max_iter,
        alpha=1.0, beta=3.0, rho=0.1, q0=0.9, xi=0.1,
    )
    all_results.extend(results)
    convergence_data["ACS"] = conv
    _print_summary("ACS", results)

    # --- GA ---
    print(f"\n{'='*60}")
    print(f"Running GA ({n_runs} runs, {max_iter} generations each)...")
    print(f"{'='*60}")
    results, conv = run_single_algorithm(
        "GA", solve_ga, dist_matrix,
        n_runs=n_runs,
        pop_size=100, generations=max_iter,
        crossover_rate=0.8, mutation_rate=0.02, tournament_k=3,
    )
    all_results.extend(results)
    convergence_data["GA"] = conv
    _print_summary("GA", results)

    # --- SA ---
    print(f"\n{'='*60}")
    print(f"Running SA ({n_runs} runs)...")
    print(f"{'='*60}")
    results, conv = run_single_algorithm(
        "SA", solve_sa, dist_matrix,
        n_runs=n_runs,
        T0=10000, cooling_rate=0.999, T_end=1.0, iters_per_temp=50,
    )
    all_results.extend(results)
    convergence_data["SA"] = conv
    _print_summary("SA", results)

    # --- Nearest Neighbor ---
    print(f"\n{'='*60}")
    print("Running Nearest Neighbor (all starts)...")
    print(f"{'='*60}")
    results, conv = run_deterministic_algorithm(
        "NN", solve_nn_all_starts, dist_matrix,
    )
    all_results.extend(results)
    convergence_data["NN"] = conv
    print(f"  NN distance: {results[0]['best_distance']:.2f} m")

    # --- Brute Force (optional) ---
    if run_brute_force and n <= 12:
        print(f"\n{'='*60}")
        print(f"Running Brute Force (n={n})...")
        print(f"{'='*60}")
        results, conv = run_deterministic_algorithm(
            "BruteForce", solve_brute_force, dist_matrix,
        )
        all_results.extend(results)
        convergence_data["BruteForce"] = conv
        print(f"  Optimal distance: {results[0]['best_distance']:.2f} m")
    elif run_brute_force:
        print(f"\n  Skipping Brute Force: n={n} > 12 (not feasible)")

    # Save results
    _save_results(all_results, convergence_data)
    return all_results, convergence_data


def _print_summary(name, results):
    """Print summary statistics for an algorithm."""
    dists = [r["best_distance"] for r in results]
    times = [r["time_seconds"] for r in results]
    print(f"  Best:  {min(dists):.2f} m")
    print(f"  Mean:  {np.mean(dists):.2f} m")
    print(f"  Worst: {max(dists):.2f} m")
    print(f"  Std:   {np.std(dists):.2f} m")
    print(f"  Avg time: {np.mean(times):.4f} s")


def _save_results(all_results, convergence_data):
    """Save experiment results to CSV files."""
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)

    # Main results (without tour column for cleanliness)
    rows = []
    for r in all_results:
        rows.append({
            "algorithm": r["algorithm"],
            "run": r["run"],
            "seed": r["seed"],
            "best_distance_m": r["best_distance"],
            "time_seconds": r["time_seconds"],
        })
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "tables", "raw_results.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved raw results to {path}")

    # Save tours separately
    tours = []
    for r in all_results:
        tours.append({
            "algorithm": r["algorithm"],
            "run": r["run"],
            "tour": " -> ".join(str(x) for x in r["tour"]),
            "distance_m": r["best_distance"],
        })
    df_tours = pd.DataFrame(tours)
    path = os.path.join(RESULTS_DIR, "tables", "tours.csv")
    df_tours.to_csv(path, index=False)

    # Save convergence data
    for algo_name, conv_list in convergence_data.items():
        if conv_list:
            # Average convergence across runs
            max_len = max(len(c) for c in conv_list)
            padded = [c + [c[-1]] * (max_len - len(c)) for c in conv_list]
            avg_conv = np.mean(padded, axis=0)
            df_conv = pd.DataFrame({
                "iteration": range(1, len(avg_conv) + 1),
                "avg_best_distance": avg_conv,
            })
            path = os.path.join(RESULTS_DIR, "tables", f"convergence_{algo_name}.csv")
            df_conv.to_csv(path, index=False)

    print("Saved convergence data.")


def run_parameter_sensitivity(dist_matrix, n_runs=10, max_iter=500):
    """
    Run MMAS parameter sensitivity analysis.
    Tests different combinations of alpha, beta, rho.
    """
    alphas = [0.5, 1.0, 2.0]
    betas = [2.0, 3.0, 5.0]
    rhos = [0.02, 0.05, 0.1]

    results = []
    total = len(alphas) * len(betas) * len(rhos)
    done = 0

    print(f"\n{'='*60}")
    print(f"MMAS Parameter Sensitivity Analysis ({total} combinations)")
    print(f"{'='*60}")

    for alpha in alphas:
        for beta in betas:
            for rho in rhos:
                dists = []
                times = []
                for run in range(n_runs):
                    seed = 100 + run
                    solver = MMAS_TSP(
                        dist_matrix, alpha=alpha, beta=beta, rho=rho,
                        max_iter=max_iter, seed=seed,
                    )
                    _, dist, elapsed, _ = solver.solve()
                    dists.append(dist)
                    times.append(elapsed)

                results.append({
                    "alpha": alpha,
                    "beta": beta,
                    "rho": rho,
                    "mean_distance": np.mean(dists),
                    "std_distance": np.std(dists),
                    "best_distance": min(dists),
                    "worst_distance": max(dists),
                    "mean_time": np.mean(times),
                })
                done += 1
                print(f"  [{done}/{total}] α={alpha}, β={beta}, ρ={rho} "
                      f"→ mean={np.mean(dists):.2f} m")

    df = pd.DataFrame(results)
    path = os.path.join(RESULTS_DIR, "tables", "parameter_sensitivity.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved parameter sensitivity to {path}")
    return df


def run_scalability_analysis(dist_matrix, node_counts=None, n_runs=10, max_iter=500):
    """
    Test how algorithms scale with different numbers of nodes.
    Uses subsets of the full distance matrix.
    """
    if node_counts is None:
        n = len(dist_matrix)
        node_counts = [c for c in [10, 12, 15, 20] if c <= n]

    results = []

    print(f"\n{'='*60}")
    print(f"Scalability Analysis: {node_counts}")
    print(f"{'='*60}")

    for nc in node_counts:
        sub_matrix = dist_matrix[:nc, :nc]
        print(f"\n--- n = {nc} ---")

        for algo_name, solve_func, kwargs in [
            ("MMAS", solve_mmas, {"max_iter": max_iter, "alpha": 1.0, "beta": 3.0, "rho": 0.02}),
            ("ACS", solve_acs, {"max_iter": max_iter}),
            ("GA", solve_ga, {"pop_size": 100, "generations": max_iter}),
            ("SA", solve_sa, {}),
        ]:
            dists = []
            times = []
            for run in range(n_runs):
                _, dist, elapsed, _ = solve_func(sub_matrix, seed=200 + run, **kwargs)
                dists.append(dist)
                times.append(elapsed)

            results.append({
                "n_nodes": nc,
                "algorithm": algo_name,
                "mean_distance": np.mean(dists),
                "std_distance": np.std(dists),
                "best_distance": min(dists),
                "mean_time": np.mean(times),
            })
            print(f"  {algo_name}: mean={np.mean(dists):.2f} m, time={np.mean(times):.4f}s")

        # NN for this size
        tour, dist, elapsed, _ = solve_nn_all_starts(sub_matrix)
        results.append({
            "n_nodes": nc,
            "algorithm": "NN",
            "mean_distance": dist,
            "std_distance": 0,
            "best_distance": dist,
            "mean_time": elapsed,
        })
        print(f"  NN: {dist:.2f} m")

    df = pd.DataFrame(results)
    path = os.path.join(RESULTS_DIR, "tables", "scalability.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved scalability results to {path}")
    return df
