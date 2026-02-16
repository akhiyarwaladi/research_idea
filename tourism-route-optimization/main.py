#!/usr/bin/env python3
"""
Tourism Route Optimization in Yogyakarta
Using Max-Min Ant System with Real Road Distances from OpenStreetMap

Main entry point — run the complete experiment pipeline.

Usage:
    python main.py                  # Full pipeline
    python main.py --data-only      # Only prepare data
    python main.py --experiment     # Only run experiments (data must exist)
    python main.py --analyze        # Only run analysis (experiments must exist)
    python main.py --visualize      # Only generate figures
    python main.py --quick          # Quick test with 5 runs instead of 30
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_preparation import (
    prepare_all_data,
    load_distance_matrix,
    load_pois,
    download_road_network,
    map_pois_to_network,
    compute_euclidean_distance_matrix,
)
from experiment_runner import (
    run_full_experiment,
    run_parameter_sensitivity,
    run_scalability_analysis,
)
from statistical_analysis import run_all_analysis
from visualization import generate_all_figures


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DIST_MATRIX_FILE = os.path.join(DATA_DIR, "distance_matrix.csv")
EUC_MATRIX_FILE = os.path.join(DATA_DIR, "euclidean_matrix.csv")


def step1_prepare_data():
    """Step 1: Download OSM data and compute distance matrices."""
    print("\n" + "#" * 60)
    print("# STEP 1: DATA PREPARATION")
    print("#" * 60)
    G, pois_df, dist_matrix, names = prepare_all_data()
    return G, pois_df, dist_matrix, names


def step2_run_experiments(dist_matrix, n_runs=30, max_iter=500):
    """Step 2: Run all algorithms and collect results."""
    print("\n" + "#" * 60)
    print("# STEP 2: RUNNING EXPERIMENTS")
    print("#" * 60)

    n = len(dist_matrix)

    # Main experiment
    all_results, convergence_data = run_full_experiment(
        dist_matrix,
        n_runs=n_runs,
        max_iter=max_iter,
        run_brute_force=(n <= 12),
    )

    # Parameter sensitivity
    print("\n")
    param_df = run_parameter_sensitivity(
        dist_matrix,
        n_runs=max(5, n_runs // 3),
        max_iter=max_iter,
    )

    # Scalability analysis
    print("\n")
    scale_df = run_scalability_analysis(
        dist_matrix,
        n_runs=max(5, n_runs // 3),
        max_iter=max_iter,
    )

    return all_results, convergence_data


def step3_analyze(all_results=None):
    """Step 3: Statistical analysis."""
    print("\n" + "#" * 60)
    print("# STEP 3: STATISTICAL ANALYSIS")
    print("#" * 60)
    run_all_analysis()


def step4_visualize(G=None, pois_df=None, names=None,
                     dist_matrix=None, convergence_data=None,
                     all_results=None):
    """Step 4: Generate all figures."""
    print("\n" + "#" * 60)
    print("# STEP 4: GENERATING FIGURES")
    print("#" * 60)

    # Load euclidean matrix if available
    euc_matrix = None
    if os.path.exists(EUC_MATRIX_FILE):
        euc_df = pd.read_csv(EUC_MATRIX_FILE, index_col=0)
        euc_matrix = euc_df.values

    # Find best MMAS tour from results
    best_tour = None
    if all_results:
        mmas_results = [r for r in all_results if r["algorithm"] == "MMAS"]
        if mmas_results:
            best_mmas = min(mmas_results, key=lambda x: x["best_distance"])
            best_tour = best_mmas["tour"]

    # Build convergence dict for plotting
    conv_dict = None
    if convergence_data:
        conv_dict = {}
        for algo, conv_list in convergence_data.items():
            if conv_list:
                max_len = max(len(c) for c in conv_list)
                padded = [c + [c[-1]] * (max_len - len(c)) for c in conv_list]
                conv_dict[algo] = np.mean(padded, axis=0)

    generate_all_figures(
        pois_df=pois_df,
        G=G,
        names=names,
        road_matrix=dist_matrix,
        euclidean_matrix=euc_matrix,
        best_tour=best_tour,
        convergence_data=conv_dict,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Tourism Route Optimization — ACO + OpenStreetMap"
    )
    parser.add_argument(
        "--data-only", action="store_true",
        help="Only run data preparation step",
    )
    parser.add_argument(
        "--experiment", action="store_true",
        help="Only run experiments (data must already exist)",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Only run statistical analysis (experiments must be done)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Only generate figures",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test mode: 5 runs, 200 iterations",
    )
    parser.add_argument(
        "--runs", type=int, default=30,
        help="Number of independent runs per algorithm (default: 30)",
    )
    parser.add_argument(
        "--iterations", type=int, default=500,
        help="Max iterations per run (default: 500)",
    )
    args = parser.parse_args()

    if args.quick:
        args.runs = 5
        args.iterations = 200
        print("*** QUICK MODE: 5 runs, 200 iterations ***\n")

    # Determine what to run
    run_all = not (args.data_only or args.experiment or args.analyze or args.visualize)

    G = None
    pois_df = None
    dist_matrix = None
    names = None
    all_results = None
    convergence_data = None

    # Step 1: Data
    if run_all or args.data_only:
        G, pois_df, dist_matrix, names = step1_prepare_data()
    elif args.experiment or args.visualize:
        # Need to load existing data
        if os.path.exists(DIST_MATRIX_FILE):
            dist_matrix, names = load_distance_matrix()
            pois_df = load_pois()
            if args.visualize:
                G = download_road_network()
                pois_df = map_pois_to_network(G, pois_df)
        else:
            print("ERROR: Distance matrix not found. Run --data-only first.")
            sys.exit(1)

    if args.data_only:
        print("\nData preparation complete!")
        return

    # Step 2: Experiments
    if run_all or args.experiment:
        all_results, convergence_data = step2_run_experiments(
            dist_matrix, n_runs=args.runs, max_iter=args.iterations,
        )

    if args.experiment:
        print("\nExperiments complete!")

    # Step 3: Analysis
    if run_all or args.analyze:
        step3_analyze(all_results)

    # Step 4: Visualization
    if run_all or args.visualize:
        step4_visualize(
            G=G, pois_df=pois_df, names=names,
            dist_matrix=dist_matrix,
            convergence_data=convergence_data,
            all_results=all_results,
        )

    if run_all:
        print("\n" + "#" * 60)
        print("# ALL DONE!")
        print("#" * 60)
        print(f"\nResults saved in: results/")
        print(f"  tables/   — CSV data for paper tables")
        print(f"  figures/  — PNG/HTML figures for paper")
        print(f"\nNext steps:")
        print(f"  1. Review results in results/tables/summary_table.csv")
        print(f"  2. Open results/figures/fig2_best_route_MMAS.html in browser")
        print(f"  3. Write your paper using the blueprint!")


if __name__ == "__main__":
    main()
