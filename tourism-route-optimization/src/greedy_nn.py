"""
Nearest Neighbor (Greedy) Heuristic for TSP

Deterministic baseline algorithm.
"""

import numpy as np
import time


def solve_nn(dist_matrix, start=0, **kwargs):
    """
    Nearest Neighbor greedy heuristic.

    Parameters
    ----------
    dist_matrix : 2D array
    start : int
        Starting node index.

    Returns
    -------
    tour, distance, elapsed, convergence
    """
    D = np.array(dist_matrix, dtype=float)
    n = len(D)

    start_time = time.time()

    visited = {start}
    tour = [start]
    total = 0.0
    current = start

    for _ in range(n - 1):
        dists = D[current].copy()
        dists[list(visited)] = np.inf
        nxt = int(np.argmin(dists))
        total += dists[nxt]
        visited.add(nxt)
        tour.append(nxt)
        current = nxt

    total += D[current][start]
    elapsed = time.time() - start_time

    # NN is deterministic — flat convergence
    convergence = [total] * 500

    return tour, total, elapsed, convergence


def solve_nn_all_starts(dist_matrix, **kwargs):
    """
    Run NN from every possible starting node and return the best result.
    """
    D = np.array(dist_matrix, dtype=float)
    n = len(D)

    best_tour = None
    best_dist = float("inf")
    total_time = 0.0

    for s in range(n):
        tour, dist, elapsed, _ = solve_nn(D, start=s)
        total_time += elapsed
        if dist < best_dist:
            best_dist = dist
            best_tour = tour

    convergence = [best_dist] * 500
    return best_tour, best_dist, total_time, convergence
