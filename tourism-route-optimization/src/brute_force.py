"""
Brute Force (Exact) TSP solver.

Only feasible for n <= 12. Used to validate heuristic solutions
against the known optimal.
"""

import numpy as np
import time
from itertools import permutations


def solve_brute_force(dist_matrix, max_nodes=12, **kwargs):
    """
    Exact TSP solver using brute force enumeration.

    Parameters
    ----------
    dist_matrix : 2D array
    max_nodes : int
        Safety limit. Raises error if n > max_nodes.

    Returns
    -------
    tour, distance, elapsed, convergence
    """
    D = np.array(dist_matrix, dtype=float)
    n = len(D)

    if n > max_nodes:
        raise ValueError(
            f"Brute force not feasible for n={n} (limit={max_nodes}). "
            f"Would need to evaluate {np.math.factorial(n - 1):,} permutations."
        )

    start_time = time.time()

    # Fix node 0 as start to eliminate rotational duplicates
    other_nodes = list(range(1, n))
    best_tour = None
    best_distance = float("inf")
    convergence = []
    checked = 0

    for perm in permutations(other_nodes):
        tour = [0] + list(perm)
        dist = sum(D[tour[i]][tour[i + 1]] for i in range(n - 1))
        dist += D[tour[-1]][tour[0]]
        checked += 1

        if dist < best_distance:
            best_distance = dist
            best_tour = list(tour)

    elapsed = time.time() - start_time
    convergence = [best_distance] * 500

    print(f"Brute force: checked {checked:,} permutations in {elapsed:.2f}s")
    return best_tour, best_distance, elapsed, convergence
