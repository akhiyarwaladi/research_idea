"""
Simulated Annealing for TSP

Uses 2-opt neighborhood with O(1) delta evaluation and geometric cooling schedule.
"""

import numpy as np
import math
import time


class SA_TSP:
    """Simulated Annealing solver for TSP."""

    def __init__(
        self,
        dist_matrix,
        T0=10000,
        cooling_rate=0.999,
        T_end=1.0,
        iters_per_temp=50,
        seed=None,
    ):
        self.D = np.array(dist_matrix, dtype=np.float64)
        self.n = len(self.D)
        self.T0 = T0
        self.cooling_rate = cooling_rate
        self.T_end = T_end
        self.iters_per_temp = iters_per_temp

        self.rng = np.random.default_rng(seed)

        self.best_tour = None
        self.best_distance = float("inf")
        self.convergence = []

    def _tour_distance(self, tour):
        """Compute total Hamiltonian cycle distance (vectorized)."""
        return self.D[tour[:-1], tour[1:]].sum() + self.D[tour[-1], tour[0]]

    def _two_opt_delta(self, tour, i, j):
        """
        Compute the change in distance from a 2-opt reversal of segment [i..j].
        O(1) instead of O(n) - only the 2 changed edges matter.

        Before: ... tour[i-1] - tour[i] ... tour[j] - tour[j+1] ...
        After:  ... tour[i-1] - tour[j] ... tour[i] - tour[j+1] ...
        """
        n = self.n
        a = tour[i - 1] if i > 0 else tour[n - 1]
        b = tour[i]
        c = tour[j]
        d = tour[(j + 1) % n]

        old_cost = self.D[a, b] + self.D[c, d]
        new_cost = self.D[a, c] + self.D[b, d]
        return new_cost - old_cost

    def _two_opt_apply(self, tour, i, j):
        """Apply 2-opt reversal in-place."""
        tour[i:j + 1] = tour[i:j + 1][::-1]

    def _initial_solution(self):
        """Random initial tour as numpy array."""
        tour = np.arange(self.n, dtype=np.intp)
        self.rng.shuffle(tour)
        return tour

    def solve(self):
        start_time = time.time()

        # Initial solution
        current_tour = self._initial_solution()
        current_dist = self._tour_distance(current_tour)
        self.best_tour = current_tour.copy()
        self.best_distance = current_dist

        T = self.T0
        iteration = 0

        # Pre-compute total iterations for convergence tracking
        total_temp_steps = int(
            math.log(self.T_end / self.T0) / math.log(self.cooling_rate)
        )
        log_interval = max(1, total_temp_steps // 500)

        while T > self.T_end:
            for _ in range(self.iters_per_temp):
                # Generate 2-opt neighbor indices
                pts = self.rng.choice(self.n, size=2, replace=False)
                i, j = int(pts.min()), int(pts.max())
                if i == 0 and j == self.n - 1:
                    continue  # Skip trivial reversal

                # O(1) delta evaluation instead of O(n) full recalculation
                delta = self._two_opt_delta(current_tour, i, j)

                # Accept or reject
                if delta < 0 or self.rng.random() < math.exp(-delta / T):
                    self._two_opt_apply(current_tour, i, j)
                    current_dist += delta

                    if current_dist < self.best_distance:
                        self.best_distance = current_dist
                        self.best_tour = current_tour.copy()

            # Cool down
            T *= self.cooling_rate
            iteration += 1

            if iteration % log_interval == 0:
                self.convergence.append(self.best_distance)

        # Ensure we have exactly 500 convergence points (pad if needed)
        while len(self.convergence) < 500:
            self.convergence.append(self.best_distance)

        elapsed = time.time() - start_time
        return self.best_tour.tolist(), self.best_distance, elapsed, self.convergence[:500]


def solve_sa(dist_matrix, seed=None, **kwargs):
    """Convenience function to run SA."""
    solver = SA_TSP(dist_matrix, seed=seed, **kwargs)
    return solver.solve()
