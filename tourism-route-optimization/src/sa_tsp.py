"""
Simulated Annealing for TSP

Uses 2-opt neighborhood and geometric cooling schedule.
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
        """
        Parameters
        ----------
        T0 : float
            Initial temperature.
        cooling_rate : float
            Geometric cooling factor (T *= cooling_rate each step).
        T_end : float
            Stop when temperature drops below this.
        iters_per_temp : int
            Number of neighbor evaluations per temperature level.
        """
        self.D = np.array(dist_matrix, dtype=float)
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
        total = sum(self.D[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
        total += self.D[tour[-1]][tour[0]]
        return total

    def _two_opt_swap(self, tour, i, j):
        """Reverse the segment between indices i and j (2-opt move)."""
        new_tour = tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]
        return new_tour

    def _initial_solution(self):
        """Random initial tour."""
        tour = list(range(self.n))
        self.rng.shuffle(tour)
        return tour

    def solve(self):
        start_time = time.time()

        # Initial solution
        current_tour = self._initial_solution()
        current_dist = self._tour_distance(current_tour)
        self.best_tour = list(current_tour)
        self.best_distance = current_dist

        T = self.T0
        iteration = 0

        # Pre-compute total iterations for convergence tracking
        # Log convergence every ~equivalent to MMAS iteration count
        total_temp_steps = int(
            math.log(self.T_end / self.T0) / math.log(self.cooling_rate)
        )
        log_interval = max(1, total_temp_steps // 500)

        while T > self.T_end:
            for _ in range(self.iters_per_temp):
                # Generate 2-opt neighbor
                i, j = sorted(self.rng.choice(self.n, size=2, replace=False))
                if i == 0 and j == self.n - 1:
                    continue  # Skip trivial reversal

                neighbor = self._two_opt_swap(current_tour, i, j)
                neighbor_dist = self._tour_distance(neighbor)

                delta = neighbor_dist - current_dist

                # Accept or reject
                if delta < 0 or self.rng.random() < math.exp(-delta / T):
                    current_tour = neighbor
                    current_dist = neighbor_dist

                    if current_dist < self.best_distance:
                        self.best_distance = current_dist
                        self.best_tour = list(current_tour)

            # Cool down
            T *= self.cooling_rate
            iteration += 1

            if iteration % log_interval == 0:
                self.convergence.append(self.best_distance)

        # Ensure we have exactly 500 convergence points (pad if needed)
        while len(self.convergence) < 500:
            self.convergence.append(self.best_distance)

        elapsed = time.time() - start_time
        return self.best_tour, self.best_distance, elapsed, self.convergence[:500]


def solve_sa(dist_matrix, seed=None, **kwargs):
    """Convenience function to run SA."""
    solver = SA_TSP(dist_matrix, seed=seed, **kwargs)
    return solver.solve()
