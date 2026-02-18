"""
Ant Colony System (ACS) for TSP

Reference:
  Dorigo, M. & Gambardella, L.M. (1997). "Ant Colony System: A Cooperative
  Learning Approach to the Traveling Salesman Problem." IEEE Trans. Evol. Comp.
"""

import numpy as np
import time


class ACS_TSP:
    """Ant Colony System solver for TSP."""

    def __init__(
        self,
        dist_matrix,
        n_ants=None,
        alpha=1.0,
        beta=3.0,
        rho=0.1,
        q0=0.9,
        xi=0.1,
        max_iter=500,
        stagnation_limit=100,
        seed=None,
    ):
        self.D = np.array(dist_matrix, dtype=np.float64)
        self.n = len(self.D)
        self.n_ants = n_ants if n_ants else self.n
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.xi = xi
        self.max_iter = max_iter
        self.stagnation_limit = stagnation_limit

        self.rng = np.random.default_rng(seed)

        with np.errstate(divide="ignore", invalid="ignore"):
            self.eta = np.where(self.D > 0, 1.0 / self.D, 0.0)

        # Precompute eta^beta (constant throughout the run)
        self.eta_beta = self.eta ** self.beta

        nn_dist = self._nearest_neighbor_distance()
        self.tau0 = 1.0 / (self.n * nn_dist)
        self.tau = np.full((self.n, self.n), self.tau0)

        self.best_tour = None
        self.best_distance = float("inf")
        self.convergence = []

    def _nearest_neighbor_distance(self):
        visited = np.zeros(self.n, dtype=bool)
        current = 0
        visited[0] = True
        total = 0.0
        for _ in range(self.n - 1):
            dists = self.D[current].copy()
            dists[visited] = np.inf
            nxt = int(np.argmin(dists))
            total += dists[nxt]
            visited[nxt] = True
            current = nxt
        total += self.D[current, 0]
        return total

    def _tour_distance(self, tour):
        """Compute total Hamiltonian cycle distance (vectorized)."""
        idx = np.asarray(tour)
        return self.D[idx[:-1], idx[1:]].sum() + self.D[idx[-1], idx[0]]

    def _construct_tour(self):
        """ACS tour construction with pseudo-random proportional rule (vectorized)."""
        n = self.n
        visited = np.zeros(n, dtype=bool)
        start = self.rng.integers(0, n)
        tour = np.empty(n, dtype=np.intp)
        tour[0] = start
        visited[start] = True
        current = start

        for step in range(1, n):
            q = self.rng.random()

            if q <= self.q0:
                # Exploitation: pick best using vectorized scoring
                scores = self.tau[current] * self.eta_beta[current]
                scores[visited] = -np.inf
                next_node = int(np.argmax(scores))
            else:
                # Exploration: probabilistic
                probs = (self.tau[current] ** self.alpha) * self.eta_beta[current]
                probs[visited] = 0.0
                prob_sum = probs.sum()
                if prob_sum == 0:
                    candidates = np.flatnonzero(~visited)
                    next_node = self.rng.choice(candidates)
                else:
                    probs /= prob_sum
                    next_node = self.rng.choice(n, p=probs)

            # Local pheromone update
            new_val = (1 - self.xi) * self.tau[current, next_node] + self.xi * self.tau0
            self.tau[current, next_node] = new_val
            self.tau[next_node, current] = new_val

            tour[step] = next_node
            visited[next_node] = True
            current = next_node

        return tour

    def _global_pheromone_update(self):
        """Only global-best ant deposits pheromone (vectorized)."""
        deposit = 1.0 / self.best_distance
        idx = np.asarray(self.best_tour)
        from_nodes = idx
        to_nodes = np.roll(idx, -1)
        self.tau[from_nodes, to_nodes] = (
            (1 - self.rho) * self.tau[from_nodes, to_nodes] + self.rho * deposit
        )
        self.tau[to_nodes, from_nodes] = self.tau[from_nodes, to_nodes]

    def solve(self):
        start_time = time.time()
        no_improve_count = 0

        for iteration in range(self.max_iter):
            prev_best = self.best_distance

            for _ in range(self.n_ants):
                tour = self._construct_tour()
                dist = self._tour_distance(tour)
                if dist < self.best_distance:
                    self.best_distance = dist
                    self.best_tour = tour.tolist()

            # Track stagnation
            if self.best_distance < prev_best:
                no_improve_count = 0
            else:
                no_improve_count += 1

            if self.best_tour is not None:
                self._global_pheromone_update()

            self.convergence.append(self.best_distance)

            # Early stopping if no improvement
            if no_improve_count >= self.stagnation_limit:
                remaining = self.max_iter - iteration - 1
                self.convergence.extend([self.best_distance] * remaining)
                break

        elapsed = time.time() - start_time
        return self.best_tour, self.best_distance, elapsed, self.convergence


def solve_acs(dist_matrix, seed=None, **kwargs):
    """Convenience function to run ACS."""
    solver = ACS_TSP(dist_matrix, seed=seed, **kwargs)
    return solver.solve()
