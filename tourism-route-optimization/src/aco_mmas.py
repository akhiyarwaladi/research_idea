"""
Max-Min Ant System (MMAS) for TSP

Reference:
  Stutzle, T. & Hoos, H. (2000). "MAX-MIN Ant System."
  Future Generation Computer Systems, 16(8), 889-914.
"""

import numpy as np
import time


class MMAS_TSP:
    """Max-Min Ant System solver for the Travelling Salesman Problem."""

    def __init__(
        self,
        dist_matrix,
        n_ants=None,
        alpha=1.0,
        beta=3.0,
        rho=0.02,
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
        self.max_iter = max_iter
        self.stagnation_limit = stagnation_limit

        self.rng = np.random.default_rng(seed)

        # Heuristic information: eta[i][j] = 1 / d(i,j)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.eta = np.where(self.D > 0, 1.0 / self.D, 0.0)

        # Precompute eta^beta (constant throughout the run)
        self.eta_beta = self.eta ** self.beta

        # Initialize pheromone bounds using NN tour
        nn_dist = self._nearest_neighbor_distance()
        self.tau_max = 1.0 / (self.rho * nn_dist)
        self.tau_min = self.tau_max / (2 * self.n)

        # Pheromone matrix - start at tau_max
        self.tau = np.full((self.n, self.n), self.tau_max)

        # Result tracking
        self.best_tour = None
        self.best_distance = float("inf")
        self.convergence = []

    def _nearest_neighbor_distance(self):
        """Compute nearest-neighbor tour distance for initialization."""
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
        """Build a single ant's tour using probabilistic transition rule (vectorized)."""
        n = self.n
        visited = np.zeros(n, dtype=bool)
        start = self.rng.integers(0, n)
        tour = np.empty(n, dtype=np.intp)
        tour[0] = start
        visited[start] = True
        current = start

        for step in range(1, n):
            # Vectorized probability: tau^alpha * eta^beta for unvisited only
            tau_row = self.tau[current]
            probs = (tau_row ** self.alpha) * self.eta_beta[current]
            probs[visited] = 0.0

            prob_sum = probs.sum()
            if prob_sum == 0:
                # Fallback: uniform over unvisited
                candidates = np.flatnonzero(~visited)
                next_node = self.rng.choice(candidates)
            else:
                probs /= prob_sum
                next_node = self.rng.choice(n, p=probs)

            tour[step] = next_node
            visited[next_node] = True
            current = next_node

        return tour

    def _update_pheromones(self, iter_best_tour, iter_best_dist):
        """MMAS pheromone update: evaporate + deposit by best ant + clip."""
        # Evaporation
        self.tau *= (1.0 - self.rho)

        # Deposit - only iteration-best ant
        deposit = 1.0 / iter_best_dist
        idx = np.asarray(iter_best_tour)
        from_nodes = idx
        to_nodes = np.roll(idx, -1)
        self.tau[from_nodes, to_nodes] += deposit
        self.tau[to_nodes, from_nodes] += deposit

        # Update bounds based on current global best
        self.tau_max = 1.0 / (self.rho * self.best_distance)
        self.tau_min = self.tau_max / (2 * self.n)

        # Clip pheromone values
        np.clip(self.tau, self.tau_min, self.tau_max, out=self.tau)

    def _check_stagnation(self):
        """Re-initialize pheromones if stagnation detected."""
        unique_vals = np.unique(np.round(self.tau, decimals=10))
        if len(unique_vals) <= 2:
            self.tau[:] = self.tau_max

    def solve(self):
        """
        Run MMAS and return results.

        Returns
        -------
        best_tour : list of int
        best_distance : float
        elapsed : float
        convergence : list of float
        """
        start_time = time.time()
        no_improve_count = 0

        for iteration in range(self.max_iter):
            iter_best_tour = None
            iter_best_dist = float("inf")

            # Construct tours for all ants
            for _ in range(self.n_ants):
                tour = self._construct_tour()
                dist = self._tour_distance(tour)
                if dist < iter_best_dist:
                    iter_best_dist = dist
                    iter_best_tour = tour

            # Update global best
            if iter_best_dist < self.best_distance:
                self.best_distance = iter_best_dist
                self.best_tour = iter_best_tour.tolist()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Pheromone update
            self._update_pheromones(iter_best_tour, iter_best_dist)

            # Stagnation check every 50 iterations
            if (iteration + 1) % 50 == 0:
                self._check_stagnation()

            self.convergence.append(self.best_distance)

            # Early stopping if no improvement
            if no_improve_count >= self.stagnation_limit:
                # Pad convergence to expected length
                remaining = self.max_iter - iteration - 1
                self.convergence.extend([self.best_distance] * remaining)
                break

        elapsed = time.time() - start_time
        return self.best_tour, self.best_distance, elapsed, self.convergence


def solve_mmas(dist_matrix, seed=None, **kwargs):
    """Convenience function to run MMAS with default parameters."""
    solver = MMAS_TSP(dist_matrix, seed=seed, **kwargs)
    return solver.solve()
