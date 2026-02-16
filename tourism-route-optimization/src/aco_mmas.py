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
        seed=None,
    ):
        """
        Parameters
        ----------
        dist_matrix : 2D array-like
            n x n distance matrix (symmetric, zero diagonal).
        n_ants : int, optional
            Number of ants per iteration. Default = n (number of nodes).
        alpha : float
            Pheromone importance exponent.
        beta : float
            Heuristic (visibility) importance exponent.
        rho : float
            Pheromone evaporation rate (0 < rho < 1).
        max_iter : int
            Maximum number of iterations.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.D = np.array(dist_matrix, dtype=float)
        self.n = len(self.D)
        self.n_ants = n_ants if n_ants else self.n
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.max_iter = max_iter

        self.rng = np.random.default_rng(seed)

        # Heuristic information: eta[i][j] = 1 / d(i,j)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.eta = np.where(self.D > 0, 1.0 / self.D, 0.0)

        # Initialize pheromone bounds using NN tour
        nn_dist = self._nearest_neighbor_distance()
        self.tau_max = 1.0 / (self.rho * nn_dist)
        self.tau_min = self.tau_max / (2 * self.n)

        # Pheromone matrix — start at tau_max
        self.tau = np.full((self.n, self.n), self.tau_max)

        # Result tracking
        self.best_tour = None
        self.best_distance = float("inf")
        self.convergence = []

    def _nearest_neighbor_distance(self):
        """Compute nearest-neighbor tour distance for initialization."""
        visited = {0}
        current = 0
        total = 0.0
        for _ in range(self.n - 1):
            dists = self.D[current].copy()
            dists[list(visited)] = np.inf
            nxt = int(np.argmin(dists))
            total += dists[nxt]
            visited.add(nxt)
            current = nxt
        total += self.D[current][0]
        return total

    def _construct_tour(self):
        """Build a single ant's tour using probabilistic transition rule."""
        start = self.rng.integers(0, self.n)
        tour = [start]
        visited = set(tour)

        for _ in range(self.n - 1):
            current = tour[-1]
            unvisited = [j for j in range(self.n) if j not in visited]

            # Compute transition probabilities
            tau_vals = np.array([self.tau[current][j] for j in unvisited])
            eta_vals = np.array([self.eta[current][j] for j in unvisited])
            probs = (tau_vals ** self.alpha) * (eta_vals ** self.beta)

            prob_sum = probs.sum()
            if prob_sum == 0:
                # Fallback: uniform random
                probs = np.ones(len(unvisited)) / len(unvisited)
            else:
                probs /= prob_sum

            next_node = self.rng.choice(unvisited, p=probs)
            tour.append(next_node)
            visited.add(next_node)

        return tour

    def _tour_distance(self, tour):
        """Compute total Hamiltonian cycle distance."""
        total = sum(self.D[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
        total += self.D[tour[-1]][tour[0]]
        return total

    def _update_pheromones(self, iter_best_tour, iter_best_dist):
        """MMAS pheromone update: evaporate + deposit by best ant + clip."""
        # Evaporation
        self.tau *= 1 - self.rho

        # Deposit — only iteration-best (or global-best) ant
        deposit = 1.0 / iter_best_dist
        for i in range(len(iter_best_tour) - 1):
            a, b = iter_best_tour[i], iter_best_tour[i + 1]
            self.tau[a][b] += deposit
            self.tau[b][a] += deposit
        # Close the cycle
        a, b = iter_best_tour[-1], iter_best_tour[0]
        self.tau[a][b] += deposit
        self.tau[b][a] += deposit

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
            Node indices of the best tour found.
        best_distance : float
            Total distance of the best tour.
        elapsed : float
            Wall-clock time in seconds.
        convergence : list of float
            Best distance found at each iteration.
        """
        start_time = time.time()

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
                self.best_tour = list(iter_best_tour)

            # Pheromone update
            self._update_pheromones(iter_best_tour, iter_best_dist)

            # Stagnation check every 50 iterations
            if (iteration + 1) % 50 == 0:
                self._check_stagnation()

            self.convergence.append(self.best_distance)

        elapsed = time.time() - start_time
        return self.best_tour, self.best_distance, elapsed, self.convergence


def solve_mmas(dist_matrix, seed=None, **kwargs):
    """Convenience function to run MMAS with default parameters."""
    solver = MMAS_TSP(dist_matrix, seed=seed, **kwargs)
    return solver.solve()
