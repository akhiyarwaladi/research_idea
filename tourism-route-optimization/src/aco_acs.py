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
        seed=None,
    ):
        """
        Parameters
        ----------
        q0 : float
            Exploitation probability (pseudo-random proportional rule).
            Higher q0 = more greedy exploitation.
        xi : float
            Local pheromone decay rate.
        """
        self.D = np.array(dist_matrix, dtype=float)
        self.n = len(self.D)
        self.n_ants = n_ants if n_ants else self.n
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.xi = xi
        self.max_iter = max_iter

        self.rng = np.random.default_rng(seed)

        with np.errstate(divide="ignore", invalid="ignore"):
            self.eta = np.where(self.D > 0, 1.0 / self.D, 0.0)

        nn_dist = self._nearest_neighbor_distance()
        self.tau0 = 1.0 / (self.n * nn_dist)
        self.tau = np.full((self.n, self.n), self.tau0)

        self.best_tour = None
        self.best_distance = float("inf")
        self.convergence = []

    def _nearest_neighbor_distance(self):
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
        """ACS tour construction with pseudo-random proportional rule."""
        start = self.rng.integers(0, self.n)
        tour = [start]
        visited = set(tour)

        for _ in range(self.n - 1):
            current = tour[-1]
            unvisited = [j for j in range(self.n) if j not in visited]

            q = self.rng.random()
            if q <= self.q0:
                # Exploitation: pick best
                scores = [
                    self.tau[current][j] * (self.eta[current][j] ** self.beta)
                    for j in unvisited
                ]
                next_node = unvisited[int(np.argmax(scores))]
            else:
                # Exploration: probabilistic
                tau_vals = np.array([self.tau[current][j] for j in unvisited])
                eta_vals = np.array([self.eta[current][j] for j in unvisited])
                probs = (tau_vals ** self.alpha) * (eta_vals ** self.beta)
                prob_sum = probs.sum()
                if prob_sum == 0:
                    probs = np.ones(len(unvisited)) / len(unvisited)
                else:
                    probs /= prob_sum
                next_node = self.rng.choice(unvisited, p=probs)

            # Local pheromone update
            self.tau[current][next_node] = (
                (1 - self.xi) * self.tau[current][next_node] + self.xi * self.tau0
            )
            self.tau[next_node][current] = self.tau[current][next_node]

            tour.append(next_node)
            visited.add(next_node)

        return tour

    def _tour_distance(self, tour):
        total = sum(self.D[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
        total += self.D[tour[-1]][tour[0]]
        return total

    def _global_pheromone_update(self):
        """Only global-best ant deposits pheromone."""
        deposit = 1.0 / self.best_distance
        for i in range(len(self.best_tour) - 1):
            a, b = self.best_tour[i], self.best_tour[i + 1]
            self.tau[a][b] = (1 - self.rho) * self.tau[a][b] + self.rho * deposit
            self.tau[b][a] = self.tau[a][b]
        a, b = self.best_tour[-1], self.best_tour[0]
        self.tau[a][b] = (1 - self.rho) * self.tau[a][b] + self.rho * deposit
        self.tau[b][a] = self.tau[a][b]

    def solve(self):
        start_time = time.time()

        for iteration in range(self.max_iter):
            for _ in range(self.n_ants):
                tour = self._construct_tour()
                dist = self._tour_distance(tour)
                if dist < self.best_distance:
                    self.best_distance = dist
                    self.best_tour = list(tour)

            self._global_pheromone_update()
            self.convergence.append(self.best_distance)

        elapsed = time.time() - start_time
        return self.best_tour, self.best_distance, elapsed, self.convergence


def solve_acs(dist_matrix, seed=None, **kwargs):
    """Convenience function to run ACS."""
    solver = ACS_TSP(dist_matrix, seed=seed, **kwargs)
    return solver.solve()
