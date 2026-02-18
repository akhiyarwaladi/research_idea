"""
Genetic Algorithm for TSP

Uses Order Crossover (OX), swap mutation, tournament selection, and elitism.
"""

import numpy as np
import time


class GA_TSP:
    """Genetic Algorithm solver for TSP."""

    def __init__(
        self,
        dist_matrix,
        pop_size=100,
        generations=500,
        crossover_rate=0.8,
        mutation_rate=0.02,
        tournament_k=3,
        elite_ratio=0.1,
        stagnation_limit=100,
        seed=None,
    ):
        self.D = np.array(dist_matrix, dtype=np.float64)
        self.n = len(self.D)
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.elite_count = max(1, int(pop_size * elite_ratio))
        self.stagnation_limit = stagnation_limit

        self.rng = np.random.default_rng(seed)

        self.best_tour = None
        self.best_distance = float("inf")
        self.convergence = []

    def _tour_distance_single(self, tour):
        """Compute tour distance for a single tour (vectorized)."""
        return self.D[tour[:-1], tour[1:]].sum() + self.D[tour[-1], tour[0]]

    def _population_fitness(self, population):
        """Compute fitness for entire population at once."""
        pop = np.asarray(population)
        # Vectorized: compute all edge costs for all tours
        from_nodes = pop[:, :-1]
        to_nodes = pop[:, 1:]
        edge_costs = self.D[from_nodes, to_nodes].sum(axis=1)
        # Add return edge
        edge_costs += self.D[pop[:, -1], pop[:, 0]]
        return edge_costs

    def _init_population(self):
        """Generate random permutation population as numpy array."""
        pop = np.empty((self.pop_size, self.n), dtype=np.intp)
        base = np.arange(self.n)
        for i in range(self.pop_size):
            pop[i] = base.copy()
            self.rng.shuffle(pop[i])
        return pop

    def _tournament_selection(self, population, fitnesses):
        """Select individual via tournament selection."""
        indices = self.rng.choice(len(population), size=self.tournament_k, replace=False)
        best_idx = indices[np.argmin(fitnesses[indices])]
        return population[best_idx].copy()

    def _order_crossover(self, parent1, parent2):
        """Order Crossover (OX) operator using set for O(1) lookup."""
        n = self.n
        pts = self.rng.choice(n, size=2, replace=False)
        start, end = int(pts.min()), int(pts.max())

        child = np.full(n, -1, dtype=np.intp)
        child[start:end + 1] = parent1[start:end + 1]

        # Use set for O(1) membership test
        in_child = set(child[start:end + 1].tolist())
        fill_values = [g for g in parent2 if g not in in_child]

        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill_values[idx]
                idx += 1

        return child

    def _swap_mutation(self, tour):
        """Swap two random positions."""
        pts = self.rng.choice(self.n, size=2, replace=False)
        i, j = int(pts[0]), int(pts[1])
        tour[i], tour[j] = tour[j], tour[i]
        return tour

    def solve(self):
        start_time = time.time()

        population = self._init_population()
        fitnesses = self._population_fitness(population)

        # Track global best
        best_idx = np.argmin(fitnesses)
        self.best_tour = population[best_idx].tolist()
        self.best_distance = fitnesses[best_idx]
        no_improve_count = 0

        for gen in range(self.generations):
            # Elitism: preserve top individuals
            elite_indices = np.argsort(fitnesses)[:self.elite_count]
            new_pop = [population[i].copy() for i in elite_indices]

            # Fill rest of population
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)

                if self.rng.random() < self.crossover_rate:
                    child = self._order_crossover(p1, p2)
                else:
                    child = p1.copy()

                if self.rng.random() < self.mutation_rate:
                    child = self._swap_mutation(child)

                new_pop.append(child)

            population = np.array(new_pop[:self.pop_size])
            fitnesses = self._population_fitness(population)

            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < self.best_distance:
                self.best_distance = fitnesses[gen_best_idx]
                self.best_tour = population[gen_best_idx].tolist()
                no_improve_count = 0
            else:
                no_improve_count += 1

            self.convergence.append(self.best_distance)

            # Early stopping
            if no_improve_count >= self.stagnation_limit:
                remaining = self.generations - gen - 1
                self.convergence.extend([self.best_distance] * remaining)
                break

        elapsed = time.time() - start_time
        return self.best_tour, self.best_distance, elapsed, self.convergence


def solve_ga(dist_matrix, seed=None, **kwargs):
    """Convenience function to run GA."""
    solver = GA_TSP(dist_matrix, seed=seed, **kwargs)
    return solver.solve()
