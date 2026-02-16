"""
Genetic Algorithm for TSP

Uses Order Crossover (OX), swap mutation, and tournament selection.
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
        seed=None,
    ):
        self.D = np.array(dist_matrix, dtype=float)
        self.n = len(self.D)
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k

        self.rng = np.random.default_rng(seed)

        self.best_tour = None
        self.best_distance = float("inf")
        self.convergence = []

    def _tour_distance(self, tour):
        total = sum(self.D[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
        total += self.D[tour[-1]][tour[0]]
        return total

    def _init_population(self):
        """Generate random permutation population."""
        pop = []
        for _ in range(self.pop_size):
            individual = list(range(self.n))
            self.rng.shuffle(individual)
            pop.append(individual)
        return pop

    def _tournament_selection(self, population, fitnesses):
        """Select individual via tournament selection."""
        indices = self.rng.choice(len(population), size=self.tournament_k, replace=False)
        best_idx = indices[np.argmin([fitnesses[i] for i in indices])]
        return list(population[best_idx])

    def _order_crossover(self, parent1, parent2):
        """Order Crossover (OX) operator."""
        n = self.n
        start, end = sorted(self.rng.choice(n, size=2, replace=False))

        child = [-1] * n
        child[start : end + 1] = parent1[start : end + 1]

        fill_values = [g for g in parent2 if g not in child]
        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill_values[idx]
                idx += 1

        return child

    def _swap_mutation(self, tour):
        """Swap two random positions."""
        i, j = self.rng.choice(self.n, size=2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]
        return tour

    def solve(self):
        start_time = time.time()

        population = self._init_population()
        fitnesses = np.array([self._tour_distance(t) for t in population])

        # Track global best
        best_idx = np.argmin(fitnesses)
        self.best_tour = list(population[best_idx])
        self.best_distance = fitnesses[best_idx]

        for gen in range(self.generations):
            new_pop = []

            for _ in range(self.pop_size):
                # Selection
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)

                # Crossover
                if self.rng.random() < self.crossover_rate:
                    child = self._order_crossover(p1, p2)
                else:
                    child = list(p1)

                # Mutation
                if self.rng.random() < self.mutation_rate:
                    child = self._swap_mutation(child)

                new_pop.append(child)

            population = new_pop
            fitnesses = np.array([self._tour_distance(t) for t in population])

            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < self.best_distance:
                self.best_distance = fitnesses[gen_best_idx]
                self.best_tour = list(population[gen_best_idx])

            self.convergence.append(self.best_distance)

        elapsed = time.time() - start_time
        return self.best_tour, self.best_distance, elapsed, self.convergence


def solve_ga(dist_matrix, seed=None, **kwargs):
    """Convenience function to run GA."""
    solver = GA_TSP(dist_matrix, seed=seed, **kwargs)
    return solver.solve()
