"""Permutation genetic algorithm for CVRP."""
import random
import numpy as np
from src.decoder import decode_permutation, calculate_total_distance
from src.utils import OptimizationResult, timed, seed_everything

class GeneticAlgorithm:
    """GA with tournament selection, OX crossover, inversion mutation, and elitism."""
    def __init__(self, instance, config: dict, seed: int = 0):
        self.instance, self.cfg, self.seed = instance, config, seed
        seed_everything(seed); self.evaluations = 0

    def _evaluate(self, permutation):
        routes = decode_permutation(permutation, self.instance.demands, self.instance.capacity)
        value = calculate_total_distance(routes, self.instance.distance_matrix); self.evaluations += 1
        return value, routes

    @staticmethod
    def _ox(a, b):
        n = len(a); left, right = sorted(random.sample(range(n), 2)); child = [-1] * n
        child[left:right] = a[left:right]; used = set(child[left:right]); pos = right % n
        for gene in b:
            if gene not in used: child[pos] = gene; used.add(gene); pos = (pos + 1) % n
        return child

    def _mutate(self, p):
        if random.random() < self.cfg["mutation_rate"]:
            i, j = sorted(random.sample(range(len(p)), 2)); p[i:j] = reversed(p[i:j])
        return p

    def optimize(self, max_evaluations: int) -> OptimizationResult:
        """Run until the common objective evaluation budget is exhausted."""
        start = timed(); n = self.instance.num_customers; size = self.cfg["population_size"]
        population = [random.sample(range(1, n + 1), n) for _ in range(size)]
        scored = [(self._evaluate(p), p) for p in population]
        best_val, best_routes = min((s[0] for s in scored), key=lambda x: x[0])
        best_perm = next(p for (s, p) in scored if s[0] == best_val); convergence = [best_val]; history=[]; generation=0
        while self.evaluations < max_evaluations:
            generation += 1; scored.sort(key=lambda x: x[0][0]); elite = scored[:self.cfg["elite_size"]]
            new_pop = [p.copy() for _, p in elite]
            while len(new_pop) < size and self.evaluations + 2 <= max_evaluations:
                def select(): return min(random.sample(scored, self.cfg["tournament_size"]), key=lambda x: x[0][0])[1]
                p1, p2 = select(), select(); child = self._ox(p1, p2) if random.random() < self.cfg["crossover_rate"] else p1.copy()
                new_pop.append(self._mutate(child))
            scored = [(self._evaluate(p), p) for p in new_pop]
            vals = np.array([s[0] for s, _ in scored]); current = int(np.argmin(vals))
            if vals[current] < best_val: best_val, best_perm = float(vals[current]), scored[current][1].copy(); best_routes = scored[current][0][1]
            convergence.append(best_val); history.append({"generation":generation,"best_distance":best_val,"mean_distance":float(vals.mean()),"std_distance":float(vals.std()),"fitness_evaluations":self.evaluations})
        return OptimizationResult(best_val, best_routes, best_perm, history, timed()-start, self.evaluations, convergence)
