"""
NSGA-II multi-objective feature selection.
Objectives: maximize classification accuracy, minimize feature count.
"""
import random
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (NSGA_POP, NSGA_GEN, CX_PROB, MUT_PROB, INDPB,
                     CV_FOLDS, KNN_K, KNN_WEIGHTS, SEED)


class NSGA2Selector:
    """
    NSGA-II based multi-objective feature selector.

    Objectives:
        1. Maximize classification accuracy (CV)
        2. Minimize number of selected features
    """

    def __init__(self, n_features, pop_size=NSGA_POP, n_gen=NSGA_GEN,
                 cx_prob=CX_PROB, mut_prob=MUT_PROB, indpb=INDPB,
                 classifier=None, cv_folds=CV_FOLDS, seed=SEED):
        self.n_features = n_features
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.indpb = indpb
        self.cv_folds = cv_folds
        self.seed = seed
        self.groups = None
        self.history = {"pareto_size": [], "best_acc": [],
                        "min_features": [], "avg_features": []}

        if classifier is None:
            self.classifier = KNeighborsClassifier(
                n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1
            )
        else:
            self.classifier = classifier

    def set_groups(self, group_indices):
        """Set feature groups for grouping operator."""
        self.groups = group_indices

    def _evaluate(self, individual, X, y):
        """
        Evaluate an individual.
        Returns (accuracy, n_features) — we maximize accuracy and minimize features.
        Internally we store both as-is; domination logic handles the directions.
        """
        idx = [i for i, b in enumerate(individual) if b == 1]
        if len(idx) < 2:
            return (0.0, self.n_features)
        Xs = X[:, idx]
        scores = cross_val_score(
            self.classifier, Xs, y, cv=self.cv_folds,
            scoring="accuracy", n_jobs=-1
        )
        return (float(scores.mean()), len(idx))

    @staticmethod
    def _dominates(obj_a, obj_b):
        """
        Check if solution a dominates b.
        obj = (accuracy, n_features): maximize accuracy, minimize features.
        """
        better_in_any = False
        # accuracy: higher is better
        if obj_a[0] < obj_b[0]:
            return False
        if obj_a[0] > obj_b[0]:
            better_in_any = True
        # features: lower is better
        if obj_a[1] > obj_b[1]:
            return False
        if obj_a[1] < obj_b[1]:
            better_in_any = True
        return better_in_any

    def _fast_non_dominated_sort(self, population):
        """NSGA-II fast non-dominated sort. Returns list of fronts (lists of indices)."""
        n = len(population)
        domination_count = [0] * n
        dominated_set = [[] for _ in range(n)]
        rank = [0] * n
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(population[i]["obj"], population[j]["obj"]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(population[j]["obj"], population[i]["obj"]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
            if domination_count[i] == 0:
                rank[i] = 0
                fronts[0].append(i)

        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        rank[j] = front_idx + 1
                        next_front.append(j)
            front_idx += 1
            fronts.append(next_front)

        # Remove trailing empty front
        if not fronts[-1]:
            fronts.pop()

        for i in range(n):
            population[i]["rank"] = rank[i]
        return fronts

    def _crowding_distance(self, population, front):
        """Compute crowding distance for individuals in a front."""
        n = len(front)
        if n <= 2:
            for idx in front:
                population[idx]["crowd_dist"] = float("inf")
            return

        for idx in front:
            population[idx]["crowd_dist"] = 0.0

        for m in range(2):  # two objectives
            sorted_front = sorted(front, key=lambda i: population[i]["obj"][m])
            obj_min = population[sorted_front[0]]["obj"][m]
            obj_max = population[sorted_front[-1]]["obj"][m]
            population[sorted_front[0]]["crowd_dist"] = float("inf")
            population[sorted_front[-1]]["crowd_dist"] = float("inf")
            obj_range = obj_max - obj_min
            if obj_range == 0:
                continue
            for k in range(1, n - 1):
                diff = (population[sorted_front[k + 1]]["obj"][m]
                        - population[sorted_front[k - 1]]["obj"][m])
                population[sorted_front[k]]["crowd_dist"] += diff / obj_range

    def _tournament_selection(self, population, k=2):
        """Binary tournament: prefer lower rank, then higher crowding distance."""
        candidates = random.sample(range(len(population)), k)
        best = candidates[0]
        for c in candidates[1:]:
            if (population[c]["rank"] < population[best]["rank"] or
                (population[c]["rank"] == population[best]["rank"] and
                 population[c]["crowd_dist"] > population[best]["crowd_dist"])):
                best = c
        return population[best]["genes"][:]

    def _crossover(self, p1, p2):
        """Uniform crossover, optionally group-aware."""
        c1, c2 = p1[:], p2[:]
        if self.groups is not None:
            for start, end in self.groups:
                for i in range(start, end):
                    if random.random() < 0.5:
                        c1[i], c2[i] = c2[i], c1[i]
        else:
            for i in range(len(c1)):
                if random.random() < 0.5:
                    c1[i], c2[i] = c2[i], c1[i]
        return c1, c2

    def _mutate(self, individual):
        """Bit-flip mutation."""
        if self.groups is not None:
            for start, end in self.groups:
                for i in range(start, end):
                    if random.random() < self.indpb:
                        individual[i] = 1 - individual[i]
        else:
            for i in range(len(individual)):
                if random.random() < self.indpb:
                    individual[i] = 1 - individual[i]
        return individual

    def run(self, X, y):
        """
        Run NSGA-II feature selection.

        Returns:
            best_indices: np.array — features selected by the best compromise solution
            pareto_front: list of (accuracy, n_features) tuples
            history: dict of per-generation stats
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Initialize population
        pop = []
        for _ in range(self.pop_size):
            genes = [random.randint(0, 1) for _ in range(self.n_features)]
            obj = self._evaluate(genes, X, y)
            pop.append({"genes": genes, "obj": obj, "rank": 0, "crowd_dist": 0.0})

        for gen in range(self.n_gen):
            # Generate offspring
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._tournament_selection(pop)
                p2 = self._tournament_selection(pop)
                if random.random() < self.cx_prob:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]
                if random.random() < self.mut_prob:
                    c1 = self._mutate(c1)
                if random.random() < self.mut_prob:
                    c2 = self._mutate(c2)
                offspring.append(c1)
                if len(offspring) < self.pop_size:
                    offspring.append(c2)

            # Evaluate offspring
            for child_genes in offspring:
                obj = self._evaluate(child_genes, X, y)
                pop.append({"genes": child_genes, "obj": obj,
                            "rank": 0, "crowd_dist": 0.0})

            # Non-dominated sort on combined population
            fronts = self._fast_non_dominated_sort(pop)

            # Assign crowding distance
            for front in fronts:
                self._crowding_distance(pop, front)

            # Select next generation
            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend([pop[i] for i in front])
                else:
                    # Sort by crowding distance (descending) and fill
                    remaining = self.pop_size - len(new_pop)
                    sorted_front = sorted(
                        front, key=lambda i: pop[i]["crowd_dist"], reverse=True
                    )
                    new_pop.extend([pop[i] for i in sorted_front[:remaining]])
                    break
            pop = new_pop

            # Track history
            pareto_front = [p["obj"] for p in pop if p["rank"] == 0]
            accs = [o[0] for o in pareto_front]
            feats = [o[1] for o in pareto_front]
            self.history["pareto_size"].append(len(pareto_front))
            self.history["best_acc"].append(max(accs) if accs else 0)
            self.history["min_features"].append(min(feats) if feats else 0)
            all_feats = [p["obj"][1] for p in pop]
            self.history["avg_features"].append(np.mean(all_feats))

            if (gen + 1) % 10 == 0 or gen == 0:
                print(f"  Gen {gen+1:3d}/{self.n_gen}: "
                      f"pareto={len(pareto_front)}  "
                      f"best_acc={max(accs):.4f}  "
                      f"min_feat={min(feats)}")

        # Extract Pareto front
        pareto_front = [(p["obj"], p["genes"]) for p in pop if p["rank"] == 0]
        pareto_objs = [pf[0] for pf in pareto_front]

        # Best compromise: highest accuracy among solutions with ≤ median features
        if pareto_objs:
            median_feat = np.median([o[1] for o in pareto_objs])
            candidates = [(obj, genes) for obj, genes in pareto_front
                          if obj[1] <= median_feat]
            if not candidates:
                candidates = pareto_front
            best_obj, best_genes = max(candidates, key=lambda x: x[0][0])
        else:
            best_genes = pop[0]["genes"]
            best_obj = pop[0]["obj"]

        selected = np.array([i for i, b in enumerate(best_genes) if b == 1], dtype=int)
        print(f"  NSGA-II done: selected {len(selected)}/{self.n_features} features "
              f"(acc={best_obj[0]:.4f})")
        return selected, pareto_objs, self.history
