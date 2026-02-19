"""
Standard (non-adaptive) Genetic Algorithm for feature selection.
Fixed crossover/mutation rates, no grouping operator. Used as ablation baseline.
"""
import random
import numpy as np
from deap import base, creator, tools
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import os
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in __import__('sys').path:
    __import__('sys').path.insert(0, _parent)
from config import (POP_SIZE, N_GEN, CX_PROB, MUT_PROB, INDPB,
                     CV_FOLDS, KNN_K, KNN_WEIGHTS, SEED)


class BaselineGA:
    """
    Standard GA with fixed rates and no grouping.
    Same interface as AdaptiveGA for fair comparison.
    """

    def __init__(self, n_features, pop_size=POP_SIZE, n_gen=N_GEN,
                 cx_prob=CX_PROB, mut_prob=MUT_PROB, indpb=INDPB,
                 classifier=None, cv_folds=CV_FOLDS, seed=SEED,
                 l0_penalty=0.001):
        self.n_features = n_features
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.indpb = indpb
        self.cv_folds = cv_folds
        self.seed = seed
        self.l0_penalty = l0_penalty
        self.history = {"best_fitness": [], "avg_fitness": [], "n_selected": []}

        if classifier is None:
            self.classifier = KNeighborsClassifier(
                n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1
            )
        else:
            self.classifier = classifier

    def _eval_fitness(self, individual, X, y):
        """Evaluate fitness: CV accuracy - L0 penalty."""
        idx = [i for i, b in enumerate(individual) if b == 1]
        if len(idx) < 2:
            return (0.0,)
        Xs = X[:, idx]
        scores = cross_val_score(
            self.classifier, Xs, y, cv=self.cv_folds,
            scoring="accuracy", n_jobs=-1
        )
        fitness = scores.mean() - self.l0_penalty * (len(idx) / self.n_features)
        return (float(fitness),)

    def run(self, X, y):
        """
        Run standard GA feature selection.

        Returns:
            selected_indices: np.array
            history: dict
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        if "FitnessMax" not in creator.__dict__:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMax)

        tb = base.Toolbox()
        tb.register("attr_bool", random.randint, 0, 1)
        tb.register("individual", tools.initRepeat, creator.Individual,
                     tb.attr_bool, self.n_features)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("evaluate", self._eval_fitness)
        tb.register("mate", tools.cxTwoPoint)
        tb.register("mutate", tools.mutFlipBit, indpb=self.indpb)
        tb.register("select", tools.selTournament, tournsize=3)

        pop = tb.population(n=self.pop_size)

        # Evaluate initial population
        for ind in pop:
            ind.fitness.values = self._eval_fitness(ind, X, y)

        for gen in range(self.n_gen):
            offspring = tb.select(pop, len(pop))
            offspring = list(map(tb.clone, offspring))

            # Fixed-rate crossover
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_prob:
                    tb.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            # Fixed-rate mutation
            for ind in offspring:
                if random.random() < self.mut_prob:
                    tb.mutate(ind)
                    del ind.fitness.values

            # Evaluate invalid
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = self._eval_fitness(ind, X, y)

            pop[:] = offspring

            fits = [ind.fitness.values[0] for ind in pop]
            best_ind = tools.selBest(pop, 1)[0]
            self.history["best_fitness"].append(max(fits))
            self.history["avg_fitness"].append(np.mean(fits))
            self.history["n_selected"].append(sum(best_ind))

            if (gen + 1) % 10 == 0 or gen == 0:
                print(f"  Gen {gen+1:3d}/{self.n_gen}: "
                      f"best={max(fits):.4f}  avg={np.mean(fits):.4f}  "
                      f"sel={sum(best_ind)}")

        best = tools.selBest(pop, 1)[0]
        selected = np.array([i for i, b in enumerate(best) if b == 1], dtype=int)
        print(f"  Baseline GA done: selected {len(selected)}/{self.n_features} features")
        return selected, self.history
