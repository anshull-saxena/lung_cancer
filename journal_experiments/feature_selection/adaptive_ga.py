"""
Adaptive Genetic Algorithm with grouping operator for feature selection.
Crossover and mutation rates adapt based on population fitness diversity.
"""
import random
import numpy as np
from tqdm import tqdm
try:
    from deap import base, creator, tools
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing optional dependency 'deap' required for GA-based feature selection. "
        "Install it with: pip install deap (or pip install -r requirements.txt)."
    ) from e
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import os
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in __import__('sys').path:
    __import__('sys').path.insert(0, _parent)
from config import (POP_SIZE, N_GEN, CX_PROB, MUT_PROB, INDPB,
                     CV_FOLDS, KNN_K, KNN_WEIGHTS, SEED)


class AdaptiveGA:
    """
    Adaptive GA-based feature selector.

    Adaptive rates:
      - When population fitness diversity is HIGH → increase exploitation
        (higher CX, lower MUT)
      - When diversity is LOW → increase exploration
        (lower CX, higher MUT)

    Grouping operator:
      - Features can be assigned to groups (e.g., by backbone source).
      - Crossover and mutation respect group boundaries.
    """

    def __init__(self, n_features, pop_size=POP_SIZE, n_gen=N_GEN,
                 cx_prob=CX_PROB, mut_prob=MUT_PROB, indpb=INDPB,
                 classifier=None, cv_folds=CV_FOLDS, seed=SEED,
                 l0_penalty=0.001):
        self.n_features = n_features
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.cx_prob_init = cx_prob
        self.mut_prob_init = mut_prob
        self.indpb = indpb
        self.cv_folds = cv_folds
        self.seed = seed
        self.l0_penalty = l0_penalty
        self.groups = None  # list of (start, end) tuples for grouping
        self.history = {"best_fitness": [], "avg_fitness": [],
                        "cx_rate": [], "mut_rate": [], "n_selected": []}

        if classifier is None:
            self.classifier = KNeighborsClassifier(
                n_neighbors=KNN_K, weights=KNN_WEIGHTS, n_jobs=-1
            )
        else:
            self.classifier = classifier

    def set_groups(self, group_indices):
        """
        Set feature groups for the grouping operator.

        Args:
            group_indices: list of (start, end) tuples, e.g.
                           [(0, 1024), (1024, 3072), (3072, 3584)]
        """
        self.groups = group_indices

    def _adapt_rates(self, fitnesses):
        """Adapt CX and MUT rates based on fitness diversity."""
        f_std = np.std(fitnesses)
        f_mean = np.mean(fitnesses)
        if f_mean == 0:
            diversity = 0.0
        else:
            diversity = f_std / abs(f_mean)  # coefficient of variation

        # High diversity → exploit (high CX, low MUT)
        # Low diversity  → explore (low CX, high MUT)
        cx_rate = self.cx_prob_init * (0.5 + 0.5 * diversity)
        cx_rate = np.clip(cx_rate, 0.4, 0.95)

        mut_rate = self.mut_prob_init * (1.5 - diversity)
        mut_rate = np.clip(mut_rate, 0.01, 0.3)

        return float(cx_rate), float(mut_rate)

    def _eval_fitness(self, individual, X, y):
        """Evaluate fitness: CV accuracy − L0 penalty."""
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

    def _grouped_crossover(self, ind1, ind2):
        """Two-point crossover respecting group boundaries."""
        if self.groups is None:
            return tools.cxTwoPoint(ind1, ind2)
        for start, end in self.groups:
            seg_len = end - start
            if seg_len < 3:
                continue
            pt1 = random.randint(start, end - 2)
            pt2 = random.randint(pt1 + 1, end - 1)
            ind1[pt1:pt2], ind2[pt1:pt2] = ind2[pt1:pt2], ind1[pt1:pt2]
        return ind1, ind2

    def _grouped_mutate(self, individual):
        """Bit-flip mutation respecting group boundaries."""
        if self.groups is None:
            return tools.mutFlipBit(individual, indpb=self.indpb)
        for start, end in self.groups:
            for i in range(start, end):
                if random.random() < self.indpb:
                    individual[i] = 1 - individual[i]
        return (individual,)

    def run(self, X, y):
        """
        Run the adaptive GA.

        Args:
            X: feature matrix (N, D)
            y: labels (N,)

        Returns:
            selected_indices: np.array of selected feature indices
            history: dict of per-generation stats
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        # DEAP setup — guard against re-creation
        if "FitnessMax" not in creator.__dict__:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMax)

        tb = base.Toolbox()
        tb.register("attr_bool", random.randint, 0, 1)
        tb.register("individual", tools.initRepeat, creator.Individual,
                     tb.attr_bool, self.n_features)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("select", tools.selTournament, tournsize=3)

        pop = tb.population(n=self.pop_size)

        # Evaluate initial population
        for ind in pop:
            ind.fitness.values = self._eval_fitness(ind, X, y)

        cx_rate = self.cx_prob_init
        mut_rate = self.mut_prob_init

        pbar = tqdm(range(self.n_gen), desc="  GA Search", unit="gen", leave=False)
        for gen in pbar:
            # Adapt rates
            fitnesses = [ind.fitness.values[0] for ind in pop]
            cx_rate, mut_rate = self._adapt_rates(fitnesses)
            best_f = max(fitnesses)
            pbar.set_postfix(best=f"{best_f:.4f}", cx=f"{cx_rate:.2f}", mut=f"{mut_rate:.2f}")

            # Selection
            offspring = tb.select(pop, len(pop))
            offspring = list(map(tb.clone, offspring))

            # Crossover
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cx_rate:
                    self._grouped_crossover(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            # Mutation
            for ind in offspring:
                if random.random() < mut_rate:
                    self._grouped_mutate(ind)
                    del ind.fitness.values

            # Evaluate
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = self._eval_fitness(ind, X, y)

            pop[:] = offspring

            # Track stats
            fits = [ind.fitness.values[0] for ind in pop]
            best_ind = tools.selBest(pop, 1)[0]
            n_sel = sum(best_ind)
            self.history["best_fitness"].append(max(fits))
            self.history["avg_fitness"].append(np.mean(fits))
            self.history["cx_rate"].append(cx_rate)
            self.history["mut_rate"].append(mut_rate)
            self.history["n_selected"].append(n_sel)

        # Best individual
        best = tools.selBest(pop, 1)[0]
        selected = np.array([i for i, b in enumerate(best) if b == 1], dtype=int)
        print(f"  GA done: selected {len(selected)}/{self.n_features} features")
        return selected, self.history
