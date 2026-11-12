"""Genetic algorithm operators for evolving spider gait chromosomes.

This project represents a chromosome as a list of 24 joint angles. The
fitness function in ``fitness.py`` expects a full gait (a sequence of poses), so
for evaluation we repeatedly tile the chromosome into a 300-frame gait.
"""

from __future__ import annotations

import random
from typing import Callable, List, Sequence, Tuple

import numpy as np

from fitness import fitness
from intial_population import create_population
from joint_limits import lower, upper
from selection import select_for_next_generation

Chromosome = Sequence[float]


def single_point_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    rng: random.Random | None = None,
) -> Tuple[List[float], List[float]]:
    """Return two children formed by one-point crossover."""
    if rng is None:
        rng = random.Random()
    if len(parent_a) != len(parent_b):
        raise ValueError("parents must have the same chromosome length")
    if len(parent_a) < 2:
        return list(parent_a), list(parent_b)

    crossover_point = rng.randint(1, len(parent_a) - 1)
    child_a = list(parent_a[:crossover_point]) + list(parent_b[crossover_point:])
    child_b = list(parent_b[:crossover_point]) + list(parent_a[crossover_point:])
    return child_a, child_b


def mutate(
    chromosome: Chromosome,
    mutation_rate: float = 0.05,
    mutation_scale: float = 0.2,
    rng: random.Random | None = None,
) -> List[float]:
    """Mutate a chromosome by perturbing each gene with a small probability."""
    if rng is None:
        rng = random.Random()
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1")

    mutated = list(chromosome)
    for i in range(len(mutated)):
        if rng.random() < mutation_rate:
            delta = rng.uniform(-mutation_scale, mutation_scale)
            mutated[i] = float(np.clip(mutated[i] + delta, lower[i], upper[i]))
    return mutated


def evaluate_population(
    population: Sequence[Chromosome],
    frames_per_gait: int = 300,
    fitness_fn: Callable[[np.ndarray], float] = fitness,
) -> List[float]:
    """Evaluate each chromosome by converting it to a repeated gait."""
    scores: List[float] = []
    for chromosome in population:
        gait = np.tile(np.asarray(chromosome, dtype=float), (frames_per_gait, 1))
        scores.append(float(fitness_fn(gait)))
    return scores


def evolve_population(
    population: Sequence[Chromosome],
    fitnesses: Sequence[float],
    elite_count: int = 2,
    tournament_size: int = 3,
    mutation_rate: float = 0.05,
    mutation_scale: float = 0.2,
    rng: random.Random | None = None,
) -> List[List[float]]:
    """Create the next generation using elites, tournament selection, crossover, mutation."""
    if rng is None:
        rng = random.Random()

    elites, parents = select_for_next_generation(
        population=population,
        fitnesses=fitnesses,
        elite_count=elite_count,
        tournament_size=tournament_size,
        rng=rng,
    )

    next_generation: List[List[float]] = [list(chrom) for chrom in elites]

    while len(next_generation) < len(population):
        parent_a, parent_b = rng.sample(parents, 2)
        child_a, child_b = single_point_crossover(parent_a, parent_b, rng=rng)
        next_generation.append(mutate(child_a, mutation_rate, mutation_scale, rng=rng))
        if len(next_generation) < len(population):
            next_generation.append(mutate(child_b, mutation_rate, mutation_scale, rng=rng))

    if len(next_generation) > len(population):
        next_generation = next_generation[: len(population)]

    return next_generation


def run_genetic_algorithm(
    population_size: int = 30,
    generations: int = 50,
    elite_count: int = 2,
    tournament_size: int = 3,
    mutation_rate: float = 0.05,
    mutation_scale: float = 0.2,
    seed: int | None = None,
) -> Tuple[List[List[float]], List[float], List[float]]:
    """Run a simple GA and return the final population, best scores, and generation history."""
    rng = random.Random(seed)
    population = create_population(population_size)
    best_scores: List[float] = []
    generation_best: List[float] = []

    for _ in range(generations):
        fitness_values = evaluate_population(population)
        best_index = max(range(len(population)), key=lambda i: fitness_values[i])
        current_best = fitness_values[best_index]
        best_scores.append(current_best)
        generation_best.append(current_best)

        population = evolve_population(
            population=population,
            fitnesses=fitness_values,
            elite_count=elite_count,
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
            rng=rng,
        )

    final_fits = evaluate_population(population)
    return population, final_fits, generation_best


if __name__ == "__main__":
    test_population = create_population(10)
    scores = evaluate_population(test_population)
    best = max(scores)
    print("Best initial score:", best)

    evolved_population, final_scores, history = run_genetic_algorithm(
        population_size=10,
        generations=10,
        elite_count=2,
        tournament_size=3,
        mutation_rate=0.08,
        mutation_scale=0.25,
        seed=7,
    )

    print("Best final score:", max(final_scores))
    print("Score history (first 10 generations):", history[:10])
    print("Population size:", len(evolved_population))
