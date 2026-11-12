"""
Selection operators for the spider Genetic Algorithm.

The GA is assumed to be a maximisation problem: a larger fitness value means
that a chromosome is better. Each chromosome may be represented by any
sequence of 24 joint angles, while the population is represented as a list of
chromosomes and the fitness values as a list of numbers with matching indices.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple


Chromosome = Sequence[float]


def tournament_selection(
    population: Sequence[Chromosome],
    fitnesses: Sequence[float],
    number_to_select: int,
    tournament_size: int = 3,
    rng: random.Random | None = None,
) -> List[List[float]]:
    """Select chromosomes using tournament selection.

    For every parent requested, ``tournament_size`` candidates are sampled
    without replacement from the population. The candidate with the highest
    fitness wins and is copied into the selected-parent list.

    Parameters
    ----------
    population:
        Sequence of chromosomes. For this coursework each chromosome should
        contain 24 joint angles in radians.
    fitnesses:
        Fitness value for each chromosome. Higher is better.
    number_to_select:
        Number of parents required by the next GA stage.
    tournament_size:
        Number of candidates competing in each tournament.
    rng:
        Optional local random-number generator, useful for reproducible tests.

    Returns
    -------
    list[list[float]]
        Copies of selected chromosomes. Copies prevent crossover/mutation from
        accidentally modifying the original population.
    """
    if rng is None:
        rng = random.Random()

    _validate_inputs(population, fitnesses, number_to_select, tournament_size)

    selected: List[List[float]] = []
    population_indices = list(range(len(population)))

    for _ in range(number_to_select):
        competitors = rng.sample(population_indices, tournament_size)
        winner_index = max(competitors, key=lambda index: fitnesses[index])
        selected.append(list(population[winner_index]))

    return selected


def elitism(
    population: Sequence[Chromosome],
    fitnesses: Sequence[float],
    elite_count: int,
) -> List[List[float]]:
    """Return copies of the ``elite_count`` highest-fitness chromosomes."""
    if len(population) != len(fitnesses):
        raise ValueError("population and fitnesses must have the same length")
    if elite_count < 0 or elite_count > len(population):
        raise ValueError("elite_count must be between 0 and population size")
    if any(not isinstance(value, (int, float)) for value in fitnesses):
        raise TypeError("fitnesses must contain numeric values")

    ranked_indices = sorted(
        range(len(population)), key=lambda index: fitnesses[index], reverse=True
    )
    return [list(population[index]) for index in ranked_indices[:elite_count]]


def select_for_next_generation(
    population: Sequence[Chromosome],
    fitnesses: Sequence[float],
    elite_count: int = 2,
    tournament_size: int = 3,
    rng: random.Random | None = None,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Select elites and parents for a generational GA.

    The caller can copy ``elites`` directly into the next generation and use
    ``parents`` as input to crossover. Parent selection requests the same
    number of chromosomes as the current population, which makes it easy to
    pair parents in the crossover stage.
    """
    if not population:
        raise ValueError("population must not be empty")

    elites = elitism(population, fitnesses, elite_count)
    parents = tournament_selection(
        population=population,
        fitnesses=fitnesses,
        number_to_select=len(population),
        tournament_size=tournament_size,
        rng=rng,
    )
    return elites, parents


def _validate_inputs(
    population: Sequence[Chromosome],
    fitnesses: Sequence[float],
    number_to_select: int,
    tournament_size: int,
) -> None:
    """Validate arguments and fail early with clear messages."""
    if not population:
        raise ValueError("population must not be empty")
    if len(population) != len(fitnesses):
        raise ValueError("population and fitnesses must have the same length")
    if number_to_select < 0:
        raise ValueError("number_to_select must be non-negative")
    if tournament_size < 1 or tournament_size > len(population):
        raise ValueError("tournament_size must be between 1 and population size")
    if any(not isinstance(value, (int, float)) for value in fitnesses):
        raise TypeError("fitnesses must contain numeric values")
    if any(len(chromosome) != 24 for chromosome in population):
        raise ValueError("each spider chromosome must contain exactly 24 angles")


if __name__ == "__main__":
    # Small smoke test showing the intended interface.
    generator = random.Random(7)
    demo_population = [[float(gene) for gene in range(24)] for _ in range(6)]
    demo_fitnesses = [1.0, 4.0, 2.0, 6.0, 3.0, 5.0]

    elites, parents = select_for_next_generation(
        demo_population,
        demo_fitnesses,
        elite_count=2,
        tournament_size=3,
        rng=generator,
    )

    assert elites[0] == demo_population[3]
    assert elites[1] == demo_population[5]
    assert len(parents) == len(demo_population)
    assert all(len(parent) == 24 for parent in parents)
    print("Selection smoke test passed.")
