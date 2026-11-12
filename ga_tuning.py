"""Quick GA tuning: compare three candidate configurations.

This lightweight version is intended to finish quickly and help choose a good
mutation rate / tournament size combination without the long grid-search cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from genetic_algorithm import run_genetic_algorithm


@dataclass
class TrialResult:
    label: str
    population_size: int
    generations: int
    elite_count: int
    tournament_size: int
    mutation_rate: float
    mutation_scale: float
    seed: int
    best_final_score: float
    history: List[float]


def run_trial(
    label: str,
    population_size: int,
    generations: int,
    elite_count: int,
    tournament_size: int,
    mutation_rate: float,
    mutation_scale: float,
    seed: int,
) -> TrialResult:
    _, final_scores, history = run_genetic_algorithm(
        population_size=population_size,
        generations=generations,
        elite_count=elite_count,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        mutation_scale=mutation_scale,
        seed=seed,
    )
    return TrialResult(
        label=label,
        population_size=population_size,
        generations=generations,
        elite_count=elite_count,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        mutation_scale=mutation_scale,
        seed=seed,
        best_final_score=max(final_scores) if final_scores else 0.0,
        history=history,
    )


if __name__ == "__main__":
    trials = [
        run_trial(
            label="A",
            population_size=20,
            generations=10,
            elite_count=2,
            tournament_size=3,
            mutation_rate=0.05,
            mutation_scale=0.2,
            seed=0,
        ),
        run_trial(
            label="B",
            population_size=20,
            generations=10,
            elite_count=1,
            tournament_size=2,
            mutation_rate=0.10,
            mutation_scale=0.2,
            seed=0,
        ),
        run_trial(
            label="C",
            population_size=30,
            generations=10,
            elite_count=2,
            tournament_size=3,
            mutation_rate=0.05,
            mutation_scale=0.2,
            seed=0,
        ),
    ]

    trials = sorted(trials, key=lambda t: t.best_final_score, reverse=True)

    print("Quick GA tuning comparison")
    for trial in trials:
        print(
            f"{trial.label}: pop={trial.population_size}, "
            f"elite={trial.elite_count}, tourney={trial.tournament_size}, "
            f"mut={trial.mutation_rate}, final_best={trial.best_final_score:.4f}"
        )
        print(f"  history: {[round(x, 3) for x in trial.history]}")
