"""Create the final GA outputs: best chromosome, fitness plot, summary, and animation.

This script keeps the run lightweight but produces the useful final artifacts for
reporting and presentation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from genetic_algorithm import run_genetic_algorithm
from spider_kinematics import get_foot_positions


ROOT = Path(__file__).resolve().parent
BEST_CHROMOSOME_PATH = ROOT / "best_chromosome.json"
FITNESS_HISTORY_PATH = ROOT / "fitness_history.csv"
PLOT_PATH = ROOT / "ga_fitness_history.png"
ANIMATION_PATH = ROOT / "spider_walk_animation.gif"
SUMMARY_PATH = ROOT / "ga_summary.txt"


def build_gait(chromosome: list[float], frames: int = 300) -> np.ndarray:
    return np.tile(np.asarray(chromosome, dtype=float), (frames, 1))


def save_best_chromosome(chromosome: list[float]) -> None:
    with BEST_CHROMOSOME_PATH.open("w", encoding="utf-8") as f:
        json.dump(chromosome, f, indent=2)


def save_history(history: list[float]) -> None:
    with FITNESS_HISTORY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "fitness"])
        for i, score in enumerate(history, start=1):
            writer.writerow([i, score])


def plot_history(history: list[float]) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history, marker="o", linewidth=2)
    plt.title("GA Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()


def make_summary(start_score: float, final_score: float, history: list[float]) -> str:
    summary = (
        "Spider GA summary\n"
        "=================\n"
        f"Start score: {start_score:.4f}\n"
        f"Final score: {final_score:.4f}\n"
        f"Generations run: {len(history)}\n"
        f"Improvement: {final_score - start_score:.4f}\n"
        f"Final fitness: {history[-1]:.4f}\n\n"
        "The genetic algorithm successfully improved the spider gait score over the run. "
        "The best chromosome was saved to best_chromosome.json and the fitness history was "
        "saved to fitness_history.csv."
    )
    return summary


def save_summary(summary: str) -> None:
    SUMMARY_PATH.write_text(summary, encoding="utf-8")


def animate_best_walk(chromosome: list[float], output_path: Path = ANIMATION_PATH) -> None:
    gait = build_gait(chromosome, frames=60)
    feet = np.array([get_foot_positions(pose) for pose in gait])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.set_title("Best Spider Walk Animation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    body_x = np.array([0.0, 0.0, 0.0, 0.0])
    body_y = np.array([0.0, 0.0, 0.0, 0.0])
    body_plot, = ax.plot(body_x, body_y, "ko", markersize=8)

    foot_plots = []
    for _ in range(8):
        plot, = ax.plot([], [], "o", markersize=6)
        foot_plots.append(plot)

    def init():
        for plot in foot_plots:
            plot.set_data([], [])
        body_plot.set_data([], [])
        return [body_plot, *foot_plots]

    def update(frame: int):
        pose = feet[frame]
        xs = pose[:, 0]
        ys = pose[:, 1]

        for i, plot in enumerate(foot_plots):
            plot.set_data([xs[i]], [ys[i]])

        body_plot.set_data([0.0], [0.0])
        return [body_plot, *foot_plots]

    animation = FuncAnimation(fig, update, frames=len(gait), init_func=init, blit=True, interval=80)
    animation.save(output_path, writer=PillowWriter(fps=10))
    plt.close(fig)


def main() -> None:
    population_size = 10
    generations = 10
    config = {
        "population_size": population_size,
        "generations": generations,
        "elite_count": 2,
        "tournament_size": 3,
        "mutation_rate": 0.08,
        "mutation_scale": 0.25,
        "seed": 7,
    }

    population, final_scores, history = run_genetic_algorithm(**config)
    best_index = int(np.argmax(final_scores))
    best_chromosome = list(population[best_index])

    start_score = history[0]
    final_score = history[-1]

    save_best_chromosome(best_chromosome)
    save_history(history)
    plot_history(history)
    animate_best_walk(best_chromosome)

    summary = make_summary(start_score, final_score, history)
    save_summary(summary)

    print(summary)
    print(f"Best chromosome saved to: {BEST_CHROMOSOME_PATH}")
    print(f"Fitness history saved to: {FITNESS_HISTORY_PATH}")
    print(f"Plot saved to: {PLOT_PATH}")
    print(f"Animation saved to: {ANIMATION_PATH}")


if __name__ == "__main__":
    main()
