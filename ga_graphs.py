"""Generate graphs for the spider GA project.

This script uses data files already produced by the project. It also contains
optional neural-network plotting functions for the teammate to use once
training history and results are saved as CSV files.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
GA_HISTORY_PATH = ROOT / "fitness_history.csv"
GA_PLOT_PATH = ROOT / "ga_fitness_over_generations.png"
GA_TUNING_PLOT_PATH = ROOT / "ga_parameter_comparison.png"
NN_LOSS_PLOT_PATH = ROOT / "nn_loss_over_epochs.png"
NN_ACCURACY_PLOT_PATH = ROOT / "nn_accuracy_over_epochs.png"


def read_csv_columns(path: Path) -> dict[str, list[float]]:
    """Read a CSV whose first row contains numeric column names."""
    if not path.exists():
        return {}

    columns: dict[str, list[float]] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return {}
        for name in reader.fieldnames:
            columns[name] = []
        for row in reader:
            for name in reader.fieldnames:
                try:
                    columns[name].append(float(row[name]))
                except (TypeError, ValueError):
                    continue
    return columns


def plot_ga_history(path: Path = GA_HISTORY_PATH) -> Path | None:
    """Plot fitness over GA generations."""
    data = read_csv_columns(path)
    if not data:
        return None

    generations = data.get("generation")
    fitness = data.get("fitness")
    if not generations or not fitness:
        return None

    plt.figure(figsize=(8, 5))
    plt.plot(generations, fitness, marker="o", linewidth=2, label="Best fitness")
    plt.title("Genetic Algorithm Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GA_PLOT_PATH, dpi=200)
    plt.close()
    return GA_PLOT_PATH


def plot_ga_tuning(
    labels: list[str],
    scores: list[float],
    path: Path = GA_TUNING_PLOT_PATH,
) -> Path:
    """Plot final fitness for each GA parameter configuration."""
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, scores, color="#4C78A8")
    plt.title("GA Parameter Comparison")
    plt.xlabel("Configuration")
    plt.ylabel("Best Final Fitness")
    plt.grid(axis="y", alpha=0.3)

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_nn_training(path: Path) -> tuple[Path | None, Path | None]:
    """Plot NN loss and accuracy from a CSV containing epoch values.

    The CSV may contain: epoch, loss, val_loss, accuracy, val_accuracy.
    """
    data = read_csv_columns(path)
    if not data:
        return None, None

    epochs = data.get("epoch")
    if not epochs:
        epochs = list(range(1, len(next(iter(data.values()))) + 1))

    loss_plot: Path | None = None
    if "loss" in data:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, data["loss"], marker="o", label="Training loss")
        if "val_loss" in data:
            plt.plot(epochs, data["val_loss"], marker="o", label="Validation loss")
        plt.title("Neural Network Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(NN_LOSS_PLOT_PATH, dpi=200)
        plt.close()
        loss_plot = NN_LOSS_PLOT_PATH

    accuracy_plot: Path | None = None
    if "accuracy" in data:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, data["accuracy"], marker="o", label="Training accuracy")
        if "val_accuracy" in data:
            plt.plot(epochs, data["val_accuracy"], marker="o", label="Validation accuracy")
        plt.title("Neural Network Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(NN_ACCURACY_PLOT_PATH, dpi=200)
        plt.close()
        accuracy_plot = NN_ACCURACY_PLOT_PATH

    return loss_plot, accuracy_plot


if __name__ == "__main__":
    ga_plot = plot_ga_history()
    if ga_plot:
        print(f"GA graph saved to: {ga_plot}")
    else:
        print(f"No GA history found at: {GA_HISTORY_PATH}")

    nn_candidates = [ROOT / "nn_history.csv", ROOT / "training_history.csv"]
    nn_plotted = False
    for candidate in nn_candidates:
        loss_plot, accuracy_plot = plot_nn_training(candidate)
        if loss_plot:
            print(f"NN loss graph saved to: {loss_plot}")
            nn_plotted = True
        if accuracy_plot:
            print(f"NN accuracy graph saved to: {accuracy_plot}")
            nn_plotted = True
        if nn_plotted:
            break

    if not nn_plotted:
        print("No neural-network history CSV found yet.")
