"""tuning_lab.py -- Spider gait GA parameter-tuning harness.

Role: "parameters tuning" workstream of the spider-gait GA project.

What it fixes compared with the shipped ``ga_tuning.py``
-------------------------------------------------------
1. REPRODUCIBILITY  ``run_genetic_algorithm(seed=...)`` only seeds crossover /
   mutation; ``create_population`` uses the *global* ``random`` module, so the
   initial population differs on every replicate and "same seed" comparisons are
   statistically meaningless.  Here the initial population is drawn from the
   trial's own RNG, so a repeated run with the same (config, seed) is identical.
2. EQUAL BUDGET     every trial uses the same number of fitness evaluations
   (pop * (gen + 1)), otherwise "better config" just means "more compute".
3. REPLICATES       each config is run ``--reps`` times with different seeds and
   reported as mean +/- sem.  A single run tells you nothing: the GA is stochastic.
4. FAST EVAL        ``evaluate_population`` tiles the chromosome into 300
   identical frames.  Because every frame is the same pose, scoring 1 frame is
   numerically identical to scoring 300 (verified: max abs diff 1e-17) but
   ~250x cheaper.  That is what makes a real search affordable.
5. METRICS          records convergence behaviour, not just the final number:
   best_overall / final_best / mean-of-history (AUC, speed proxy) / plateau_gen.

Usage (run from the project root, next to genetic_algorithm.py)
---------------------------------------------------------------
    python3 tuning_lab.py --strategy lhs --n 24 --reps 5 --pop 24 --gen 40
    python3 tuning_lab.py --strategy grid --reps 4 --pop 24 --gen 40

Outputs into --out (default ./tuning_results):
    tuning_trials.csv    one row per (config, seed) run
    tuning_summary.csv   one row per config, aggregated, sorted by best score
    fig_convergence.png  top-5 configs: mean curve +/- sem band
    fig_sensitivity.png  each knob vs. best score (one row per knob)
    fig_interaction.png  mutation_rate x tournament_size heat map (if both varied)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fitness import fitness
from joint_limits import lower, upper
from selection import select_for_next_generation
from genetic_algorithm import (
    single_point_crossover,
    mutate,
)

NUM_JOINTS = 24


# seeded version of intial_population.create_chromosome
def create_chromosome_seeded(rng: random.Random) -> List[float]:
    """Uniform draw inside the joint limits, using the *given* rng."""
    return [rng.uniform(float(lo), float(hi)) for lo, hi in zip(lower, upper)]


def evaluate_fast(population: Sequence[Sequence[float]], frames: int = 1) -> List[float]:
    """fitness() over each chromosome tiled into `frames` poses.

    frames=1 is exact for the current static-pose encoding (all frames equal)
    and ~250x faster than the project default of 300.
    """
    scores = []
    for chrom in population:
        gait = np.tile(np.asarray(chrom, dtype=float), (frames, 1))
        scores.append(float(fitness(gait)))
    return scores


@dataclass
class GAParams:
    population_size: int = 24
    generations: int = 40
    elite_count: int = 2
    tournament_size: int = 3
    mutation_rate: float = 0.05
    mutation_scale: float = 0.2
    crossover_rate: float = 1.0 
    frames: int = 1 

    def key(self) -> str:
        return (f"pop{self.population_size}_el{self.elite_count}_tk{self.tournament_size}"
                f"_mr{self.mutation_rate:.3f}_ms{self.mutation_scale:.3f}_cr{self.crossover_rate:.2f}")


@dataclass
class TrialResult:
    config_id: str
    seed: int
    population_size: int
    generations: int
    elite_count: int
    tournament_size: int
    mutation_rate: float
    mutation_scale: float
    crossover_rate: float
    frames: int
    n_evals: int
    final_best: float
    best_overall: float
    auc: float
    plateau_gen: int
    wall_time_s: float
    history: str = field(default="")


def run_ga(p: GAParams, seed: int) -> TrialResult:
    rng = random.Random(seed)
    t0 = time.time()

    population = [create_chromosome_seeded(rng) for _ in range(p.population_size)]
    history: List[float] = []
    n_evals = 0

    for _ in range(p.generations):
        fits = evaluate_fast(population, p.frames)
        n_evals += len(fits)
        history.append(max(fits))

        elites, parents = select_for_next_generation(
            population=population,
            fitnesses=fits,
            elite_count=p.elite_count,
            tournament_size=p.tournament_size,
            rng=rng,
        )
        nxt: List[List[float]] = [list(c) for c in elites]
        while len(nxt) < p.population_size:
            a, b = rng.sample(parents, 2)
            if rng.random() < p.crossover_rate:
                ca, cb = single_point_crossover(a, b, rng=rng)
            else:
                ca, cb = list(a), list(b)
            nxt.append(mutate(ca, p.mutation_rate, p.mutation_scale, rng=rng))
            if len(nxt) < p.population_size:
                nxt.append(mutate(cb, p.mutation_rate, p.mutation_scale, rng=rng))
        population = nxt[: p.population_size]

    final_fits = evaluate_fast(population, p.frames)
    n_evals += len(final_fits)
    history.append(max(final_fits))

    best = max(history)
    plateau = next((i for i, v in enumerate(history) if best - v <= 1e-4), len(history) - 1)

    return TrialResult(
        config_id=p.key(),
        seed=seed,
        population_size=p.population_size,
        generations=p.generations,
        elite_count=p.elite_count,
        tournament_size=p.tournament_size,
        mutation_rate=round(p.mutation_rate, 4),
        mutation_scale=round(p.mutation_scale, 4),
        crossover_rate=round(p.crossover_rate, 3),
        frames=p.frames,
        n_evals=n_evals,
        final_best=round(max(final_fits), 6),
        best_overall=round(best, 6),
        auc=round(float(np.mean(history[1:])), 6),
        plateau_gen=plateau,
        wall_time_s=round(time.time() - t0, 2),
        history=";".join(f"{v:.4f}" for v in history),
    )


GRID: Dict[str, Sequence] = {
    "elite_count": [0, 1, 2, 4],
    "tournament_size": [2, 3, 5, 7],
    "mutation_rate": [0.02, 0.05, 0.10, 0.20],
    "mutation_scale": [0.1, 0.2, 0.4],
    "crossover_rate": [0.6, 0.9],
}

LHS_BOX: Dict[str, tuple] = {          # (lo, hi, is_int)
    "elite_count": (0, 4, True),
    "tournament_size": (2, 7, True),
    "mutation_rate": (0.01, 0.25, False),
    "mutation_scale": (0.05, 0.5, False),
    "crossover_rate": (0.5, 1.0, False),
}


def latin_hypercube(n: int, seed: int) -> List[Dict[str, float]]:
    """Stratified random sampling of the knob box -- n points, spread out."""
    rng = np.random.default_rng(seed)
    out: List[Dict[str, float]] = []
    cols = {}
    for name, (lo, hi, is_int) in LHS_BOX.items():
        perm = rng.permutation(n)
        u = (perm + rng.random(n)) / n
        if is_int:
            vals = np.floor(u * (hi - lo + 1)).astype(int) + lo
        else:
            vals = lo + u * (hi - lo)
        cols[name] = vals
    for i in range(n):
        out.append({k: (int(v[i]) if LHS_BOX[k][2] else float(v[i])) for k, v in cols.items()})
    return out


def build_configs(args) -> List[GAParams]:
    base = dict(population_size=args.pop, generations=args.gen, frames=args.frames)
    if args.strategy == "grid":
        names = list(GRID)
        cfgs = []
        for combo in itertools.product(*(GRID[n] for n in names)):
            cfgs.append(GAParams(**base, **dict(zip(names, combo))))
        if args.n and len(cfgs) > args.n:
            rng = np.random.default_rng(args.seed_pool)
            cfgs = [cfgs[i] for i in rng.choice(len(cfgs), size=args.n, replace=False)]
        return cfgs
    return [GAParams(**base, **c) for c in latin_hypercube(args.n, args.seed_pool)]


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def aggregate(trials: List[TrialResult]) -> List[dict]:
    by: Dict[str, List[TrialResult]] = {}
    for t in trials:
        by.setdefault(t.config_id, []).append(t)
    rows = []
    for cid, ts in by.items():
        vals = np.array([t.best_overall for t in ts])
        aucs = np.array([t.auc for t in ts])
        pla = np.array([t.plateau_gen for t in ts])
        n = len(ts)
        rows.append(dict(
            config_id=cid,
            n_runs=n,
            population_size=ts[0].population_size,
            generations=ts[0].generations,
            elite_count=ts[0].elite_count,
            tournament_size=ts[0].tournament_size,
            mutation_rate=ts[0].mutation_rate,
            mutation_scale=ts[0].mutation_scale,
            crossover_rate=ts[0].crossover_rate,
            n_evals=ts[0].n_evals,
            mean_best=round(float(vals.mean()), 6),
            std_best=round(float(vals.std(ddof=1)) if n > 1 else 0.0, 6),
            sem_best=round(float(vals.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0, 6),
            min_best=round(float(vals.min()), 6),
            max_best=round(float(vals.max()), 6),
            mean_auc=round(float(aucs.mean()), 6),
            mean_plateau_gen=int(round(float(pla.mean()))),
            mean_wall_time_s=round(float(np.mean([t.wall_time_s for t in ts])), 2),
        ))
    rows.sort(key=lambda r: (-r["mean_best"], r["mean_auc"]))
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_figures(trials: List[TrialResult], rows: List[dict], out: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []

    # ---- convergence curves of top-5 configs (mean +/- sem) ---------------- #
    top = rows[:5]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(top):
        ys = []
        for t in trials:
            if t.config_id == r["config_id"]:
                ys.append([float(v) for v in t.history.split(";")])
        m = min(len(y) for y in ys)
        arr = np.array([y[:m] for y in ys])
        mean, sem = arr.mean(0), (arr.std(0, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else np.zeros(m)
        x = np.arange(m)
        ax.plot(x, mean, color=cmap(i), lw=1.8,
                label=f"{r['config_id']}  mean={r['mean_best']:.4f}+-{r['sem_best']:.4f}")
        ax.fill_between(x, mean - sem, mean + sem, color=cmap(i), alpha=0.2)
    ax.set_xlabel("generation")
    ax.set_ylabel("best fitness")
    ax.set_title("Top-5 configurations (mean of replicates, band = +/- 1 SEM)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    p = out / "fig_convergence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)

    # ---- one-dimensional sensitivity: knob vs best score ------------------- #
    knobs = ["mutation_rate", "mutation_scale", "tournament_size",
             "elite_count", "crossover_rate", "population_size"]
    knobs = [k for k in knobs if len({r[k] for r in rows}) > 1]
    if knobs:
        fig, axes = plt.subplots(1, len(knobs), figsize=(2.9 * len(knobs), 3.0), squeeze=False)
        for j, k in enumerate(knobs):
            ax = axes[0][j]
            xs = [r[k] for r in rows]
            ys = [r["mean_best"] for r in rows]
            es = [max(r["sem_best"], 1e-5) for r in rows]
            ax.errorbar(xs, ys, yerr=es, fmt="o", ms=4, alpha=0.75, lw=0.8)
            if all(isinstance(v, (int, float)) for v in xs):
                rho = np.corrcoef(xs, ys)[0, 1] if len(set(xs)) > 1 else 0.0
                ax.set_title(f"{k}\ncorr={rho:+.2f}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
        fig.suptitle("Sensitivity of best fitness to each knob", fontsize=11)
        fig.tight_layout()
        p = out / "fig_sensitivity.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p.name)

    # ---- 2-way interaction heat map --------------------------------------- #
    if {"mutation_rate", "tournament_size"} <= set(knobs):
        mr = sorted({r["mutation_rate"] for r in rows})
        tk = sorted({r["tournament_size"] for r in rows})
        mat = np.full((len(tk), len(mr)), np.nan)
        for i, t in enumerate(tk):
            for j, m in enumerate(mr):
                v = [r["mean_best"] for r in rows if r["tournament_size"] == t and r["mutation_rate"] == m]
                if v:
                    mat[i, j] = float(np.mean(v))
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(mr)), [f"{x:g}" for x in mr])
        ax.set_yticks(range(len(tk)), [f"{x:g}" for x in tk])
        ax.set_xlabel("mutation_rate")
        ax.set_ylabel("tournament_size")
        ax.set_title("Mean best fitness: mutation_rate x tournament_size", fontsize=10)
        for i in range(len(tk)):
            for j in range(len(mr)):
                if not math.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=7,
                            color="white" if mat[i, j] < np.nanmean(mat) else "black")
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        p = out / "fig_interaction.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p.name)

    return made


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="GA parameter tuning harness")
    ap.add_argument("--strategy", choices=["grid", "lhs"], default="lhs")
    ap.add_argument("--n", type=int, default=16, help="number of configs (lhs) / cap (grid)")
    ap.add_argument("--reps", type=int, default=5, help="replicates (different seeds) per config")
    ap.add_argument("--pop", type=int, default=24, help="population size, held fixed for budget")
    ap.add_argument("--gen", type=int, default=40, help="generations, held fixed for budget")
    ap.add_argument("--frames", type=int, default=1, help="1 = fast (exact here), 300 = original")
    ap.add_argument("--seed-pool", type=int, default=2026, help="RNG for the search itself")
    ap.add_argument("--out", default="tuning_results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfgs = build_configs(args)
    budget = args.pop * (args.gen + 1)
    print(f"strategy={args.strategy} configs={len(cfgs)} reps={args.reps} "
          f"trials={len(cfgs)*args.reps}  evals/trial={budget}")

    trials: List[TrialResult] = []
    t_all = time.time()
    for ci, cfg in enumerate(cfgs, 1):
        for rep in range(args.reps):
            seed = args.seed_pool + 1000 * ci + rep     # distinct seed per replicate
            r = run_ga(cfg, seed)
            trials.append(r)
        first = trials[-args.reps]
        vals = [t.best_overall for t in trials[-args.reps:]]
        eta = (time.time() - t_all) / ci * (len(cfgs) - ci)
        print(f"[{ci:>3}/{len(cfgs)}] {cfg.key():<52} best={np.mean(vals):.4f}"
              f"+-{np.std(vals, ddof=1)/math.sqrt(len(vals)) if len(vals)>1 else 0:.4f}"
              f"  eta={eta/60:.1f}min", flush=True)

    rows = aggregate(trials)
    write_csv(out / "tuning_trials.csv", [asdict(t) for t in trials])
    write_csv(out / "tuning_summary.csv", rows)
    figs = make_figures(trials, rows, out)

    print("\n=== RANKED (by mean best over replicates) ===")
    print(f"{'rank':>4} {'mean_best':>9} {'sem':>7} {'auc':>8} {'plateau':>7}  config")
    for i, r in enumerate(rows[:10], 1):
        print(f"{i:>4} {r['mean_best']:>9.4f} {r['sem_best']:>7.4f} {r['mean_auc']:>8.4f} "
              f"{r['mean_plateau_gen']:>7}  {r['config_id']}")

    # sanity: does the spread between configs exceed the noise?
    all_vals = np.array([r["mean_best"] for r in rows])
    avg_sem = float(np.mean([r["sem_best"] for r in rows]))
    print(f"\nspread across configs (std) = {all_vals.std():.4f} | average noise (sem) = {avg_sem:.4f}")
    print("=> signal/noise = %.2f  (%s)" % (
        all_vals.std() / avg_sem if avg_sem > 0 else float("inf"),
        "conclusions are defensible" if avg_sem > 0 and all_vals.std() > 2 * avg_sem
        else "increase --reps, differences are inside the noise"))
    print(f"\nwrote: tuning_trials.csv tuning_summary.csv {' '.join(figs)} -> {out}")


if __name__ == "__main__":
    main()
