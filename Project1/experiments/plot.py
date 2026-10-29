"""Curvas media ± desviación entre seeds, una línea por carpeta de resultados.

Desde Project1/:
    python -m experiments.plot results/CartPole-v1/tabular/bins6 results/CartPole-v1/tabular/bins10 \
        results/CartPole-v1/tabular/bins20 --out figures/h2_tabular_cartpole --threshold 475
Deja <out>_eval.png y <out>_train.png (entrenamiento y evaluación van en figuras separadas).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load(run_dir):
    runs = [pd.read_csv(f) for f in sorted(Path(run_dir).glob("seed*.csv"))]
    return pd.concat(runs).groupby("step"), len(runs)


def summary(run_dir, threshold):
    """Imprime retorno final (100 ep.), pasos hasta el umbral y tiempo de reloj por seed."""
    for f in sorted(Path(run_dir).glob("seed*.csv")):
        df = pd.read_csv(f)
        hit = df.loc[df.eval_mean >= threshold, "step"]
        reach = int(hit.iloc[0]) if len(hit) else "no alcanzado"
        last = df.iloc[-1]
        print(f"{run_dir} {f.stem}: final {last.eval_mean:.1f} ± {last.eval_std:.1f} | "
              f"umbral en {reach} | {last.wall_s:.0f} s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float)
    a = p.parse_args()

    for col, label in [("eval_mean", "eval"), ("train_return", "train")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for d in a.dirs:
            g, n = load(d)
            mean, std = g[col].mean(), g[col].std()
            ax.plot(mean.index, mean, label=f"{Path(d).name} (n={n})")
            ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
        if a.threshold is not None:
            ax.axhline(a.threshold, ls="--", c="gray", lw=1)
        ax.set_xlabel("pasos de entorno")
        ax.set_ylabel("retorno de evaluación (greedy)" if label == "eval" else "retorno de entrenamiento (últimos 10 ep.)")
        ax.legend()
        fig.tight_layout()
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{a.out}_{label}.png", dpi=150)

    if a.threshold is not None:
        for d in a.dirs:
            summary(d, a.threshold)
