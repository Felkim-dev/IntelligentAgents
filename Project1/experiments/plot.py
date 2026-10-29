"""Curvas media ± desviación entre seeds, una línea por carpeta de resultados.

Desde Project1/:
    python -m experiments.plot results/CartPole-v1/tabular/bins6 results/CartPole-v1/tabular/bins10 \
        results/CartPole-v1/tabular/bins20 --labels "6 bins" "10 bins" "20 bins" \
        --title "Q-learning tabular, CartPole-v1" --out figures/h2_tabular_cartpole --threshold 475
Deja <out>_eval.png y <out>_train.png (entrenamiento y evaluación van en figuras separadas) y, con --threshold,
<out>_summary.csv con las métricas entre seeds.
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


def summary(run_dir, label, threshold):
    """Imprime por seed y devuelve la fila agregada: final (100 ep.), pasos hasta el umbral, reloj."""
    finals, reaches, fracs, walls = [], [], [], []
    for f in sorted(Path(run_dir).glob("seed*.csv")):
        df = pd.read_csv(f)
        hit = df.loc[df.eval_mean >= threshold, "step"]
        reach = int(hit.iloc[0]) if len(hit) else None
        last = df.iloc[-1]
        finals.append(last.eval_mean)
        reaches.append(reach)
        fracs.append((df.eval_mean >= threshold).mean())   # qué tan estable se queda sobre el umbral
        walls.append(last.wall_s)
        print(f"{label} {f.stem}: final {last.eval_mean:.1f} ± {last.eval_std:.1f} | "
              f"umbral en {reach or 'no alcanzado'} | {last.wall_s:.0f} s")
    finals, reached = pd.Series(finals), [r for r in reaches if r is not None]
    row = {"config": label, "seeds": len(finals),
           "final_mean": round(finals.mean(), 1), "final_std": round(finals.std(), 1),   # std muestral, como las bandas
           "seeds_reach_threshold": len(reached),
           "median_steps_to_threshold": int(pd.Series(reached).median()) if reached else "no alcanzado",
           "frac_evals_above_threshold": round(sum(fracs) / len(fracs), 2),
           "wall_s_mean": round(sum(walls) / len(walls))}
    print(f"{label}: final {row['final_mean']} ± {row['final_std']} entre seeds | "
          f"{len(reached)}/{len(finals)} seeds llegan al umbral")
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float)
    p.add_argument("--labels", nargs="+", help="un nombre por carpeta, en el mismo orden (por defecto, la carpeta)")
    p.add_argument("--title", default="")
    a = p.parse_args()
    labels = a.labels or [Path(d).name for d in a.dirs]
    if len(labels) != len(a.dirs):
        p.error("--labels necesita un nombre por carpeta")

    for col, kind in [("eval_mean", "eval"), ("train_return", "train")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for d, name in zip(a.dirs, labels):
            g, n = load(d)
            mean, std = g[col].mean(), g[col].std()
            ax.plot(mean.index, mean, label=f"{name} (n={n})")
            ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
        if a.threshold is not None:
            ax.axhline(a.threshold, ls="--", c="gray", lw=1, label=f"umbral {a.threshold:g}")
        ax.set_title(f"{a.title}\n{'evaluación greedy (ε = 0)' if kind == 'eval' else 'entrenamiento'}".strip(), fontsize=11)
        ax.set_xlabel("pasos de entorno")
        ax.set_ylabel("retorno medio (10 ep.)" if kind == "eval" else "retorno (últimos 10 ep. de entrenamiento)")
        ax.legend()
        fig.tight_layout()
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{a.out}_{kind}.png", dpi=150)

    if a.threshold is not None:
        rows = [summary(d, name, a.threshold) for d, name in zip(a.dirs, labels)]
        pd.DataFrame(rows).to_csv(f"{a.out}_summary.csv", index=False)
