"""
Presentation plots:
  figures/progress_h6.png, progress_h1.png — val_score per experiment, running best, reference lines
  figures/test_summary.png                 — held-out TEST skill vs persistence: teammate vs best, h=1 and h=6
  figures/test_per_target.png              — TEST RMSE / persistence per target, both horizons
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"; OUT.mkdir(exist_ok=True)
TARGETS = ["temp", "rhum", "msl", "wdsp"]
C = {"persist": "#444444", "diurnal": "#9e9e9e", "teammate": "#e07b39", "best": "#2a6fb0", "discard": "#c8c8c8"}


def progress(tsv, horizon):
    if not (HERE / tsv).exists():
        return
    df = pd.read_csv(HERE / tsv, sep="\t")
    ref = df[df.status == "reference"].set_index("description")
    runs = df[df.status.isin(["baseline", "keep", "discard"])].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    col = runs.status.map({"keep": C["best"], "baseline": C["teammate"], "discard": C["discard"]})
    ax.scatter(runs.index, runs.val_score, c=col, s=30, zorder=3)
    best = runs.val_score.where(runs.status.isin(["keep", "baseline"])).cummin().ffill()
    ax.step(runs.index, best, where="post", color=C["best"], lw=2, label="best so far")
    ax.axhline(1.0, color=C["persist"], ls="--", lw=1, label="persistence")
    ax.axhline(ref.loc["persistence + climatological diurnal change", "val_score"], color=C["diurnal"],
               ls=":", lw=1.5, label="persistence + diurnal cycle (no learning)")
    for i, r in runs[runs.status == "keep"].iterrows():
        ax.annotate(r.description[:38], (i, r.val_score), fontsize=6.5, rotation=28,
                    xytext=(4, 6), textcoords="offset points", color="#1d4f80")
    ax.set_ylim(top=1.05 if horizon == 6 else 2.0)                     # keep persistence line visible
    ax.set_ylim(bottom=runs.val_score.min() * 0.92)
    ax.set_xlabel("experiment #  (5-minute GPU runs)")
    ax.set_ylabel("val_score = RMSE / persistence RMSE  (lower = better)")
    ax.set_title(f"Autoresearch progress, {horizon} h ahead, 24 Irish stations (val 2023-24)")
    ax.legend(fontsize=8, loc="upper right"); fig.tight_layout()
    fig.savefig(OUT / f"progress_h{horizon}.png", dpi=160); plt.close(fig)


def test_figures():
    f = HERE / "final_test.tsv"
    if not f.exists():
        return
    df = pd.read_csv(f, sep="\t")
    def pick(h, key):
        sub = df[(df.horizon == h) & df.description.str.contains(key, regex=False)]
        return sub.iloc[-1] if len(sub) else None
    exact = lambda h, name: df[(df.horizon == h) & (df.description == name)].iloc[-1]
    rows = {h: {"persistence": exact(h, "persistence"),
                "persistence + diurnal": pick(h, "persistence + climatological"),
                "teammate ST-GCN": pick(h, "teammate"),
                "best (5 min)": df[(df.horizon == h) & df.description.str.startswith("FINAL") &
                                   df.description.str.contains("best") & (df.budget_min == 5)].iloc[-1:].squeeze()
                                if ((df.horizon == h) & df.description.str.contains("best")).any() else None,
                "best (30 min)": pick(h, "30-min")}
            for h in (1, 6)}

    # summary bars
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, h in zip(axes, (1, 6)):
        items = [(k, v) for k, v in rows[h].items() if v is not None and not (isinstance(v, pd.Series) and v.empty)]
        names = [k for k, _ in items]; vals = [float(v.test_score) for _, v in items]
        colors = [C["persist"] if "persistence" == k else C["diurnal"] if "diurnal" in k
                  else C["teammate"] if "teammate" in k else C["best"] for k in names]
        b = ax.bar(range(len(vals)), vals, color=colors)
        ax.bar_label(b, fmt="%.3f", fontsize=8)
        ax.axhline(1, color=C["persist"], ls="--", lw=1)
        ax.set_xticks(range(len(vals)), names, rotation=25, ha="right", fontsize=8)
        ax.set_title(f"{h} h ahead"); ax.set_ylabel("TEST RMSE / persistence RMSE")
    fig.suptitle("Held-out test (2025 - Sep 2026): lower is better, < 1 beats persistence")
    fig.tight_layout(); fig.savefig(OUT / "test_summary.png", dpi=160); plt.close(fig)

    # per target
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, h in zip(axes, (1, 6)):
        p = rows[h]["persistence"]
        series = [(k, v) for k, v in rows[h].items() if k in ("teammate ST-GCN", "best (5 min)") and v is not None
                  and not (isinstance(v, pd.Series) and v.empty)]
        w = 0.8 / max(len(series), 1); x = np.arange(len(TARGETS))
        for i, (k, v) in enumerate(series):
            ax.bar(x + i * w, [float(v[f"rmse_{t}"]) / float(p[f"rmse_{t}"]) for t in TARGETS], w,
                   label=k, color=C["teammate"] if "teammate" in k else C["best"])
        ax.axhline(1, color=C["persist"], ls="--", lw=1, label="persistence")
        ax.set_xticks(x + w * (len(series) - 1) / 2, TARGETS); ax.set_title(f"{h} h ahead")
        ax.set_ylabel("TEST RMSE / persistence"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "test_per_target.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    progress("results.tsv", 6)
    progress("results_h1.tsv", 1)
    test_figures()
    print("saved:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
