"""
Presentation plots from results.tsv:
  figures/progress.png      — val_score per experiment, running best, reference lines
  figures/per_target.png    — RMSE ratio vs persistence per target: refs vs teammate vs best
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "results.tsv", sep="\t")
out = HERE / "figures"; out.mkdir(exist_ok=True)

ref = df[df.status == "reference"].set_index("description")
runs = df[df.status != "reference"].reset_index(drop=True)
persist = ref.loc["persistence"]

# 1. progress
fig, ax = plt.subplots(figsize=(9, 4.5))
ok = runs[runs.status != "crash"]
colors = ok.status.map({"keep": "#1f77b4", "baseline": "#ff7f0e", "discard": "#bbbbbb"})
ax.scatter(ok.index, ok.val_score, c=colors, zorder=3, s=28)
ax.step(ok.index, ok.val_score.where(ok.status.isin(["keep", "baseline"])).cummin().ffill(),
        where="post", color="#1f77b4", lw=1.5, label="best so far")
for name, ls in [("persistence", "--"), ("persistence + climatological diurnal change", ":")]:
    ax.axhline(ref.loc[name, "val_score"], color="k", ls=ls, lw=1, label=name)
ax.set_xlabel("experiment #"); ax.set_ylabel("val_score  (RMSE / persistence, lower is better)")
ax.set_title("Autoresearch progress — 6 h forecast, 24 Irish stations")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "progress.png", dpi=150)

# 2. per-target
targets = ["temp", "rhum", "msl", "wdsp"]
bars = {"persistence + diurnal": ref.loc["persistence + climatological diurnal change"]}
base = runs[runs.status == "baseline"]
if len(base): bars["teammate ST-GCN"] = base.iloc[0]
kept = runs[runs.status.isin(["keep", "baseline"])]
if len(kept): bars["best found"] = kept.loc[kept.val_score.idxmin()]
fig, ax = plt.subplots(figsize=(8, 4))
w = 0.8 / len(bars); x = np.arange(len(targets))
for i, (name, row) in enumerate(bars.items()):
    ax.bar(x + i * w, [row[f"rmse_{t}"] / persist[f"rmse_{t}"] for t in targets], w, label=name)
ax.axhline(1, color="k", ls="--", lw=1, label="persistence")
ax.set_xticks(x + w * (len(bars) - 1) / 2, targets); ax.set_ylabel("RMSE / persistence RMSE")
ax.set_title("Per-target skill at +6 h (validation 2023–24)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "per_target.png", dpi=150)
print(f"saved {out}/progress.png and per_target.png")
