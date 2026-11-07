"""
Non-learned reference baselines (run once, appends rows to results.tsv):
  - persistence          : value at t+H = value at t        (score 1.0 by definition)
  - climatology          : per-station mean for that (month, hour-of-day), from train split
  - persistence+diurnal  : value at t + climatological change from hour(t) to hour(t+H)
"""

import os
import numpy as np
from prepare import HORIZON, TARGET_IDX, TARGETS, evaluate, load_data

d = load_data()
raw = d["raw"][..., TARGET_IDX]                                  # (H, S, 4)
tr = d["split_idx"]["train"]
month = (d["hours"].astype("datetime64[M]").astype(int) % 12)
hour = d["hour_of_day"]

clim = np.zeros((12, 24) + raw.shape[1:], np.float32)            # (12, 24, S, 4)
for m in range(12):
    for h in range(24):
        sel = tr[(month[tr] == m) & (hour[tr] == h)]
        clim[m, h] = np.nanmean(raw[sel], axis=0)
clim = np.nan_to_num(clim, nan=np.nanmean(raw[tr]))

rows = []
def run(name, fn):
    print(f"\n== {name}")
    score, rmse = evaluate(fn, d, "val")
    rows.append((name, score, rmse))

run("persistence", lambda t: np.nan_to_num(raw[t]))
run("climatology (month x hour)", lambda t: clim[month[t + HORIZON], hour[t + HORIZON]])
run("persistence + climatological diurnal change",
    lambda t: np.where(np.isnan(raw[t]), clim[month[t], hour[t]], raw[t])
              + clim[month[t + HORIZON], hour[t + HORIZON]] - clim[month[t], hour[t]])

path = "results.tsv"
if not os.path.exists(path):
    with open(path, "w") as f:
        f.write("commit\tval_score\trmse_temp\trmse_rhum\trmse_msl\trmse_wdsp\tparams\tstatus\tdescription\n")
with open(path, "a") as f:
    for name, score, r in rows:
        f.write(f"-\t{score:.6f}\t" + "\t".join(f"{r[k]:.4f}" for k in TARGETS) + f"\t0\treference\t{name}\n")
