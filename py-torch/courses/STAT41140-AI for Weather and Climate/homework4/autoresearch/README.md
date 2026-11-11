# Autoresearch: GNN weather forecasting for 24 Irish stations (James)

> **📊 Results: [RESULTS.md](RESULTS.md)** (one-page summary) · **🧪 Every experiment: [EXPERIMENTS.md](EXPERIMENTS.md)** (full log)

On held-out test data (2025 to Sep 2026), score = RMSE / persistence RMSE, lower is better:

| model | 1 h ahead | 6 h ahead |
|---|---|---|
| persistence | 1.000 | 1.000 |
| teammate ST-GCN | 1.956 | 0.757 |
| **best found by autoresearch** | **0.644** | **0.435** |

## How it works

This is a Karpathy-style autoresearch loop. An agent edits `train.py`, and each run trains for a fixed 5 minutes on the GPU. A change is kept only if `val_score` beats the best by more than seed noise.

| file | role |
|---|---|
| `prepare.py` | FIXED: data (24 stations, 2011–2026), splits, persistence-relative metric |
| `train.py` | the only file the agent edits (best 6 h model on this branch; best 1 h on `dev_james_h1`) |
| `run.sh` | one time-boxed experiment → `results.tsv`, prints KEEP/DISCARD (`--horizon`, `--budget`, `--final-test`) |
| `baselines.py` | persistence, climatology, persistence+diurnal reference rows |
| `program.md` | instructions for the agent |
| `plot_results.py` | presentation figures → `figures/` |

```bash
python prepare.py                          # build cache (~30 s)
python baselines.py                        # reference rows
./run.sh "description of the idea"         # one 5-minute experiment
./run.sh --final-test "FINAL 6h best"      # also score the test split
python plot_results.py
```
The data is the CSVs in `../data` (Git LFS). Finner (hly2075) is missing, so there are 24 stations.
