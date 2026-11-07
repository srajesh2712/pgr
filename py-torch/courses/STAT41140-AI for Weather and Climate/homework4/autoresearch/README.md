# Autoresearch: 6 h GNN forecast (James)

Karpathy-style autoresearch loop: an agent edits `train.py`, each run trains for a fixed
`TIME_BUDGET` minutes on the GPU, and changes are kept only if `val_score` improves.

| file | role |
|---|---|
| `prepare.py` | FIXED: data (24 stations, 2011–2026), splits, persistence-relative metric |
| `train.py` | the only file the agent edits; starts as teammate's ST-GCN at h=6 |
| `run.sh` | one time-boxed experiment → `results.tsv`, prints KEEP/DISCARD |
| `baselines.py` | persistence, climatology, persistence+diurnal reference rows |
| `program.md` | instructions for the agent |
| `plot_results.py` | presentation figures |

```bash
python prepare.py                                   # build cache (~30 s)
python baselines.py                                 # reference rows
STATUS_OVERRIDE=baseline ./run.sh "teammate ST-GCN h=6"
# then point an agent at program.md
python plot_results.py
```
Data: CSVs in `../data` (Git LFS). Finner (hly2075) is absent, so 24 stations.
