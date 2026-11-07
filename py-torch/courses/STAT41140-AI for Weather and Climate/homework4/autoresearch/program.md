# Autoresearch: 6-hour GNN forecast for Irish weather stations

You are an autonomous research agent. Your goal: **minimise `val_score`** of `train.py`.

## The task (fixed — defined in `prepare.py`)
- 24 Met Éireann stations, hourly, 2011–2026. Train 2011–2022, val 2023–2024, test 2025–26 (never look at test).
- Predict temp, rhum, msl, wdsp at **t + 6 h** using only data at hours ≤ t.
- `val_score` = mean over the 4 targets of RMSE_model / RMSE_persistence. Lower is better; < 1 beats persistence.

## Reference points (in `results.tsv`)
| model | val_score |
|---|---|
| persistence | 1.000 |
| persistence + climatological diurnal change (no learning) | 0.820 |
| teammate ST-GCN (first `train.py`) | see first `baseline` row |

A GNN that does not beat **0.820** has not learned anything beyond the daily cycle.

## Rules
1. Edit **only `train.py`**. Never edit `prepare.py`, `run.sh`, `baselines.py`, or this file.
2. Keep the contract: wall-clock training ≤ `TIME_BUDGET`, call `evaluate(predict, d, "val")`, print the summary block. No test-set evaluation.
3. No leakage: inputs for forecast time t may use only hours ≤ t (calendar features of t+6 are fine — they are known in advance).
4. GPU has 8 GB (≈6 GB free). Keep `peak_vram_mb` < 5500.
5. One idea per experiment, so results are attributable.

## Loop (repeat until told to stop or N experiments done)
1. Read `results.tsv` and `git log --oneline -15` to see what has been tried.
2. Pick ONE idea; edit `train.py`.
3. `git commit train.py -m "<idea>"` (commit ONLY train.py)
4. `./run.sh "<idea>"` (run from this folder, exactly this form — it is the only form allowed to use the GPU)
5. If it prints **KEEP**: leave the commit. If **DISCARD** or **CRASH**: `git reset --hard HEAD~1`.
   (`results.tsv` and `logs/` are git-ignored, so they survive the reset.)
6. If CRASH from a simple bug (typo, shape), fix and retry once; otherwise move on.
7. Write one line in `notes.md` (git-ignored) on what you learned.

## Idea list (not exhaustive, roughly by expected value)
- **Predict the residual**: output Δ = y(t+6) − y(t) (+ climatological diurnal change) instead of y directly.
- Add calendar inputs: sin/cos hour-of-day and day-of-year for both t and t+6.
- Use the wind vector channels (`wind_u`, `wind_v`, indices 8–9) as inputs.
- Static station features: lat/lon, learned station embeddings.
- Adjacency: learned (Graph WaveNet adaptive adjacency), different `sigma`, k-NN, **wind-direction-aware edges** (upwind stations matter more — weather moves W→E over Ireland).
- Message passing: GAT attention, edge features (distance, bearing), more/fewer GCN layers, residual connections, LayerNorm instead of BatchNorm.
- Temporal model: TCN / 1-D conv, Transformer over time, shorter/longer `SEQ_LEN`.
- Optimisation: AdamW, LR schedule (warmup + cosine to the time budget), larger batch, bf16 autocast, `torch.compile`, per-target loss weighting.
- Loss in physical-ratio space (weight each target by 1/persistence RMSE) to directly match the metric.

## Simplicity criterion
If two versions score within 0.002, keep the simpler one.
