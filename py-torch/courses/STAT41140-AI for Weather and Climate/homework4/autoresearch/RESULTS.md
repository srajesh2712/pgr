# Results: autoresearch GNN for Irish station forecasting

**Task.** 24 Met Éireann stations, hourly, 2011–2026. Predict temp, rhum, msl and wdsp at every station *h* hours ahead.
**Splits.** Train 2011–2022, validation 2023–24 (used to choose models), test 2025 to Sep 2026 (scored once at the end).
**Metric.** `score` = mean over the 4 targets of RMSE_model / RMSE_persistence. Below 1 beats persistence; lower is better.
**Budget.** Every experiment trained for **5 minutes** on an RTX 5060 Laptop GPU (8 GB).

## Headline (held-out TEST, 2025 – Sep 2026)

| model | h = 1 | h = 6 |
|---|---|---|
| persistence (y(t+h) = y(t)) | 1.000 | 1.000 |
| persistence + climatological diurnal change (no learning) | 0.918 | 0.802 |
| teammate ST-GCN (`train-gnn.py` architecture, retrained here) | **1.956** | 0.757 |
| **best found by autoresearch** | **0.644** | **0.435** |
| best, trained 30 min instead of 5 | not run | 0.458 (worse: overfits) |

Test RMSE in physical units:

| model | temp °C | rhum % | msl hPa | wdsp kt |
|---|---|---|---|---|
| h6 persistence | 3.22 | 13.57 | 2.83 | 4.13 |
| h6 teammate | 1.70 | 7.94 | 2.64 | 4.06 |
| **h6 best** | **1.03** | **5.78** | **1.15** | **2.43** |
| h1 persistence | 0.86 | 4.46 | 0.56 | 1.69 |
| h1 teammate | 1.25 | 6.02 | 1.62 | 3.59 |
| **h1 best** | **0.52** | **3.29** | **0.24** | **1.34** |

**Points for the presentation**
- At **1 h**, the teammate's model is about **2× worse than persistence**. The weather changes very little in an hour, and that model predicts the full value from scratch. The original script never compared against persistence, so this never showed up.
- At **6 h**, the no-learning "persistence + diurnal cycle" trick already gets 0.80. The teammate's GNN beats it by only 6%. The best model halves the persistence error.
- Validation and test agree: 6 h best is 0.454 on val and 0.435 on test, so the model selection did not overfit validation.
- **More training did not help.** At 6 h, a 30-minute run of the same model scored 0.481 on val and 0.458 on test, worse than the 5-minute run. Training loss kept falling the whole time, so the model overfits with dropout 0. The 5-minute LR schedule acts as early stopping.
- Seed noise: the teammate architecture varied by ±0.005 between seeds. The final architecture varies by ±0.0005 (6 h val seeds 0.4544 / 0.4553 / 0.4548).

## What the autoresearch loop found (6 h, validation score)

See `figures/progress_h6.png`. Kept changes, in order:

| # | change | val | why it helped |
|---|---|---|---|
| 0 | teammate ST-GCN + linear LR decay | 0.777 | baseline (the decay halved seed noise) |
| 1 | **predict residual on persistence** | 0.720 | the model only learns the correction |
| 2 | calendar inputs (hour, day-of-year, target hour) | 0.704 | daily and seasonal cycle |
| 3 | wind vector (u, v) inputs | 0.675 | direction of advection |
| 4 | learned station embeddings | 0.657 | local site effects (coast, shelter) |
| 6 | bf16 autocast | 0.649 | 2.8× more steps in 5 min |
| 8 | adaptive (learned) adjacency mixed with distance | 0.639 | links that are not only about distance |
| 9 | **raw inputs skip past the GCN into the GRU** | 0.559 | BatchNorm + dropout in the GCN were blurring each station's own signal |
| 10 | dropout 0 | 0.538 | 5 min of training is underfitting, not overfitting |
| 11 | loss weights = 1/persistence MSE | 0.524 | the loss now matches the metric |
| 21 | combination: 12 h history + wind-aware edges + multi-lead loss | 0.516 | small near-misses add up |
| 26 | **temporal-then-spatial: GRU per station → distance-biased self-attention** | **0.454** | one spatial mixing step after the time encoding, instead of a GCN at every hour |

Did not help (within noise): wider or deeper models, LayerNorm, longer history (24 or 48 h), EMA weights, torch.compile / CUDA graphs, pressure-difference edges, wind-aware attention bias, missing-value flags.
For h = 1, the 6 h architecture transferred directly (0.730). Adding a **third attention layer** and **6 h of history** brought it to 0.720.

Full logs: `results.tsv` (6 h), `results_h1.tsv`, `final_test.tsv`, `notes.md`, `logs/`.

## Reproduce

| model | branch / commit |
|---|---|
| teammate baseline (both horizons) | `b01d71b` (`train.py`) |
| best 6 h | `dev_james_branch` @ `86340f2` (`train.py`) |
| best 1 h | `dev_james_h1` @ `2d20a9d` (`train.py`) |

```bash
python prepare.py                    # build cache
./run.sh --final-test "..."          # 6 h, scores test
./run.sh --horizon 1 --final-test "..."
python plot_results.py               # figures/
```

## Caveats
- **Data window differs from the teammate's.** Their script used each station's whole record (from 1946 onward) and forward/back-filled gaps. Here the teammate's architecture is retrained on the same 2011–2026 data and splits as everything else, so the comparison is like for like. It is not a rerun of their exact pipeline.
- Finner (hly2075) is missing from the data, so there are 24 stations, not 25.
- The best models were selected on validation over about 40 experiments, so a small optimism bias remains. The test numbers above are the unbiased ones.
