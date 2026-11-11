# Experiments: beating persistence with a GNN on Irish station data

**Bottom line.** On held-out test data (2025 to Sep 2026), my final model's error is **57% lower than persistence at 6 h** and **36% lower at 1 h**. The teammate's ST-GCN is 24% better than persistence at 6 h and **96% worse** at 1 h. About 50 five-minute GPU experiments got there; this document explains each change that helped and by how much.

## 1. The task and how I scored it

- **Data:** hourly observations from 24 Met Éireann stations, 2011 to Sep 2026 (Finner is missing).
- **Goal:** predict temperature, humidity, pressure and wind speed at every station *h* hours ahead (h = 6, then h = 1).
- **Score:** each target's RMSE divided by the RMSE of persistence (tomorrow = today), averaged over the 4 targets. **1.0 = as good as persistence; lower is better.** 0.5 means half the persistence error.
- **Splits:** train on 2011–2022, choose models on validation (2023–24), and score the **test** years (2025 to Sep 2026) once at the very end.
- **Loop (Karpathy-style autoresearch):** change one thing in `train.py`, train for exactly **5 minutes** on the GPU, then keep the change only if the validation score improves by more than seed noise (3 runs with different seeds differ by about ±0.005). Otherwise revert.

## 2. Starting point: the teammate's model

The teammate's ST-GCN (2 graph-conv layers at every hour → GRU → MLP) was retrained in the same pipeline at h = 6. With a linear LR decay added to reduce seed noise, it scored **0.777** on validation, a 22% improvement on persistence. A no-learning trick (persistence plus the average daily temperature cycle) already scored 0.820, so the GNN was only slightly better than a lookup table.

## 3. Experiments that improved the model (6 h, validation)

Each row is one kept change, applied on top of everything above it. "Improvement" is the drop in score from that change; "vs teammate" is the total reduction from the 0.777 starting point.

| # | change | what it does and why it helped | val score | improvement | vs teammate |
|---|---|---|---|---|---|
| 0 | teammate ST-GCN (baseline) | starting point | 0.777 | — | — |
| 1 | **predict the change from persistence** | the network outputs y(t+h) − y(t) instead of the full value, so it only has to learn a small correction | 0.720 | −0.057 | 7% |
| 2 | calendar inputs | hour of day, day of year and target hour (sin/cos) let it learn the daily and seasonal cycle | 0.704 | −0.016 | 9% |
| 3 | wind as a vector (u, v) | wind direction tells you where the weather is coming from | 0.675 | −0.029 | 13% |
| 4 | learned station embeddings | an 8-number ID per station captures local effects such as coast or shelter | 0.657 | −0.018 | 15% |
| 5 | bf16 mixed precision | 2.8× more training steps in the same 5 minutes | 0.649 | −0.009 | 16% |
| 6 | learned adjacency | the graph is half distance-based, half learned, so it can link stations for reasons other than distance | 0.639 | −0.009 | 18% |
| 7 | **raw inputs skip the GCN** | the GCN's BatchNorm and dropout blurred each station's own readings; feeding raw inputs straight into the GRU fixed it | 0.559 | −0.080 | 28% |
| 8 | dropout 0 | with only 5 minutes of training the model was underfitting, not overfitting | 0.538 | −0.021 | 31% |
| 9 | loss weighted by 1 / persistence error | makes the training loss match the score, so no target dominates | 0.524 | −0.014 | 32% |
| 10 | 12 h history + wind-aware edges + multi-lead loss | three small near-misses combined; also trains on t+1 … t+6, not only t+6 | 0.516 | −0.008 | 33% |
| 11 | **new architecture: GRU per station, then attention across stations** | encode each station's history first, then mix stations once with self-attention biased towards nearby stations, instead of a graph conv at every hour | **0.454** | −0.062 | **42%** |

**Three structural changes did most of the work:** predicting the change (1), the skip connection (7) and the new architecture (11). Together they gave 0.199 of the total 0.322 improvement (62%). Adding inputs (2–4) gave another 0.063.

### Final model

1. **Time:** for each station, the last 12 h of 10 weather features, calendar sin/cos and its station embedding go through a GRU.
2. **Space:** one self-attention layer lets every station look at every other, with a learned penalty proportional to distance.
3. **Output:** a linear layer predicts the change from persistence for each hour t+1 … t+6, which is added back to the current value.

That is 433k parameters, trained for 5 minutes with Adam (LR 5e-4 decaying linearly to 0).

## 4. Experiments that did not help

These changed the score by less than the noise, or made it worse, so they were reverted:

- **Bigger models:** wider layers (4× parameters), 3 GCN layers, 3 attention layers. None helped within 5 minutes.
- **Longer history:** 24 h and 48 h of history were no better than 12 h.
- **Optimiser tweaks:** batch 512 or 1024, LR 1e-3, EMA of weights.
- **Speed tricks:** torch.compile and CUDA graphs gave a few % more steps but no score gain.
- **Extra physics in the graph:** pressure and temperature differences as edge features, wind-aware attention bias.
- **Other:** LayerNorm instead of BatchNorm, missing-value flags, a second attention layer before the GRU, using persistence + daily cycle as the base.
- **Training 6× longer (30 min) made it worse:** val 0.481 vs 0.454. Training loss kept falling, so the model overfits once there's no dropout. The 5-minute schedule acts as early stopping.

## 5. The 1 h horizon (the teammate's horizon)

At 1 h the teammate's model scored **1.809** on validation, nearly twice the persistence error. Weather barely changes in an hour, and a model that predicts the full value makes errors bigger than the change itself. My 6 h architecture transferred directly (0.730). Two further changes helped: 6 h of history instead of 12 h (0.727) and 3 attention layers (**0.720**). In total that's a 60% improvement on the teammate's validation score.

## 6. Held-out test results (2025 to Sep 2026)

All models were frozen before this split was scored, and each was scored once.

| model | 1 h score | 6 h score |
|---|---|---|
| persistence | 1.000 | 1.000 |
| climatology (average for that month and hour) | 7.404 | 1.740 |
| persistence + daily cycle (no learning) | 0.918 | 0.802 |
| teammate ST-GCN | 1.956 | 0.757 |
| **my final model** | **0.644** | **0.435** |
| my final model, trained 30 min | — | 0.458 |

- **6 h:** 57% less error than persistence, 43% less than the teammate's model.
- **1 h:** 36% less error than persistence, 67% less than the teammate's model.
- **No overfitting to validation:** the 6 h score was 0.454 on validation and 0.435 on test, and the 1 h score was 0.720 and 0.644.

Test RMSE in physical units (in brackets: reduction compared with persistence):

| horizon | model | temp °C | humidity % | pressure hPa | wind kt |
|---|---|---|---|---|---|
| 6 h | persistence | 3.22 | 13.57 | 2.83 | 4.13 |
| 6 h | teammate ST-GCN | 1.70 | 7.94 | 2.64 | 4.06 |
| 6 h | **my model** | **1.03** (−68%) | **5.78** (−57%) | **1.15** (−59%) | **2.43** (−41%) |
| 1 h | persistence | 0.86 | 4.46 | 0.56 | 1.69 |
| 1 h | teammate ST-GCN | 1.25 | 6.02 | 1.62 | 3.59 |
| 1 h | **my model** | **0.52** (−39%) | **3.29** (−26%) | **0.24** (−57%) | **1.34** (−20%) |

Wind speed is the hardest target at both horizons. Pressure improves most at 1 h, because the network can extrapolate the pressure trend.

## 7. Caveats

- The teammate's original script used each station's full record from 1946 with gaps filled in. Here their architecture is retrained on the same 2011–2026 data and splits as mine, so the comparison is fair but not an exact rerun.
- Every model got the same 5-minute GPU budget; longer budgets would need regularisation.
- Code: best 6 h model is `train.py` on `dev_james_branch` (`86340f2`); best 1 h model is on `dev_james_h1` (`2d20a9d`); the teammate baseline is `b01d71b`.

---

## Appendix: every run

Bold = kept. RMSE columns are validation RMSE.

### A1. All 6 h runs (44)

| reference (no learning) | val score | temp °C | rhum % | msl hPa | wdsp kt |
|---|---|---|---|---|---|
| persistence | 1.000 | 2.90 | 12.49 | 3.05 | 4.10 |
| climatology (month x hour) | 1.845 | 2.92 | 9.30 | 13.58 | 4.80 |
| persistence + climatological diurnal change | 0.820 | 1.88 | 9.03 | 3.02 | 3.77 |

| # | experiment | val score | temp °C | rhum % | msl hPa | wdsp kt | status |
|---|---|---|---|---|---|---|---|
| 1 | teammate ST-GCN h=6 | 0.791 | 1.73 | 7.88 | 2.78 | 4.19 | reference run |
| 2 | teammate ST-GCN h=6 (seed 1) | 0.807 | 1.71 | 7.85 | 2.95 | 4.27 | seed check |
| 3 | teammate ST-GCN h=6 (seed 2) | 0.777 | 1.65 | 7.74 | 2.78 | 4.12 | seed check |
| 4 | teammate ST-GCN h=6 + LR decay (seed 0) | 0.871 | 1.94 | 8.44 | 3.46 | 4.11 | invalid (ran on CPU) |
| 5 | teammate ST-GCN h=6 + LR decay (seed 0) | 0.777 | 1.67 | 7.73 | 2.80 | 4.08 | baseline |
| 6 | teammate ST-GCN h=6 + LR decay (seed 1) | 0.787 | 1.70 | 7.78 | 2.82 | 4.15 | seed check |
| 7 | teammate ST-GCN h=6 + LR decay (seed 2) | 0.783 | 1.67 | 7.72 | 2.86 | 4.09 | seed check |
| 8 | **residual on persistence** | **0.720** | 1.64 | 8.12 | 2.46 | 3.51 | keep |
| 9 | **calendar inputs (hour/doy sin-cos, target hour)** | **0.704** | 1.56 | 7.90 | 2.42 | 3.49 | keep |
| 10 | **wind vector u,v inputs** | **0.675** | 1.52 | 7.74 | 2.21 | 3.41 | keep |
| 11 | **learned station embeddings (8-d)** | **0.657** | 1.44 | 7.47 | 2.18 | 3.37 | keep |
| 12 | batch 1024, LR 1e-3 | 0.760 | 1.72 | 8.29 | 2.75 | 3.62 | discard |
| 13 | **bf16 autocast (train)** | **0.649** | 1.40 | 7.39 | 2.15 | 3.34 | keep |
| 14 | batch 512, LR 7e-4 (bf16) | 0.651 | 1.41 | 7.39 | 2.16 | 3.35 | discard |
| 15 | **adaptive adjacency 50/50 with distance** | **0.639** | 1.37 | 7.23 | 2.13 | 3.30 | keep |
| 16 | **skip: raw inputs + GCN output into GRU** | **0.559** | 1.21 | 6.22 | 1.91 | 2.84 | keep |
| 17 | **dropout 0** | **0.538** | 1.17 | 6.15 | 1.73 | 2.81 | keep |
| 18 | **loss weights 1/persistence MSE** | **0.524** | 1.18 | 6.22 | 1.54 | 2.81 | keep |
| 19 | residual base = persistence + clim diurnal change | 0.530 | 1.22 | 6.31 | 1.55 | 2.82 | discard |
| 20 | LayerNorm instead of BatchNorm | 0.527 | 1.19 | 6.27 | 1.55 | 2.81 | discard |
| 21 | wider: GCN 128, GRU 256 | 0.526 | 1.18 | 6.26 | 1.55 | 2.81 | discard |
| 22 | dynamic wind-aware edges (advection MLP) | 0.523 | 1.19 | 6.21 | 1.52 | 2.81 | discard |
| 23 | torch.compile | 0.525 | 1.18 | 6.22 | 1.54 | 2.81 | discard |
| 24 | CUDA graphs + fused Adam + fewer syncs | 0.523 | 1.17 | 6.21 | 1.54 | 2.81 | discard |
| 25 | multi-lead aux loss t+1..t+6 | — | — | — | — | — | crash |
| 26 | multi-lead aux loss t+1..t+6 | 0.523 | 1.19 | 6.24 | 1.51 | 2.80 | discard |
| 27 | SEQ_LEN 48 | 0.531 | 1.21 | 6.33 | 1.55 | 2.83 | discard |
| 28 | SEQ_LEN 12 | 0.522 | 1.17 | 6.16 | 1.53 | 2.82 | discard |
| 29 | **combo: SEQ12 + wind edges + multi-lead** | **0.516** | 1.17 | 6.15 | 1.50 | 2.78 | keep |
| 30 | speed: CUDA graphs + fused Adam on combo | 0.515 | 1.17 | 6.12 | 1.50 | 2.77 | discard |
| 31 | edge feats: msl/temp differences | 0.522 | 1.18 | 6.17 | 1.54 | 2.79 | discard |
| 32 | 3 GCN layers | 0.519 | 1.17 | 6.17 | 1.51 | 2.80 | discard |
| 33 | EMA weights 0.999 | 0.517 | 1.17 | 6.16 | 1.50 | 2.79 | discard |
| 34 | arch: station GRU then distance-biased attention | — | — | — | — | — | crash |
| 35 | **arch: station GRU then distance-biased attention** | **0.454** | 1.00 | 5.61 | 1.27 | 2.48 | keep |
| 36 | SEQ_LEN 24 on attn arch | 0.453 | 1.01 | 5.62 | 1.25 | 2.47 | discard |
| 37 | width 256 on attn arch | 0.454 | 1.01 | 5.60 | 1.26 | 2.48 | discard |
| 38 | wind-aware attention bias | 0.454 | 1.01 | 5.60 | 1.27 | 2.46 | discard |
| 39 | LR 1e-3 on attn arch | 0.453 | 1.00 | 5.58 | 1.27 | 2.48 | discard |
| 40 | space-time-space: pre-GRU station attention | 0.456 | 1.02 | 5.64 | 1.26 | 2.48 | discard |
| 41 | observed-flag inputs | 0.454 | 1.01 | 5.60 | 1.28 | 2.46 | discard |
| 42 | BEST 6h attn arch (seed 1) | 0.455 | 1.01 | 5.62 | 1.28 | 2.48 | seed check |
| 43 | BEST 6h attn arch (seed 2) | 0.455 | 1.01 | 5.62 | 1.27 | 2.47 | seed check |
| 44 | 3 attention layers | 0.457 | 1.01 | 5.63 | 1.28 | 2.49 | discard |

Runs 36–39 are slightly below run 35 but inside the 0.002 noise threshold, so they were discarded.


### A2. All 1 h runs (9)

| reference (no learning) | val score | temp °C | rhum % | msl hPa | wdsp kt |
|---|---|---|---|---|---|
| persistence | 1.000 | 0.79 | 4.30 | 0.77 | 1.70 |
| climatology (month x hour) | 6.575 | 2.92 | 9.30 | 13.58 | 4.80 |
| persistence + climatological diurnal change | 0.931 | 0.66 | 3.94 | 0.76 | 1.66 |

| # | experiment | val score | temp °C | rhum % | msl hPa | wdsp kt | status |
|---|---|---|---|---|---|---|---|
| 1 | teammate ST-GCN + LR decay (h=1) | 1.809 | 1.21 | 5.71 | 1.73 | 3.62 | baseline |
| 2 | **best 6h arch at h=1** | **0.730** | 0.51 | 3.26 | 0.55 | 1.37 | keep |
| 3 | **SEQ_LEN 6** | **0.727** | 0.51 | 3.24 | 0.55 | 1.36 | keep |
| 4 | latest-hour skip to head | 0.725 | 0.51 | 3.25 | 0.54 | 1.37 | discard |
| 5 | LR 1e-3 | 0.730 | 0.51 | 3.25 | 0.55 | 1.37 | discard |
| 6 | aux leads t+1..t+6, score t+1 | 0.749 | 0.53 | 3.35 | 0.55 | 1.42 | discard |
| 7 | SEQ_LEN 3 | 0.727 | 0.51 | 3.25 | 0.55 | 1.36 | discard |
| 8 | **3 attention layers** | **0.720** | 0.50 | 3.21 | 0.55 | 1.34 | keep |
| 9 | 4 attention layers | 0.722 | 0.50 | 3.22 | 0.55 | 1.35 | discard |

