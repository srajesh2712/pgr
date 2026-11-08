# Experiment log: autoresearch GNN, 24 Irish stations

Every experiment run for this project, at 6 h and 1 h ahead. For the short version, see [RESULTS.md](RESULTS.md).

- **Score** = mean over temp, rhum, msl and wdsp of RMSE_model / RMSE_persistence. Below 1 beats persistence; lower is better.
- **Val** = 2023–24 (used to pick models). **Test** = 2025 to Sep 2026 (scored once, at the end).
- Every run trained for **5 minutes** on an RTX 5060 Laptop GPU. A change was **kept** only if it beat the best val score by more than seed noise: 0.005 for the teammate architecture, 0.002 for the attention architecture.
- RMSE columns are in physical units: test RMSE in section 1, validation RMSE in the logs. **Bold** = kept change.

## 1. Final test results

| horizon | model | train time | test score | temp °C | rhum % | msl hPa | wdsp kt |
|---|---|---|---|---|---|---|---|
| 1 h | persistence | — | 1.000 | 0.86 | 4.46 | 0.56 | 1.69 |
| 1 h | climatology (month x hour) | — | 7.404 | 2.94 | 9.83 | 11.89 | 4.66 |
| 1 h | persistence + diurnal change | — | 0.918 | 0.70 | 4.02 | 0.55 | 1.65 |
| 1 h | teammate ST-GCN | 5 min | 1.956 | 1.25 | 6.02 | 1.62 | 3.59 |
| 1 h | **best** | 5 min | **0.644** | 0.52 | 3.29 | 0.24 | 1.34 |
| 6 h | persistence | — | 1.000 | 3.22 | 13.57 | 2.83 | 4.13 |
| 6 h | climatology (month x hour) | — | 1.740 | 2.94 | 9.83 | 11.89 | 4.66 |
| 6 h | persistence + diurnal change | — | 0.802 | 2.00 | 9.38 | 2.80 | 3.75 |
| 6 h | teammate ST-GCN | 5 min | 0.757 | 1.70 | 7.94 | 2.64 | 4.06 |
| 6 h | **best** | 5 min | **0.435** | 1.03 | 5.78 | 1.15 | 2.43 |
| 6 h | best, 30-min training | 30 min | 0.458 | 1.05 | 5.97 | 1.27 | 2.56 |

The 30-minute run scored worse than the 5-minute run (val 0.481 vs 0.454). Training loss kept falling, so it overfits with dropout 0. The 5-minute linear LR decay acts as early stopping.

## 2. 6 h log (44 runs)

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

## 3. 1 h log (9 runs)

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

## 4. Where the 6 h gain came from

| kept change | val after | gain |
|---|---|---|
| residual on persistence | 0.720 | −0.057 |
| calendar inputs (hour/doy sin-cos, target hour) | 0.704 | −0.016 |
| wind vector u,v inputs | 0.675 | −0.029 |
| learned station embeddings (8-d) | 0.657 | −0.018 |
| bf16 autocast (train) | 0.649 | −0.009 |
| adaptive adjacency 50/50 with distance | 0.639 | −0.009 |
| skip: raw inputs + GCN output into GRU | 0.559 | −0.080 |
| dropout 0 | 0.538 | −0.021 |
| loss weights 1/persistence MSE | 0.524 | −0.014 |
| combo: SEQ12 + wind edges + multi-lead | 0.516 | −0.008 |
| arch: station GRU then distance-biased attention | 0.454 | −0.062 |

Three structural changes account for 62% of the 0.322 gain: residual on persistence, raw inputs skipping the GCN, and GRU-per-station then attention.

## 5. Final architecture

1. Per station: the last 12 h of 10 features, calendar sin/cos and an 8-d learned station embedding go into a GRU (hidden 128).
2. Across stations: self-attention (`nn.TransformerEncoderLayer`) with an additive bias of −|α| × distance, so near stations count more.
3. Head: a linear layer predicts the change from persistence for every lead t+1 … t+h. The loss is weighted by 1/persistence MSE per target.
4. Training: bf16 autocast, Adam with LR 5e-4 decaying linearly to 0 over the 5 minutes, dropout 0, batch 256.

That is 433k parameters at 6 h. The 1 h version uses 6 h of history and 3 attention layers (564k).

| model | branch @ commit (`train.py`) |
|---|---|
| teammate baseline | `b01d71b` |
| best 6 h | `dev_james_branch` @ `86340f2` |
| best 1 h | `dev_james_h1` @ `2d20a9d` |
