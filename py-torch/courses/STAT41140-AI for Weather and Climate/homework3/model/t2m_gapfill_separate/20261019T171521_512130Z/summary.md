# Five-minute presentation

1. Problem and data (45 s): reconstruct deliberately hidden cells in daily sampled-mean ERA5 t2m; no temporal forecasting. {'train': 1826, 'validation': 365, 'test': 366}

2. Preprocessing and fairness (45 s): training seasonal climatology → anomalies → training SD scaling; separate mask channel, identical hidden pixels and chronological splits.

3. Separate neural experiments (60 s): ConvAE uses convolutional spatial encoding/decoding; Spatial Transformer uses attention between patches in one map. Harmonic interpolation is the local baseline; training-fitted PCA/EOF is the learned linear baseline, with visible-only coefficient inference.

4. Separate baseline comparisons (75 s), selected ConvAE k=64, EOF k=32: Interpolation: RMSE 0.446 °C, MAE 0.230 °C; PCA/EOF: RMSE 0.282 °C, MAE 0.191 °C; ConvAE: RMSE 0.351 °C, MAE 0.248 °C; Spatial Transformer: RMSE 0.609 °C, MAE 0.430 °C. Show equal-k validation curves, the four-method test table and paired gains versus interpolation.

5. Reconstruction and limits (75 s): show fixed_test_reconstruction.png and anomaly category results. Daily reanalysis, artificial gaps, one seed and limited training budgets constrain generalization. No uncertainty-calibrated predictions or station-observed skill are established.

## Questions answered by the measured tables

Lowest pooled test RMSE: PCA/EOF. A point estimate is not proof of superiority; report MAE, gap-specific results and intervals.

Gap dependence: small: lowest RMSE PCA/EOF; large: lowest RMSE PCA/EOF

Anomaly dependence: inspect category counts, RMSE, MAE and bias; avoid tail claims from too few unique days.

Attention interpretation: spatial patches inside one daily field, not previous days. No attention-weight physical explanation is assumed.