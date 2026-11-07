"""
train.py — the ONE file the autoresearch agent edits.

Current version: teammate's ST-GCN (homework4/train-gnn.py) ported unchanged
(same layers, sizes, optimiser, inputs) but predicting t+HORIZON and trained
for a fixed wall-clock budget with linear LR decay. This is the baseline to beat.

Contract with prepare.py (do not break):
  - train for at most TIME_BUDGET minutes (env var, default 5) of wall-clock
  - only use data at hours <= t to predict t+HORIZON
  - finish by calling evaluate(...) on "val" and printing the summary block
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn

from prepare import HORIZON, TARGET_IDX, TARGETS, evaluate, load_data

TIME_BUDGET = float(os.environ.get("TIME_BUDGET", 5)) * 60      # seconds
SEED = int(os.environ.get("SEED", 0))

# ── Hyperparameters (teammate's values) ─────────────────────────────────────────
SEQ_LEN    = 24
IN_FEATS   = list(range(8))      # teammate's 8 NODE_FEATURES (no wind vector)
HIDDEN     = 64
GRU_HIDDEN = 128
NUM_GCN    = 2
DROPOUT    = 0.2
LR         = 5e-4
WD         = 1e-5
BATCH      = 256


# ── Model ───────────────────────────────────────────────────────────────────────

class GraphConv(nn.Module):
    """H = dropout(relu(BN(A X W)))"""
    def __init__(self, in_f, out_f, dropout):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f)
        self.bn = nn.BatchNorm1d(out_f)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):                    # x (B, S, F), adj (S, S)
        x = self.lin(adj @ x)
        B, S, F = x.shape
        x = self.bn(x.reshape(B * S, F)).reshape(B, S, F)
        return self.drop(torch.relu(x))


class STGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        dims = [in_dim] + [HIDDEN] * NUM_GCN
        self.gcn = nn.ModuleList(GraphConv(a, b, DROPOUT) for a, b in zip(dims, dims[1:]))
        self.gru = nn.GRU(HIDDEN, GRU_HIDDEN, num_layers=2, batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(nn.Linear(GRU_HIDDEN, GRU_HIDDEN // 2), nn.ReLU(),
                                  nn.Dropout(DROPOUT), nn.Linear(GRU_HIDDEN // 2, out_dim))

    def forward(self, x, adj):                    # x (B, T, S, F) -> (B, S, out)
        B, T, S, _ = x.shape
        h = x.reshape(B * T, S, -1)
        for g in self.gcn:
            h = g(h, adj)
        h = h.reshape(B, T, S, -1).permute(0, 2, 1, 3).reshape(B * S, T, -1)
        _, hn = self.gru(h)
        return self.head(hn[-1]).reshape(B, S, -1)


# ── Training ────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_data()

    X = torch.tensor(d["X"], device=dev)                                 # (H, S, F)
    Y = torch.tensor(d["X"][..., TARGET_IDX], device=dev)                # normalised targets
    M = torch.tensor(d["mask"][..., TARGET_IDX], device=dev)
    adj = torch.tensor(d["adj"], device=dev)
    mu = torch.tensor(d["mu"][TARGET_IDX], device=dev)
    sd = torch.tensor(d["std"][TARGET_IDX], device=dev)
    offs = torch.arange(-SEQ_LEN + 1, 1, device=dev)
    feats = torch.tensor(IN_FEATS, device=dev)

    hod = torch.tensor(d["hour_of_day"], device=dev, dtype=torch.float32) * (2 * np.pi / 24)
    doy = torch.tensor(d["day_of_year"], device=dev, dtype=torch.float32) * (2 * np.pi / 365.25)
    CAL = torch.stack([hod.sin(), hod.cos(), doy.sin(), doy.cos()], -1)  # (H, 4)
    S_ = X.shape[1]

    def window(t):                                                       # t (B,) long
        ti = t[:, None] + offs                                           # (B, T)
        x = X[ti][..., feats]                                            # (B, T, S, F)
        cal = torch.cat([CAL[ti], CAL[t + HORIZON][:, None, :2].expand(-1, len(offs), -1)], -1)
        return torch.cat([x, cal[:, :, None, :].expand(-1, -1, S_, -1)], -1)

    def forward(t):                                                      # residual on persistence
        return model(window(t), adj) + Y[t]                             # (B, S, n_targets)

    model = STGCN(len(IN_FEATS) + 6, len(TARGETS)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    n_params = sum(p.numel() for p in model.parameters())
    train_t = torch.tensor(d["split_idx"]["train"], device=dev)

    print(f"device={dev}  params={n_params:,}  budget={TIME_BUDGET/60:.1f}min  horizon={HORIZON}h")
    t_start, step, ema = time.time(), 0, None
    model.train()
    while time.time() - t_start < TIME_BUDGET:
        t = train_t[torch.randint(len(train_t), (BATCH,), device=dev)]
        y, m = Y[t + HORIZON], M[t + HORIZON]
        pred = forward(t)
        loss = ((pred - y) ** 2 * m).sum() / m.sum().clamp(min=1)       # masked MSE
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # linear LR decay to 0 over the time budget (time-budget analogue of the
        # teammate's ReduceLROnPlateau; settles the final weights, cuts seed noise)
        for g in opt.param_groups:
            g["lr"] = LR * max(0.0, 1.0 - (time.time() - t_start) / TIME_BUDGET)
        opt.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        step += 1
        if step % 500 == 0:
            print(f"  step {step:6d}  loss {ema:.4f}  {time.time()-t_start:5.0f}s")
    train_secs = time.time() - t_start

    model.eval()
    @torch.no_grad()
    def predict(t_np):
        t = torch.as_tensor(t_np, device=dev)
        return (forward(t) * sd + mu).cpu().numpy()

    score, rmse = evaluate(predict, d, "val")
    if os.environ.get("FINAL_TEST") == "1":                              # only for the final report
        evaluate(predict, d, "test")
    os.makedirs("cache", exist_ok=True)
    torch.save(model.state_dict(), "cache/last_model.pt")

    vram = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0
    print("---")
    print(f"val_score: {score:.6f}")
    for k, v in rmse.items():
        print(f"rmse_{k}: {v:.4f}")
    print(f"params: {n_params}")
    print(f"steps: {step}")
    print(f"train_seconds: {train_secs:.0f}")
    print(f"peak_vram_mb: {vram:.0f}")


if __name__ == "__main__":
    main()
