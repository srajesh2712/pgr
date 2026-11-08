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
SEQ_LEN    = 6
IN_FEATS   = list(range(10))     # 8 teammate features + wind_u, wind_v
HIDDEN     = 64
GRU_HIDDEN = 128
NUM_GCN    = 2
DROPOUT    = 0.0
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
    def __init__(self, in_dim, out_dim, n_stations, geom, emb_dim=8):
        super().__init__()
        self.register_buffer("geom", geom)                                  # (S, S, 3): dist/100, unit vec j->i
        self.edge_mlp = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 1))
        self.emb = nn.Parameter(torch.randn(n_stations, emb_dim) * 0.1)  # learned station identity
        self.E1 = nn.Parameter(torch.randn(n_stations, 10) * 0.1)          # adaptive adjacency
        self.E2 = nn.Parameter(torch.randn(n_stations, 10) * 0.1)
        in_dim = in_dim + emb_dim
        dims = [in_dim] + [HIDDEN] * NUM_GCN
        self.gcn = nn.ModuleList(GraphConv(a, b, DROPOUT) for a, b in zip(dims, dims[1:]))
        self.gru = nn.GRU(HIDDEN + in_dim, GRU_HIDDEN, num_layers=2, batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(nn.Linear(GRU_HIDDEN, GRU_HIDDEN // 2), nn.ReLU(),
                                  nn.Dropout(DROPOUT), nn.Linear(GRU_HIDDEN // 2, out_dim))

    def forward(self, x, adj):                    # x (B, T, S, F) -> (B, S, out)
        B, T, S, _ = x.shape
        x = torch.cat([x, self.emb[None, None].expand(B, T, -1, -1)], -1)
        adj = 0.5 * adj + 0.5 * torch.softmax(torch.relu(self.E1 @ self.E2.T), dim=1)
        # dynamic wind-aware edges: score j->i from geometry + wind at source j
        uv = x[..., 8:10]                                                   # (B, T, S, 2) wind_u, wind_v
        uvj = uv[:, :, None, :, :].expand(-1, -1, S, -1, -1)                # [b,t,i,j] = wind at j
        g = self.geom[None, None].expand(B, T, -1, -1, -1)
        adv = (uvj * g[..., 1:3]).sum(-1, keepdim=True)                     # wind at j blowing toward i
        logits = self.edge_mlp(torch.cat([g, uvj, adv], -1)).squeeze(-1)    # (B, T, S, S)
        adj = 0.5 * adj + 0.5 * torch.softmax(logits, -1)                   # (B, T, S, S)
        adj = adj.reshape(B * T, S, S)
        h = x.reshape(B * T, S, -1)
        for g in self.gcn:
            h = g(h, adj)
        h = torch.cat([h.reshape(B, T, S, -1), x], -1)             # skip: own-station raw inputs
        h = h.permute(0, 2, 1, 3).reshape(B * S, T, -1)
        _, hn = self.gru(h)
        return self.head(hn[-1]).reshape(B, S, -1)


class TSAttn(nn.Module):
    """Temporal-then-spatial: per-station GRU over history, then station self-attention
    with a learned distance bias (one spatial mixing step instead of one per hour)."""
    def __init__(self, in_dim, out_dim, n_stations, geom, emb_dim=8, n_layers=3, n_heads=4):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(n_stations, emb_dim) * 0.1)
        self.register_buffer("dist", geom[..., 0])                           # (S, S) in 100 km
        self.alpha = nn.Parameter(torch.tensor(1.0))                         # distance penalty
        self.gru = nn.GRU(in_dim + emb_dim, GRU_HIDDEN, num_layers=2, batch_first=True)
        self.attn = nn.ModuleList(nn.TransformerEncoderLayer(GRU_HIDDEN, n_heads, 2 * GRU_HIDDEN,
                                  dropout=0.0, batch_first=True, norm_first=True) for _ in range(n_layers))
        self.head = nn.Sequential(nn.Linear(GRU_HIDDEN, GRU_HIDDEN // 2), nn.ReLU(),
                                  nn.Linear(GRU_HIDDEN // 2, out_dim))

    def forward(self, x, adj):                    # x (B, T, S, F) -> (B, S, out)
        B, T, S, _ = x.shape
        x = torch.cat([x, self.emb[None, None].expand(B, T, -1, -1)], -1)
        _, hn = self.gru(x.permute(0, 2, 1, 3).reshape(B * S, T, -1))
        h = hn[-1].reshape(B, S, -1)
        bias = (-self.alpha.abs() * self.dist).to(h.dtype)                   # additive attention bias
        for layer in self.attn:
            h = layer(h, src_mask=bias)
        return self.head(h)


# ── Training ────────────────────────────────────────────────────────────────────

def main():
    torch.backends.mha.set_fastpath_enabled(False)                   # fast path NaNs with float masks in eval
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

    NT = len(TARGETS)
    def forward_all(t):        # (B, S, HORIZON, NT): residual on persistence for t+1..t+HORIZON
        out = model(window(t), adj).reshape(len(t), -1, HORIZON, NT)
        return out + Y[t][:, :, None, :]

    def forward(t):
        return forward_all(t)[:, :, -1]                                  # t+HORIZON only (scored)

    lat = np.deg2rad(d["lat"].astype(np.float32)); lon = np.deg2rad(d["lon"].astype(np.float32))
    dxk = (lon[None, :] - lon[:, None]) * np.cos(lat.mean()) * 6371   # [i,j] = j minus i (east, km)
    dyk = (lat[None, :] - lat[:, None]) * 6371
    dist = np.sqrt(dxk**2 + dyk**2) + 1e-6
    geom = np.stack([dist / 100, -dxk / dist, -dyk / dist], -1)          # unit vector from j to i
    geom[np.arange(S_), np.arange(S_), 1:] = 0
    model = TSAttn(len(IN_FEATS) + 6, len(TARGETS) * HORIZON, S_, torch.tensor(geom, dtype=torch.float32)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    n_params = sum(p.numel() for p in model.parameters())
    train_t = torch.tensor(d["split_idx"]["train"], device=dev)
    # per-target weight = 1 / persistence MSE (normalised units) -> loss mirrors the metric
    pm = M[train_t] & M[train_t + HORIZON]
    pmse = (((Y[train_t + HORIZON] - Y[train_t]) ** 2) * pm).sum((0, 1)) / pm.sum((0, 1))
    W = (1.0 / pmse) / (1.0 / pmse).mean()
    print("target weights:", [f"{w:.2f}" for w in W.tolist()])

    print(f"device={dev}  params={n_params:,}  budget={TIME_BUDGET/60:.1f}min  horizon={HORIZON}h")
    t_start, step, ema = time.time(), 0, None
    model.train()
    while time.time() - t_start < TIME_BUDGET:
        t = train_t[torch.randint(len(train_t), (BATCH,), device=dev)]
        th = t[:, None] + torch.arange(1, HORIZON + 1, device=dev)        # (B, HORIZON)
        y, m = Y[th].transpose(1, 2), M[th].transpose(1, 2)               # (B, S, HORIZON, NT)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pred = forward_all(t)
        loss = ((pred.float() - y) ** 2 * m * W).sum() / m.sum().clamp(min=1)  # all lead times
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
