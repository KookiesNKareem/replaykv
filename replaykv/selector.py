# SPDX-License-Identifier: Apache-2.0
"""Learned block selector — ported from memory_decode_prototype.py.

A tiny MLP scores compressed KV blocks from cache-local features (coarse attention mass,
key/value dispersion, Quest min/max bound, norms, recency) to pick which blocks to keep
exact. The features are computed here from the resident block K/V tensors (so the same code
runs at calibration time and at decode time inside the vLLM backend).

Feature math is identical to the prototype's features_resident / coarse_logits_and_quest /
compute_resident_stats, so a selector trained by calibrate_selector.py transfers exactly.
MLP: Linear(F->32) -> GELU -> Linear(32->1); trained with cross-entropy of softmax(scores)
vs normalized oracle block mass.
"""
import math
import torch

FEATURE_NAMES = [
    "mass", "log_mass", "rel_logit", "key_bound", "quest_rel",
    "value_radius", "value_rms", "value_expected", "key_block_norm",
    "value_block_norm", "recency",
]
F_DIM = len(FEATURE_NAMES)
HIDDEN = 32


def make_mlp(device=None, hidden=HIDDEN):
    return torch.nn.Sequential(
        torch.nn.Linear(F_DIM, hidden, device=device), torch.nn.GELU(),
        torch.nn.Linear(hidden, 1, device=device))


@torch.no_grad()
def block_features(q, k_blocks, v_blocks, block_size):
    """Cache-local per-block features [Hkv, nb, F] from full block K/V.

    q        : [Hq, hd]              the decode query
    k_blocks : [Hkv, nb, bs, hd]     exact keys of nb FULL blocks (bs == block_size)
    v_blocks : [Hkv, nb, bs, hd]
    """
    Hkv, nb, bs, hd = k_blocks.shape
    rep = q.shape[0] // Hkv
    scale = 1.0 / math.sqrt(hd)
    k_blocks = k_blocks.float(); v_blocks = v_blocks.float()
    k_sum = k_blocks.mean(2); v_sum = v_blocks.mean(2)                 # [Hkv, nb, hd]
    k_min = k_blocks.amin(2); k_max = k_blocks.amax(2)
    key_radius = torch.linalg.vector_norm(k_blocks - k_sum[:, :, None], dim=-1).amax(2)  # [Hkv,nb]
    vdelta = torch.linalg.vector_norm(v_blocks - v_sum[:, :, None], dim=-1)              # [Hkv,nb,bs]
    value_radius = vdelta.amax(2)
    value_rms = torch.sqrt(vdelta.square().mean(2).clamp_min(1e-12))
    key_block_norm = torch.linalg.vector_norm(k_sum, dim=-1)
    value_block_norm = torch.linalg.vector_norm(v_sum, dim=-1)

    q_kv = q.view(Hkv, rep, hd).mean(1).float()                       # [Hkv, hd]
    coarse = torch.einsum("gd,gnd->gn", q_kv, k_sum) * scale + math.log(block_size)
    quest = torch.maximum(q_kv[:, None, :] * k_min, q_kv[:, None, :] * k_max).sum(-1) * scale
    mass = torch.softmax(coarse, dim=-1)
    q_norm = torch.linalg.vector_norm(q_kv, dim=-1, keepdim=True)
    key_bound = q_norm * key_radius / math.sqrt(hd)
    idx = torch.arange(nb, device=q.device, dtype=torch.float32)
    recency = ((nb - 1 - idx) / max(1, nb - 1))[None, :].expand(Hkv, nb)
    cols = {
        "mass": mass, "log_mass": torch.log(mass.clamp_min(1e-9)),
        "rel_logit": coarse - coarse.amax(-1, keepdim=True),
        "key_bound": key_bound, "quest_rel": quest - quest.amax(-1, keepdim=True),
        "value_radius": value_radius, "value_rms": value_rms,
        "value_expected": mass * value_radius,
        "key_block_norm": key_block_norm, "value_block_norm": value_block_norm,
        "recency": recency,
    }
    return torch.stack([cols[n] for n in FEATURE_NAMES], dim=-1)       # [Hkv, nb, F]


@torch.no_grad()
def resident_block_features(q, kbar, kmin, kmax, kradius, vbar, vradius, vrms, block_size):
    """Identical features to block_features(), but from RESIDENT per-block stats only — no full K/V.

    The nonresident path keeps exact KV on host; these small per-block stats (mean/min/max/radius/rms)
    are the GPU-resident summaries maintained in _capture_nonresident, so the learned MLP can score
    every logical block without staging it. The q-dependent cols (mass/quest/key_bound) recompute here
    with the live decode q; the q-independent cols (radius/rms/norms) come straight from the stats.

    q        : [Hq, hd]
    kbar/kmin/kmax/vbar  : [nb, Hkv, hd]   per-block mean / channel-min / channel-max key, mean value
    kradius/vradius/vrms : [nb, Hkv]        max||k-kbar||, max||v-vbar||, rms||v-vbar|| over block tokens
    """
    nb, Hkv, hd = kbar.shape
    rep = q.shape[0] // Hkv
    scale = 1.0 / math.sqrt(hd)
    kbar = kbar.transpose(0, 1).float(); vbar = vbar.transpose(0, 1).float()   # [Hkv, nb, hd]
    kmin = kmin.transpose(0, 1).float(); kmax = kmax.transpose(0, 1).float()
    key_radius = kradius.transpose(0, 1).float()                               # [Hkv, nb]
    value_radius = vradius.transpose(0, 1).float(); value_rms = vrms.transpose(0, 1).float()
    key_block_norm = torch.linalg.vector_norm(kbar, dim=-1)
    value_block_norm = torch.linalg.vector_norm(vbar, dim=-1)
    q_kv = q.view(Hkv, rep, hd).mean(1).float()                               # [Hkv, hd]
    coarse = torch.einsum("gd,gnd->gn", q_kv, kbar) * scale + math.log(block_size)
    quest = torch.maximum(q_kv[:, None, :] * kmin, q_kv[:, None, :] * kmax).sum(-1) * scale
    mass = torch.softmax(coarse, dim=-1)
    q_norm = torch.linalg.vector_norm(q_kv, dim=-1, keepdim=True)
    key_bound = q_norm * key_radius / math.sqrt(hd)
    idx = torch.arange(nb, device=q.device, dtype=torch.float32)
    recency = ((nb - 1 - idx) / max(1, nb - 1))[None, :].expand(Hkv, nb)
    cols = {
        "mass": mass, "log_mass": torch.log(mass.clamp_min(1e-9)),
        "rel_logit": coarse - coarse.amax(-1, keepdim=True),
        "key_bound": key_bound, "quest_rel": quest - quest.amax(-1, keepdim=True),
        "value_radius": value_radius, "value_rms": value_rms,
        "value_expected": mass * value_radius,
        "key_block_norm": key_block_norm, "value_block_norm": value_block_norm,
        "recency": recency,
    }
    return torch.stack([cols[n] for n in FEATURE_NAMES], dim=-1)       # [Hkv, nb, F]


def fit_selector(train_x, train_y, device, hidden=HIDDEN, epochs=200, lr=0.02):
    """train_x [N, nb, F] raw features; train_y [N, nb] normalized oracle block mass.
    Returns (mlp, mean, std)."""
    flat = train_x.reshape(-1, train_x.shape[-1])
    mean = flat.mean(0); std = flat.std(0).clamp_min(1e-6)
    xs = (train_x - mean) / std
    mlp = make_mlp(device, hidden)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        scores = mlp(xs).squeeze(-1)
        loss = -(train_y * torch.log_softmax(scores, dim=-1)).sum(-1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    mlp.eval()
    return mlp, mean.detach(), std.detach()


class LearnedSelector:
    """Loadable scorer: features -> per-block scores. Selection in the backend is shared across
    kv-heads (mean over heads) to match the shared-block mixed-attention layout."""

    def __init__(self, mlp, mean, std):
        self.mlp = mlp
        self.mean = mean
        self.std = std

    @torch.no_grad()
    def score_shared(self, q, k_blocks, v_blocks, block_size):
        """Return per-block scores [nb] (averaged over kv-heads) for the FULL blocks."""
        feats = block_features(q, k_blocks, v_blocks, block_size)      # [Hkv, nb, F]
        xs = (feats - self.mean.to(feats)) / self.std.to(feats)
        scores = self.mlp(xs).squeeze(-1)                             # [Hkv, nb]
        return scores.mean(0)                                          # [nb]

    @torch.no_grad()
    def score_resident(self, q, kbar, kmin, kmax, kradius, vbar, vradius, vrms, block_size):
        """Per-block scores [nb] from resident stats — the nonresident-path scorer (no full K/V)."""
        feats = resident_block_features(q, kbar, kmin, kmax, kradius, vbar, vradius, vrms, block_size)
        xs = (feats - self.mean.to(feats)) / self.std.to(feats)
        scores = self.mlp(xs).squeeze(-1)                             # [Hkv, nb]
        return scores.mean(0)                                          # [nb]

    def to(self, device):
        self.mlp.to(device); self.mean = self.mean.to(device); self.std = self.std.to(device)
        return self

    def save(self, path):
        torch.save({"state_dict": self.mlp.state_dict(), "mean": self.mean.cpu(),
                    "std": self.std.cpu(), "hidden": HIDDEN,
                    "feature_names": FEATURE_NAMES}, path)

    @classmethod
    def load(cls, path, device="cuda"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        assert ckpt["feature_names"] == FEATURE_NAMES, "feature set mismatch vs trained selector"
        mlp = make_mlp(device, ckpt.get("hidden", HIDDEN))
        mlp.load_state_dict(ckpt["state_dict"]); mlp.eval()
        return cls(mlp, ckpt["mean"].to(device), ckpt["std"].to(device))
