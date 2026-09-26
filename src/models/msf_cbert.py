"""MSF-CBERT reference model. Dimensions:
text a_t in R^768, graph g_t in R^256, concat R^1024, gate & fusion in R^512.

Graph path (R-GCN-lite with temporal decay):
  for each relation r: m_r = W_r · (Σ_j w_j · e_j)   over train-neighbor cached embeddings
  g_t = mean over relations present (learned UNK_r when the entity is unseen/empty)
This is a 2-hop event→entity→event aggregation with per-relation transforms — the
message-passing structure described here, kept explicit so the leakage properties
are inspectable. # ADAPT: swap in your exact R-GCN stack here if it differs; the
GraphContext interface (train-only neighbors, past_only flag) stays the same.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)   # inverse-frequency, computed on TRAIN only
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        p_t = p.gather(1, target[:, None]).squeeze(1)
        a_t = self.alpha[target]
        return (-a_t * (1 - p_t) ** self.gamma * logp_t).mean()


# ---------------------------------------------------------------------------
# Long-tail loss variants. Every one of these is a function of the TRAIN class
# counts only (invariant 4: balancing acts on train, never on val/test), so they
# introduce no new artifact and no new audit surface.
# ---------------------------------------------------------------------------

class ClassBalancedFocalLoss(nn.Module):
    """Cui et al., CVPR 2019 — reweight by the inverse *effective number* of samples,
    (1-beta)/(1-beta^n_c), rather than by raw inverse frequency."""

    def __init__(self, counts, beta=0.9999, gamma=2.0):
        super().__init__()
        counts = torch.as_tensor(counts, dtype=torch.float)
        eff = 1.0 - torch.pow(beta, counts)
        w = (1.0 - beta) / torch.clamp(eff, min=1e-12)
        w = w / w.sum() * len(counts)          # mean weight 1
        self.register_buffer("alpha", w)
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        p_t = logp_t.exp()
        return (-self.alpha[target] * (1 - p_t) ** self.gamma * logp_t).mean()


class LDAMLoss(nn.Module):
    """Cao et al., NeurIPS 2019 — subtract a class-dependent margin m_j proportional to
    n_j^(-1/4) from the true-class logit, then scale. `set_epoch` implements DRW:
    class weights stay uniform until `drw_start`, then switch to class-balanced."""

    def __init__(self, counts, max_m=0.5, s=30.0, beta=0.9999, drw_start=None):
        super().__init__()
        counts = torch.as_tensor(counts, dtype=torch.float)
        m = 1.0 / torch.sqrt(torch.sqrt(torch.clamp(counts, min=1.0)))
        m = m * (max_m / m.max())
        self.register_buffer("m_list", m)
        eff = 1.0 - torch.pow(beta, counts)
        cb = (1.0 - beta) / torch.clamp(eff, min=1e-12)
        self.register_buffer("cb_w", cb / cb.sum() * len(counts))
        self.register_buffer("uniform_w", torch.ones_like(cb))
        self.s, self.drw_start = s, drw_start
        self.weight_key = "uniform_w"

    def set_epoch(self, epoch):
        if self.drw_start is not None:
            self.weight_key = "cb_w" if epoch >= self.drw_start else "uniform_w"

    def forward(self, logits, target):
        margins = self.m_list[target].to(logits.dtype)   # buffers stay fp32 under autocast
        adjusted = logits.clone()
        idx = torch.arange(logits.size(0), device=logits.device)
        adjusted[idx, target] = adjusted[idx, target] - margins
        return F.cross_entropy(self.s * adjusted, target,
                               weight=getattr(self, self.weight_key))


class LogitAdjustedLoss(nn.Module):
    """Menon et al., ICLR 2021 (loss-modification form) — add tau * log(prior) to the
    logits during training; at inference the head is used unchanged."""

    def __init__(self, counts, tau=1.0):
        super().__init__()
        counts = torch.as_tensor(counts, dtype=torch.float)
        prior = counts / counts.sum()
        self.register_buffer("adjust", tau * torch.log(torch.clamp(prior, min=1e-12)))

    def forward(self, logits, target):
        return F.cross_entropy(logits + self.adjust, target)


class BalancedSoftmaxLoss(nn.Module):
    """Ren et al., "Balanced Meta-Softmax", NeurIPS 2020 — add log(n_j) to the logits
    during training.

    VERIFIED EQUIVALENCE: at tau=1 this is numerically identical to LogitAdjustedLoss,
    because log(n_j/N) and log(n_j) differ by the constant log(N), which softmax is
    invariant to. Both are kept because the methods are distinct, but they must not be
    reported as two independent results at tau=1 — run logit adjustment at a different
    tau (e.g. 0.5) if a genuinely separate arm is wanted."""

    def __init__(self, counts):
        super().__init__()
        counts = torch.as_tensor(counts, dtype=torch.float)
        self.register_buffer("log_n", torch.log(torch.clamp(counts, min=1.0)))

    def forward(self, logits, target):
        return F.cross_entropy(logits + self.log_n, target)


def make_loss(name, counts, alpha, cfg):
    """Factory. `counts` are per-class TRAIN counts; `alpha` the existing
    inverse-frequency vector kept for the default focal loss (unchanged behaviour)."""
    m = cfg.get("model", {})
    gamma = m.get("focal_gamma", 2.0)
    if name == "focal":
        return FocalLoss(alpha, gamma)
    if name == "cb_focal":
        return ClassBalancedFocalLoss(counts, beta=m.get("cb_beta", 0.9999), gamma=gamma)
    if name == "ldam_drw":
        return LDAMLoss(counts, max_m=m.get("ldam_max_m", 0.5), s=m.get("ldam_s", 30.0),
                        drw_start=m.get("ldam_drw_start", 2))
    if name == "logit_adjust":
        return LogitAdjustedLoss(counts, tau=m.get("logit_adjust_tau", 1.0))
    if name == "balanced_softmax":
        return BalancedSoftmaxLoss(counts)
    raise ValueError(f"unknown loss {name!r}")


class GatedFusion(nn.Module):
    def __init__(self, text_dim=768, graph_dim=256, fused=512, dropout=0.1):
        super().__init__()
        self.proj_a = nn.Linear(text_dim, fused)
        self.proj_g = nn.Linear(graph_dim, fused)
        self.gate = nn.Linear(text_dim + graph_dim, fused)
        self.drop = nn.Dropout(dropout)

    def forward(self, a, g):
        z = torch.sigmoid(self.gate(torch.cat([a, g], dim=-1)))       # R^512
        f = z * torch.tanh(self.proj_a(a)) + (1 - z) * torch.tanh(self.proj_g(g))
        return self.drop(f)


class MSFCBert(nn.Module):
    def __init__(self, cfg, relations, num_classes, use_graph=True, struct_meta=None):
        """num_classes is passed explicitly: it depends on the label scheme (D4:
        with_unknown=9, without_unknown=8), not on a fixed config value.
        struct_meta (D13, default None -> state_dict identical to matrix runs):
        {"num_dim": int, "cat_cols": [..], "cat_vocab_sizes": {col: size}} enables a
        dense structured branch — train-standardized numerics (+ optional train-vocab
        categorical embeddings, index 0 = UNK/unseen/"Unknown") gated into the fused
        representation. Numerics are disjoint from the verbalized categoricals, so
        nothing enters the model twice."""
        super().__init__()
        m = cfg["model"]
        # dtype pinned: transformers>=5 loads checkpoints in their stored dtype, and
        # microsoft/deberta-v3-base is stored in fp16, which breaks GradScaler.unscale_
        # ("Attempting to unscale FP16 gradients"). ConfliBERT/ModernBERT are fp32 already,
        # so this is a no-op for every recorded run.
        self.encoder = AutoModel.from_pretrained(m["encoder"], dtype=torch.float32)
        self.use_graph = use_graph
        self.relations = list(relations)
        gd = cfg["graph"]["dim"]
        # Stage-6a levers (config flags, default OFF -> state_dict identical to matrix runs)
        self.rel_attention = bool(m.get("rel_attention", False))
        self.aux_graph_loss = bool(m.get("aux_graph_loss", False))
        self.mo_prior = bool(m.get("mo_prior", False))
        if use_graph:
            self.rel_proj = nn.ModuleDict({r: nn.Linear(m["text_dim"], gd) for r in self.relations})
            self.rel_unk = nn.ParameterDict({r: nn.Parameter(torch.zeros(gd)) for r in self.relations})
            self.fusion = GatedFusion(m["text_dim"], gd, m["fusion_dim"], m["dropout"])
            head_in = m["fusion_dim"]
            if self.rel_attention:                      # softmax over relations, text query
                self.attn_q = nn.Linear(m["text_dim"], gd)
            if self.mo_prior:                           # decay-weighted TRAIN-neighbor label
                self.mo_proj = nn.Linear(9 * len(self.relations), gd)   # histograms (9-label pool)
            if self.aux_graph_loss:                     # deep supervision on the graph path
                self.graph_head = nn.Linear(gd, num_classes)
        else:
            self.text_head_proj = nn.Sequential(
                nn.Linear(m["text_dim"], m["fusion_dim"]), nn.Tanh(), nn.Dropout(m["dropout"]))
            head_in = m["fusion_dim"]
        self.struct_dense = struct_meta is not None
        if self.struct_dense:
            self.struct_cat_cols = list(struct_meta["cat_cols"])
            self.cat_emb = nn.ModuleDict({
                c: nn.Embedding(struct_meta["cat_vocab_sizes"][c], 32)
                for c in self.struct_cat_cols})
            sd = 256
            s_in = struct_meta["num_dim"] + 32 * len(self.struct_cat_cols)
            self.struct_mlp = nn.Sequential(
                nn.Linear(s_in, sd), nn.Tanh(), nn.Dropout(m["dropout"]))
            self.struct_proj = nn.Linear(sd, m["fusion_dim"])
            self.struct_gate = nn.Linear(m["fusion_dim"] + sd, m["fusion_dim"])
        # availability-conditioned outcome adapter. The
        # post-event fields get their own gated residual branch, scaled by a per-event
        # availability indicator m in {0,1}; with m = 0 the output does not depend on them.
        # Absent from struct_meta -> no parameters added (state_dict identical to older runs).
        self.out_dim = int((struct_meta or {}).get("out_dim", 0))
        if self.out_dim:
            od = 64
            self.out_mlp = nn.Sequential(nn.Linear(self.out_dim, od), nn.Tanh(),
                                         nn.Dropout(m["dropout"]))
            self.out_proj = nn.Linear(od, m["fusion_dim"])
            self.out_gate = nn.Linear(m["fusion_dim"] + od, m["fusion_dim"])
        self.classifier = nn.Linear(head_in, num_classes)

    def encode_text(self, input_ids, attention_mask, **_):
        # **_ swallows tokenizer extras like token_type_ids (BERT-family); the
        # encoder's default zeros are correct for single-segment input.
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]                             # [CLS], R^768

    def graph_vector(self, neigh, a=None, mo=None):
        """neigh: per-sample dict rel -> (agg_embedding R^768 | None). Batched by caller."""
        vecs = []
        for r in self.relations:
            agg = neigh[r]                                             # (B,768) with NaN rows for UNK
            proj = self.rel_proj[r](torch.nan_to_num(agg))
            unk = self.rel_unk[r].expand_as(proj)
            mask = torch.isnan(agg[:, :1])                             # (B,1) True where unseen
            vecs.append(torch.where(mask, unk, proj))
        V = torch.stack(vecs, dim=1)                                   # (B, R, 256)
        if self.rel_attention:
            q = self.attn_q(a).unsqueeze(2)                            # (B, 256, 1)
            w = torch.softmax(torch.bmm(V, q).squeeze(-1) / V.shape[-1] ** 0.5, dim=1)
            g = (V * w.unsqueeze(-1)).sum(dim=1)
        else:
            g = V.mean(dim=1)                                          # (B,256)
        if self.mo_prior and mo is not None:
            g = g + self.mo_proj(mo)
        return g

    def struct_vector(self, snum, scat):
        parts = [snum]
        for i, c in enumerate(self.struct_cat_cols):
            parts.append(self.cat_emb[c](scat[:, i]))
        return self.struct_mlp(torch.cat(parts, dim=-1))

    def forward(self, input_ids, attention_mask, neigh=None, mo=None,
                snum=None, scat=None, return_aux=False, sout=None, avail=None,
                return_feat=False):
        """sout: (B, out_dim) standardised outcome fields; avail: (B,) in {0,1},
        None = all available. return_feat: also return the pre-classifier vector."""
        a = self.encode_text(input_ids, attention_mask)
        if self.use_graph:
            g = self.graph_vector(neigh, a, mo)
            f = self.fusion(a, g)
        else:
            f = self.text_head_proj(a)
        if self.struct_dense and snum is not None:
            s = self.struct_vector(snum, scat)
            z = torch.sigmoid(self.struct_gate(torch.cat([f, s], dim=-1)))
            f = z * f + (1 - z) * torch.tanh(self.struct_proj(s))
        if self.out_dim:
            assert sout is not None, "outcome-adapter model needs sout"
            h = self.out_mlp(sout)
            u = torch.sigmoid(self.out_gate(torch.cat([f, h], dim=-1)))
            mvec = (torch.ones(f.shape[0], 1, device=f.device, dtype=f.dtype) if avail is None
                    else avail.view(-1, 1).to(f.dtype))
            f = f + mvec * u * torch.tanh(self.out_proj(h))
        logits = self.classifier(f)
        if return_feat:
            return logits, f
        if self.use_graph and return_aux and self.aux_graph_loss:
            return logits, self.graph_head(g)
        return logits
