"""Step 06 — Train / evaluate one configuration on one split.

Examples
  python -m src.train.train --split seed_42                       # full model (9-class)
  python -m src.train.train --split seed_42 --label-scheme without_unknown   # 8-class (D4)
  python -m src.train.train --split seed_42 --no-graph            # ablation
  python -m src.train.train --split seed_42 --no-aug
  python -m src.train.train --split seed_42 --no-structured
  python -m src.train.train --split seed_42 --balanced-sampler    # optional, TRAIN only
  python -m src.train.train --split chrono                        # temporal split (E2)
  python -m src.train.train --split chrono --past-only            # deployable mode (E2)

Leakage posture: train rows come from artifacts/augmented/train_aug_{split}.parquet
(original train + train-sourced paraphrases). Val/test rows are ORIGINAL events read via
the manifest. Graph neighbors always resolve inside the train-only GraphContext; with
--past-only they are additionally masked to dates strictly before the query event.

Label schemes (D4): the corpus and manifests keep all 9 labels. 'without_unknown'
drops Unknown-labeled events uniformly from train/val/test at run time (task
definition, not rebalancing). The graph neighbor pool intentionally remains the FULL
train partition in both schemes: neighbors contribute entities/dates/text, never labels.
"""
import argparse, json, os, random, time
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report
from src.util import load_config, log, ensure_dir, read_manifest
from src.graph.build_graph import GraphContext
from src.models.msf_cbert import MSFCBert, FocalLoss, make_loss
from src.features.verbalize import build_input_text, verbalize_numerics


class EventDataset(Dataset):
    def __init__(self, df, tokenizer, cfg, selected, gctx, label2id,
                 use_structured=True, use_graph=True, past_only=False, is_train=False,
                 struct_meta=None, num_meta=None):
        self.df = df.reset_index(drop=True)
        self.num_meta = num_meta                       # numerics-as-text arm (None = off)
        self.tok, self.cfg = tokenizer, cfg
        self.selected, self.gctx = selected, gctx
        self.label2id = label2id
        self.use_structured, self.use_graph = use_structured, use_graph
        self.past_only, self.is_train = past_only, is_train
        self.struct_meta = struct_meta                 # D13: train-fit scaler/vocab or None
        self.relations = list(cfg["graph"]["relations"].keys())
        self.relcols = cfg["graph"]["relations"]
        if use_graph:
            self.row_of = {e: i for i, e in enumerate(gctx.train_eventids)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        text = build_input_text(r, self.selected, self.use_structured)
        if self.num_meta is not None:
            text = (text + " " + verbalize_numerics(r, self.num_meta["num_cols"],
                                                    self.num_meta["medians"])).strip()
        item = {"text": text, "label": self.label2id[str(r["label"])],
                "eventid": str(r["eventid"])}
        if self.use_graph:
            ref = pd.Timestamp(r["date"]).toordinal()
            # original train events must not see themselves as neighbors; augmented rows
            # keep their source as a legitimate train neighbor.
            self_row = self.row_of.get(str(r["eventid"]), -1) if self.is_train else -1
            nb = {}
            for rel in self.relations:
                idx, w = self.gctx.neighbors(rel, r.get(self.relcols[rel], ""),
                                             ref, self.past_only, self_row)
                nb[rel] = (idx, w)
            item["neigh"] = nb
        if self.struct_meta is not None:
            sm = self.struct_meta
            num = np.empty(len(sm["num_cols"]), dtype=np.float32)
            for k, c in enumerate(sm["num_cols"]):
                v = r.get(c)
                v = sm["medians"][c] if pd.isna(v) else float(v)   # train-fit tierb medians
                num[k] = (v - sm["mean"][c]) / sm["std"][c]
            item["snum"] = np.clip(num, -5.0, 5.0)
            if sm.get("out_cols"):                         # outcome adapter input
                o = np.empty(len(sm["out_cols"]), dtype=np.float32)
                for k, c in enumerate(sm["out_cols"]):
                    v = r.get(c)
                    v = sm["medians"][c] if pd.isna(v) else float(v)
                    o[k] = (v - sm["mean"][c]) / sm["std"][c]
                item["sout"] = np.clip(o, -5.0, 5.0)
            if sm["cat_cols"]:
                item["scat"] = np.array(
                    [sm["vocab"][c].get(str(r.get(c)), 0) for c in sm["cat_cols"]],
                    dtype=np.int64)                                # 0 = UNK/unseen/"Unknown"
        return item


def make_collate(tokenizer, cfg, cache_holder, relations, use_graph):
    max_len = cfg["model"]["max_len"]
    mo_prior = bool(cfg["model"].get("mo_prior", False))

    def collate(batch):
        enc = tokenizer([b["text"] for b in batch], truncation=True, padding=True,
                        max_length=max_len, return_tensors="pt")
        out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
               "labels": torch.tensor([b["label"] for b in batch]),
               "eventids": [b["eventid"] for b in batch]}
        if use_graph:
            E = cache_holder["emb"]                     # (N_train, 768) float32 numpy
            neigh = {}
            for rel in relations:
                mat = np.full((len(batch), E.shape[1]), np.nan, dtype=np.float32)
                for j, b in enumerate(batch):
                    idx, w = b["neigh"][rel]
                    if idx is not None:
                        mat[j] = (E[idx] * w[:, None].astype(np.float32)).sum(0)
                neigh[rel] = torch.from_numpy(mat)
            out["neigh"] = neigh
            if mo_prior:
                # Decay-weighted label histogram of TRAIN neighbors per relation (6a
                # "MO prior"). Pool labels are TRAIN labels only; original
                # train events exclude themselves, so no row sees its own label.
                PL, K = cache_holder["pool_labels"], cache_holder["n_pool_labels"]
                mo = np.zeros((len(batch), len(relations) * K), dtype=np.float32)
                for j, b in enumerate(batch):
                    for ri, rel in enumerate(relations):
                        idx, w = b["neigh"][rel]
                        if idx is not None:
                            h = np.bincount(PL[idx], weights=w, minlength=K)[:K]
                            s = h.sum()
                            if s > 0:
                                mo[j, ri * K:(ri + 1) * K] = h / s
                out["mo"] = torch.from_numpy(mo)
        if "snum" in batch[0]:
            out["snum"] = torch.from_numpy(np.stack([b["snum"] for b in batch]))
            if "scat" in batch[0]:
                out["scat"] = torch.from_numpy(np.stack([b["scat"] for b in batch]))
            if "sout" in batch[0]:
                out["sout"] = torch.from_numpy(np.stack([b["sout"] for b in batch]))
        return out
    return collate


@torch.no_grad()
def encode_train_cache(model, tokenizer, texts, cfg, device):
    """Cache encoder embeddings for all ORIGINAL train events (graph neighbor pool)."""
    model.eval()
    bs = cfg["train"]["eval_batch_size"]
    out = np.zeros((len(texts), cfg["model"]["text_dim"]), dtype=np.float32)
    for s in range(0, len(texts), bs):
        enc = tokenizer(texts[s:s + bs], truncation=True, padding=True,
                        max_length=cfg["model"]["max_len"], return_tensors="pt").to(device)
        out[s:s + bs] = model.encode_text(**enc).float().cpu().numpy()
    model.train()
    return out


@torch.no_grad()
def evaluate_split(model, loader, device, use_graph, avail=None, feats=None):
    """avail: None/1 = outcome fields available (full regime), 0 = masked regime.
    feats: optional list that receives the pre-classifier vectors."""
    model.eval()
    ys, ps, probs, ids = [], [], [], []
    for b in loader:
        neigh = ({k: v.to(device) for k, v in b["neigh"].items()} if use_graph else None)
        mo = b["mo"].to(device) if "mo" in b else None
        snum = b["snum"].to(device) if "snum" in b else None
        scat = b["scat"].to(device) if "scat" in b else None
        sout = b["sout"].to(device) if "sout" in b else None
        av = (None if avail is None or sout is None
              else torch.full((sout.shape[0],), float(avail), device=device))
        out = model(b["input_ids"].to(device), b["attention_mask"].to(device), neigh,
                    mo=mo, snum=snum, scat=scat, sout=sout, avail=av,
                    return_feat=feats is not None)
        logits = out[0] if feats is not None else out
        if feats is not None:
            feats.append(out[1].float().cpu().numpy())
        pr = torch.softmax(logits.float(), -1).cpu().numpy()
        ps += pr.argmax(1).tolist(); probs += pr.tolist()
        ys += b["labels"].tolist(); ids += b["eventids"]
    model.train()
    return np.array(ys), np.array(ps), np.array(probs), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", required=True)
    ap.add_argument("--label-scheme", default="with_unknown",
                    choices=["with_unknown", "without_unknown"])
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--no-structured", action="store_true")
    ap.add_argument("--past-only", action="store_true")
    ap.add_argument("--balanced-sampler", action="store_true",
                    help="WeightedRandomSampler on TRAIN only (never val/test)")
    ap.add_argument("--max-train", type=int, default=0,
                    help="stratified subsample of train rows (smoke tests only)")
    ap.add_argument("--epochs", type=int, default=0, help="override config epochs (smoke tests)")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed; defaults to the split's seed number, else 42")
    ap.add_argument("--loss", default="focal",
                    choices=["focal", "cb_focal", "ldam_drw", "logit_adjust",
                             "balanced_softmax"],
                    help="training loss; all use TRAIN class counts only (invariant 4)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Seed every RNG that can affect a run, not torch's alone. --seed makes the seed
    # explicit rather than parsed out of the split name, under which 'chrono' was
    # permanently pinned to 42 and numpy/random were never seeded at all.
    seed = args.seed if args.seed is not None else (
        int(args.split.split("_")[-1]) if "seed" in args.split else 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    name = args.name or "_".join(
        [args.split] + [f for f, on in [("nounk", args.label_scheme == "without_unknown"),
                                        ("nograph", args.no_graph), ("noaug", args.no_aug),
                                        ("nostruct", args.no_structured),
                                        ("pastonly", args.past_only),
                                        ("balsamp", args.balanced_sampler),
                                        (args.loss, args.loss != "focal")] if on])
    run_dir = os.path.join(cfg["paths"]["runs_dir"], name); ensure_dir(run_dir)

    df_full = pd.read_parquet(cfg["paths"]["processed"])   # 9-label corpus (graph pool)
    drops = cfg["data"]["label_schemes"][args.label_scheme]
    df = df_full[~df_full["label"].isin(drops)]            # scheme-filtered task corpus
    man = read_manifest(cfg, args.split)
    selected = json.load(open(f"{cfg['paths']['features_dir']}/selected_{args.split}.json"))["selected"]
    labels = sorted(df["label"].unique()); label2id = {l: i for i, l in enumerate(labels)}
    log(f"scheme={args.label_scheme}: {len(labels)} classes, corpus {len(df):,}/{len(df_full):,}")

    if args.no_aug:
        train_df = df.merge(man[man.split == "train"], on="eventid")
    else:
        train_df = pd.read_parquet(f"{cfg['paths']['aug_dir']}/train_aug_{args.split}.parquet")
        train_df = train_df[~train_df["label"].isin(drops)]
    val_df = df.merge(man[man.split == "val"], on="eventid")
    test_df = df.merge(man[man.split == "test"], on="eventid")
    assert "is_augmented" not in val_df or (val_df.get("is_augmented", 0) == 0).all()
    if args.max_train and args.max_train < len(train_df):
        from sklearn.model_selection import train_test_split
        keep, _ = train_test_split(np.arange(len(train_df)), train_size=args.max_train,
                                   stratify=train_df["label"], random_state=0)
        train_df = train_df.iloc[keep].reset_index(drop=True)
        log(f"SMOKE: stratified train subsample -> {len(train_df):,} rows")
    log(f"train={len(train_df):,} (aug={0 if args.no_aug else int(train_df['is_augmented'].sum()):,}) "
        f"val={len(val_df):,} test={len(test_df):,}")

    use_graph = not args.no_graph
    gctx = GraphContext(f"{cfg['paths']['graph_dir']}/graph_{args.split}.pkl") if use_graph else None

    # Routing-isolation arm (2026-09-22): the dense branch's twelve numerics rendered as
    # text instead (same Tier-B kept set and train-fit medians; nothing else changes).
    num_meta = None
    if cfg["model"].get("num_verbalize"):
        assert not cfg["model"].get("struct_dense"), "num_verbalize and struct_dense are exclusive"
        tierb = json.load(open(f"{cfg['paths']['preprocess_dir']}/tierb_{args.split}.json"))
        num_meta = {"num_cols": tierb["kept_numeric"],
                    "medians": {c: float(tierb["numeric_medians"][c]) for c in tierb["kept_numeric"]}}
        json.dump(num_meta, open(f"{run_dir}/num_verbalize.json", "w"), indent=2)
        log(f"num_verbalize: {len(num_meta['num_cols'])} numerics appended as text")

    # D13: dense structured branch. All statistics fit on the ORIGINAL train partition
    # only (same class as focal alpha); numeric cols are tierb's PCC/VIF-pruned set with
    # its fingerprinted train-fit medians, so nothing is fit outside train (invariant 2).
    struct_meta = None
    if cfg["model"].get("struct_dense"):
        tierb = json.load(open(f"{cfg['paths']['preprocess_dir']}/tierb_{args.split}.json"))
        num_cols = tierb["kept_numeric"]
        # E7 (2026-09-22): optional exclusion of outcome-coded fields from the dense branch
        # (config-declared, row-local; the Tier-B kept set and medians are unchanged).
        excl = set(cfg["model"].get("struct_dense_exclude") or [])
        if excl:
            assert excl <= set(num_cols), f"struct_dense_exclude not in kept numerics: {excl - set(num_cols)}"
            num_cols = [c for c in num_cols if c not in excl]
            log(f"struct_dense_exclude: {sorted(excl)} -> {len(num_cols)} numerics remain")
        # outcome fields routed to the availability-
        # conditioned adapter; must be kept Tier-B numerics and disjoint from the base branch.
        out_cols = list(cfg["model"].get("outcome_branch") or [])
        if out_cols:
            assert set(out_cols) <= set(tierb["kept_numeric"]), set(out_cols) - set(tierb["kept_numeric"])
            assert not set(out_cols) & set(num_cols), "outcome fields must be excluded from the base branch"
            assert not set(out_cols) & set(selected), "outcome fields must not be verbalised as text"
            assert not cfg["model"].get("num_verbalize"), "outcome adapter excludes numerics-as-text"
            # no derived outcome proxy may stay in the base branch: casualties = nkill + nwound
            # (Tier-B drops it on every split today; this makes that a hard precondition)
            proxies = {"casualties"} & set(num_cols)
            assert not proxies, f"outcome proxy in the OAF base branch: {sorted(proxies)}"
            log(f"outcome adapter: {out_cols}, drop p={cfg['model'].get('outcome_drop_p', 0.5)}")
        stat_cols = num_cols + out_cols
        med = {c: float(tierb["numeric_medians"][c]) for c in stat_cols}
        if cfg["model"].get("numeric_impute", "median") == "zero":
            # ablation: prior-work zero-fill — every missing/unknown numeric becomes 0
            med = {c: 0.0 for c in stat_cols}
        tr_orig = df.merge(man[man.split == "train"], on="eventid")
        X = tr_orig[stat_cols].astype(float).fillna(med)
        mu = {c: float(X[c].mean()) for c in stat_cols}
        sd = {c: float(X[c].std()) or 1.0 for c in stat_cols}
        cat_cols, vocab = [], {}
        if cfg["model"].get("struct_dense_cat"):
            cat_cols = list(selected)
            for c in cat_cols:
                vals = sorted(v for v in tr_orig[c].dropna().astype(str).unique()
                              if v not in ("", "Unknown"))
                vocab[c] = {v: i + 1 for i, v in enumerate(vals)}   # 0 = UNK
        struct_meta = {"num_cols": num_cols, "medians": med, "mean": mu, "std": sd,
                       "cat_cols": cat_cols, "vocab": vocab,
                       "cat_vocab_sizes": {c: len(vocab[c]) + 1 for c in cat_cols},
                       "num_dim": len(num_cols),
                       "out_cols": out_cols, "out_dim": len(out_cols)}
        # `vocab` and `num_dim` are persisted too: without them the categorical branch
        # cannot be reconstructed at inference, which is what stopped infer_partition.py
        # from loading struct_dense runs at all.
        json.dump({k: struct_meta[k] for k in
                   ["num_cols", "medians", "mean", "std", "cat_cols", "cat_vocab_sizes",
                    "vocab", "num_dim", "out_cols", "out_dim"]},
                  open(f"{run_dir}/struct_scaler.json", "w"), indent=2)
        log(f"struct_dense: {len(num_cols)} numerics"
            + (f" + {len(cat_cols)} cat embeds" if cat_cols else " (no cat embeds)"))

    tok = AutoTokenizer.from_pretrained(cfg["model"]["encoder"])
    model = MSFCBert(cfg, cfg["graph"]["relations"].keys(), len(labels), use_graph,
                     struct_meta=struct_meta).to(device)

    freq = train_df["label"].map(label2id).value_counts().sort_index()
    alpha = torch.tensor((len(train_df) / (len(labels) * freq)).values, dtype=torch.float32).to(device)
    # Class counts come from the augmented TRAIN partition only (invariant 4), which is
    # also the population the default focal alpha has always been computed over.
    counts = [int(freq.get(i, 0)) for i in range(len(labels))]
    crit = make_loss(args.loss, counts, alpha, cfg).to(device)
    log(f"loss = {args.loss} (train class counts: {counts})")

    cache = {"emb": None}
    if use_graph:
        # graph pool = FULL train partition (9-label corpus): neighbors contribute
        # entities/dates/text only — never labels — so both schemes share it.
        orig_train = df_full[df_full["eventid"].astype(str).isin(
            set(map(str, gctx.train_eventids)))]
        orig_train = orig_train.set_index(orig_train["eventid"].astype(str)) \
                               .loc[list(map(str, gctx.train_eventids))].reset_index(drop=True)
        cache_texts = [build_input_text(r, selected, not args.no_structured)
                       for _, r in orig_train.iterrows()]
        cache["emb"] = encode_train_cache(model, tok, cache_texts, cfg, device)
        if cfg["model"].get("mo_prior"):
            # 6a MO prior (D9): TRAIN-neighbor label histograms. The pool vocabulary is
            # the full 9-label corpus (like the neighbor pool itself, shared by schemes).
            pool_vocab = sorted(df_full["label"].unique())
            cache["pool_labels"] = orig_train["label"].map(
                {l: i for i, l in enumerate(pool_vocab)}).values.astype(np.int64)
            cache["n_pool_labels"] = len(pool_vocab)

    mk = lambda d, tr: EventDataset(d, tok, cfg, selected, gctx, label2id,
                                    not args.no_structured, use_graph, args.past_only, tr,
                                    struct_meta=struct_meta, num_meta=num_meta)
    coll = make_collate(tok, cfg, cache, list(cfg["graph"]["relations"]), use_graph)
    if args.balanced_sampler:
        w = (1.0 / freq)[train_df["label"].map(label2id).values].values
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(train_df), replacement=True)
        tl = DataLoader(mk(train_df, True), batch_size=cfg["train"]["batch_size"],
                        sampler=sampler, collate_fn=coll, num_workers=2)
    else:
        tl = DataLoader(mk(train_df, True), batch_size=cfg["train"]["batch_size"],
                        shuffle=True, collate_fn=coll, num_workers=2)
    vl = DataLoader(mk(val_df, False), batch_size=cfg["train"]["eval_batch_size"],
                    collate_fn=coll, num_workers=2)
    sl = DataLoader(mk(test_df, False), batch_size=cfg["train"]["eval_batch_size"],
                    collate_fn=coll, num_workers=2)

    epochs = args.epochs or cfg["train"]["epochs"]
    accum = max(1, cfg["train"].get("grad_accum", 1))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    steps = (len(tl) + accum - 1) // accum * epochs
    sch = get_linear_schedule_with_warmup(opt, int(steps * cfg["train"]["warmup_ratio"]), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["fp16"])

    oaf = bool(struct_meta and struct_meta.get("out_dim"))
    out_p = float(cfg["model"].get("outcome_drop_p", 0.5))
    best, bad, t0 = -1.0, 0, time.time()
    best_full = best_mask = None
    epoch_losses = []
    for ep in range(epochs):
        if hasattr(crit, "set_epoch"):
            crit.set_epoch(ep)          # LDAM deferred re-weighting schedule
        run_loss, nb = 0.0, 0
        opt.zero_grad()
        for i, b in enumerate(tl):
            with torch.amp.autocast("cuda", enabled=cfg["train"]["fp16"]):
                neigh = ({k: v.to(device) for k, v in b["neigh"].items()} if use_graph else None)
                mo = b["mo"].to(device) if "mo" in b else None
                snum = b["snum"].to(device) if "snum" in b else None
                scat = b["scat"].to(device) if "scat" in b else None
                sout = b["sout"].to(device) if "sout" in b else None
                # per-event availability m ~ Bernoulli(1 - outcome_drop_p), train only
                av = (None if sout is None else
                      (torch.rand(sout.shape[0], device=device) >= out_p).float())
                lab = b["labels"].to(device)
                if use_graph and cfg["model"].get("aux_graph_loss"):
                    # 6a deep supervision: the graph path must classify on its own,
                    # so gated fusion cannot silently collapse to the text path.
                    logits, aux = model(b["input_ids"].to(device),
                                        b["attention_mask"].to(device), neigh,
                                        mo=mo, snum=snum, scat=scat, return_aux=True,
                                        sout=sout, avail=av)
                    loss = crit(logits, lab) + 0.3 * crit(aux, lab)
                else:
                    loss = crit(model(b["input_ids"].to(device),
                                      b["attention_mask"].to(device), neigh,
                                      mo=mo, snum=snum, scat=scat, sout=sout, avail=av),
                                lab)
            run_loss += float(loss.detach()); nb += 1
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(tl):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad()
        epoch_losses.append(run_loss / max(1, nb))
        y, p, vpr, vids = evaluate_split(model, vl, device, use_graph, avail=1 if oaf else None)
        vf1 = f1_score(y, p, average="macro")
        sel = vf1
        if oaf:
            # epoch selection: mean of full- and masked-regime val macro-F1
            ym, pm, vprm, _ = evaluate_split(model, vl, device, use_graph, avail=0)
            vf1m = f1_score(ym, pm, average="macro")
            sel = (vf1 + vf1m) / 2
            log(f"epoch {ep}: train loss {epoch_losses[-1]:.4f} | val macro-F1 full {vf1:.4f} "
                f"masked {vf1m:.4f} select {sel:.4f}")
        else:
            log(f"epoch {ep}: train loss {epoch_losses[-1]:.4f} | val macro-F1 {vf1:.4f}")
        if sel > best:
            best, bad = sel, 0
            if oaf:
                best_full, best_mask = vf1, vf1m
                pd.DataFrame({"eventid": vids, "true": ym, "pred": pm,
                              **{f"p{i}": vprm[:, i] for i in range(vprm.shape[1])}}) \
                    .to_csv(f"{run_dir}/preds_val_masked.csv", index=False)
            torch.save(model.state_dict(), f"{run_dir}/best.pt")
            # Persist the selected epoch's val probabilities. Any post-hoc component
            # (calibration, per-class biases) fits on THIS file, so it can never be
            # accidentally fitted on test, and no re-inference pass is needed.
            pd.DataFrame({"eventid": vids, "true": y, "pred": p,
                          **{f"p{i}": vpr[:, i] for i in range(vpr.shape[1])}}) \
                .to_csv(f"{run_dir}/preds_val.csv", index=False)
        else:
            bad += 1
            if bad >= cfg["train"]["patience"]:
                log("early stop"); break
        if use_graph and cfg["train"]["neighbor_embed_refresh"] == "each_epoch" and ep < epochs - 1:
            cache["emb"] = encode_train_cache(model, tok, cache_texts, cfg, device)

    model.load_state_dict(torch.load(f"{run_dir}/best.pt"))
    y, p, pr, ids = evaluate_split(model, sl, device, use_graph, avail=1 if oaf else None)
    pd.DataFrame({"eventid": ids, "true": y, "pred": p,
                  **{f"p{i}": pr[:, i] for i in range(pr.shape[1])}}) \
        .to_csv(f"{run_dir}/preds_test.csv", index=False)
    extra = {}
    if oaf:
        ym, pm, prm, idsm = evaluate_split(model, sl, device, use_graph, avail=0)
        pd.DataFrame({"eventid": idsm, "true": ym, "pred": pm,
                      **{f"p{i}": prm[:, i] for i in range(prm.shape[1])}}) \
            .to_csv(f"{run_dir}/preds_test_masked.csv", index=False)
        # val_macro_f1 keeps its meaning (full regime) so every existing reader works;
        # the selection criterion and the masked regime are recorded beside it.
        extra = {"val_select_mean_full_masked": best, "val_macro_f1_masked": best_mask,
                 "test_macro_f1_masked": f1_score(ym, pm, average="macro"),
                 "test_accuracy_masked": float((ym == pm).mean()),
                 "report_masked": classification_report(ym, pm, target_names=labels,
                                                        output_dict=True, zero_division=0)}
        best = best_full
    rep = classification_report(y, p, target_names=labels, output_dict=True, zero_division=0)
    peak_vram = (torch.cuda.max_memory_allocated() / 2**30) if device == "cuda" else 0.0
    json.dump({"run": name, "labels": labels, "label_scheme": args.label_scheme,
               "args": {k: v for k, v in vars(args).items() if k != "config"},
               "epoch_losses": epoch_losses, "val_macro_f1": best,
               "test_macro_f1": f1_score(y, p, average="macro"),
               "test_accuracy_secondary_imbalance_caveat": float((y == p).mean()),
               "wall_clock_sec": round(time.time() - t0, 1),
               "peak_vram_gb": round(peak_vram, 2), "report": rep, **extra},
              open(f"{run_dir}/metrics.json.tmp", "w"), indent=2)
    os.replace(f"{run_dir}/metrics.json.tmp", f"{run_dir}/metrics.json")   # atomic: never truncated
    # MTL_BLIND_TEST=1: the test score is written to
    # metrics.json only, never to the log, until the one-time readout.
    tscore = ("test score not printed (MTL_BLIND_TEST)" if os.environ.get("MTL_BLIND_TEST") == "1"
              else f"test macro-F1 = {f1_score(y, p, average='macro'):.4f}")
    log(f"[{name}] {tscore} "
        f"| wall {time.time() - t0:.0f}s | peak VRAM {peak_vram:.2f} GB -> {run_dir}")


if __name__ == "__main__":
    main()
