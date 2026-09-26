"""Dump predictions + probabilities for the VAL (or test) partition of a FINISHED run.

train.py only writes preds_test.csv; post-hoc val-fit calibration (Stage 6) needs val
probabilities. Reuses train.py's dataset/collate/eval machinery so the inference path
is identical. The graph neighbor cache is rebuilt with the run's best.pt encoder.

  python scripts/infer_partition.py --run runs/seed_42 --partition val
  # also dump the pre-classifier vectors; train = ORIGINAL
  # train events only (no paraphrases), each excluded from its own graph neighbourhood
  python scripts/infer_partition.py --config configs/structdense.yaml --run runs/seed_42_structdense \
      --partition train --features
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
from src.util import load_config, log, read_manifest
from src.graph.build_graph import GraphContext
from src.models.msf_cbert import MSFCBert
from src.features.verbalize import build_input_text
from src.train.train import EventDataset, make_collate, encode_train_cache, evaluate_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml",
                    help="must be the SAME config the run was trained with")
    ap.add_argument("--run", required=True)
    ap.add_argument("--partition", default="val", choices=["train", "val", "test"])
    ap.add_argument("--features", action="store_true",
                    help="also write features_<partition>.npz (eventid, true, probs, feats)")
    ap.add_argument("--out-suffix", default="",
                    help="suffix for the output files (e.g. _reinfer) so recorded files stay untouched")
    a = ap.parse_args()
    cfg = load_config(a.config)
    meta = json.load(open(os.path.join(a.run, "metrics.json")))
    targs, scheme = meta["args"], meta["label_scheme"]
    split = targs["split"]
    use_graph, use_struct = not targs["no_graph"], not targs["no_structured"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df_full = pd.read_parquet(cfg["paths"]["processed"])
    df = df_full[~df_full["label"].isin(cfg["data"]["label_schemes"][scheme])]
    man = read_manifest(cfg, split)
    selected = json.load(open(f"{cfg['paths']['features_dir']}/selected_{split}.json"))["selected"]
    labels = meta["labels"]; label2id = {l: i for i, l in enumerate(labels)}
    part_df = df.merge(man[man.split == a.partition], on="eventid")

    # D13 runs carry a dense structured branch. It is a config-only lever, so
    # metrics.json["args"] does not record it — the run's own struct_scaler.json is the
    # detection signal. Without this the state_dict load fails on cat_emb/struct_* keys.
    struct_meta = None
    scaler_path = os.path.join(a.run, "struct_scaler.json")
    if os.path.exists(scaler_path):
        struct_meta = json.load(open(scaler_path))
        struct_meta.setdefault("num_dim", len(struct_meta["num_cols"]))
        if struct_meta.get("cat_cols") and "vocab" not in struct_meta:
            # Older runs did not persist the vocab; re-derive it exactly as train.py
            # does (sorted unique train values, index 0 = UNK). Train partition only.
            tr_orig = df.merge(man[man.split == "train"], on="eventid")
            struct_meta["vocab"] = {
                c: {v: i + 1 for i, v in enumerate(
                    sorted(x for x in tr_orig[c].dropna().astype(str).unique()
                           if x not in ("", "Unknown")))}
                for c in struct_meta["cat_cols"]}
        struct_meta.setdefault("vocab", {})
        log(f"struct_dense run detected: {struct_meta['num_dim']} numerics, "
            f"{len(struct_meta.get('cat_cols') or [])} cat embeds")

    gctx = GraphContext(f"{cfg['paths']['graph_dir']}/graph_{split}.pkl") if use_graph else None
    tok = AutoTokenizer.from_pretrained(cfg["model"]["encoder"])
    model = MSFCBert(cfg, cfg["graph"]["relations"].keys(), len(labels), use_graph,
                     struct_meta=struct_meta).to(device)
    model.load_state_dict(torch.load(os.path.join(a.run, "best.pt"), map_location=device))
    model.eval()

    cache = {"emb": None}
    if use_graph:
        ids = list(map(str, gctx.train_eventids))
        orig = df_full[df_full["eventid"].astype(str).isin(set(ids))]
        orig = orig.set_index(orig["eventid"].astype(str)).loc[ids].reset_index(drop=True)
        texts = [build_input_text(r, selected, use_struct) for _, r in orig.iterrows()]
        cache["emb"] = encode_train_cache(model, tok, texts, cfg, device)
        model.eval()

    # is_train=True for the train partition: an original train event must not be its own
    # graph neighbour, exactly as during training.
    ds = EventDataset(part_df, tok, cfg, selected, gctx, label2id,
                      use_struct, use_graph, targs.get("past_only", False),
                      a.partition == "train", struct_meta=struct_meta)
    dl = DataLoader(ds, batch_size=cfg["train"]["eval_batch_size"],
                    collate_fn=make_collate(tok, cfg, cache, list(cfg["graph"]["relations"]),
                                            use_graph), num_workers=2)
    out = os.path.join(a.run, f"preds_{a.partition}{a.out_suffix}.csv")
    # recorded (unsuffixed) test predictions are never overwritten; a suffixed re-inference may be
    # regenerated until its completion marker (the npz) exists — an interrupted run can retry
    assert not (os.path.exists(out) and a.partition == "test" and not a.out_suffix), \
        f"{out} exists: recorded test predictions are never overwritten (use --out-suffix)"
    if a.features and a.out_suffix:
        assert not os.path.exists(os.path.join(a.run, f"features_{a.partition}{a.out_suffix}.npz")), \
            "features already complete for this partition"
    feats = [] if a.features else None
    y, p, pr, ids = evaluate_split(model, dl, device, use_graph, feats=feats)
    pd.DataFrame({"eventid": ids, "true": y, "pred": p,
                  **{f"p{i}": pr[:, i] for i in range(pr.shape[1])}}).to_csv(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)   # atomic: a crash never leaves a truncated predictions file
    if a.features:   # written last and atomically: the npz is the queue's completion marker
        fp = os.path.join(a.run, f"features_{a.partition}{a.out_suffix}.npz")
        np.savez_compressed(fp + ".tmp.npz", eventid=np.array(ids), true=y,
                            probs=pr.astype(np.float32),
                            feats=np.concatenate(feats).astype(np.float32), labels=np.array(labels))
        os.replace(fp + ".tmp.npz", fp)
        log(f"features -> {fp}")
    # never print a test score here.
    score = ("(test score not printed)" if a.partition == "test"
             else f"macro-F1 = {f1_score(y, p, average='macro'):.4f}")
    log(f"[{a.run}] {a.partition} {score} -> {out}")


if __name__ == "__main__":
    main()
