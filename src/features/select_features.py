"""Step 03 — Structured-attribute selection via Cramér's V, TRAIN PARTITION ONLY.

The selected feature list is frozen into artifacts/features/selected_{split}.json together
with a fingerprint = sha256(sorted train eventids). The audit recomputes the fingerprint
from the manifest and fails if they differ — i.e., it is impossible to silently fit
feature selection on val/test and pass the gate.
"""
import argparse, hashlib, json
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency
from src.util import load_config, log, ensure_dir, read_manifest, train_fingerprint


def cramers_v(x, y):
    """Bias-corrected Cramér's V (Bergsma 2013)."""
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum(); r, c = ct.shape
    phi2 = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1); cc = c - (c - 1) ** 2 / (n - 1)
    denom = min(rc - 1, cc - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def main(cfg, split_name):
    p, f = cfg["paths"], cfg["features"]
    df = pd.read_parquet(p["processed"])
    man = read_manifest(cfg, split_name)
    train_ids = man.loc[man.split == "train", "eventid"]
    tr = df[df["eventid"].isin(set(train_ids))]          # <- the whole point: train only

    scores = {c: cramers_v(tr[c].fillna("NA"), tr["label"]) for c in f["candidates"]}
    selected = [c for c, v in scores.items() if v >= f["cramers_v_min"]]
    out = {"split": split_name,
           "scores": {k: round(v, 4) for k, v in scores.items()},
           "selected": selected,
           "train_fingerprint": train_fingerprint(train_ids)}
    ensure_dir(p["features_dir"])
    json.dump(out, open(f"{p['features_dir']}/selected_{split_name}.json", "w"), indent=2)
    log(f"[{split_name}] Cramér's V (train-only): {out['scores']} -> selected {selected}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--split", required=True)
    a = ap.parse_args(); main(load_config(a.config), a.split)
