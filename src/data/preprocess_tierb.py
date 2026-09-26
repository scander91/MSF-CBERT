"""Step 03b — TIER-B preprocessing, fit on the TRAIN partition of one split (D8, A7).

Everything here is distribution-dependent and therefore per-split:
  * median imputation values for the numeric columns used by the structured baseline
  * mode imputation for categoricals (normally a no-op: Tier A already fills "Unknown")
  * discretization bins: none retained — XGBoost consumes raw numerics (recorded)
  * redundancy elimination on the STRUCTURED (XGBoost) feature set ONLY:
      - numeric pairs with |PCC| > pcc_max -> drop the later column
      - categorical pairs with NMI > nmi_max -> drop the later column
      - iterative VIF > vif_max on numerics -> drop worst
    PROTECTED columns (config preprocess.protected_cols) are never dropped: they feed
    the graph relations and verbalization, which are audited separately.

Artifact: artifacts/preprocess/tierb_{split}.json
  carries train_fingerprint (= sha256 of sorted train ids) and max_train_date; the
  audit (A7) recomputes both from the manifest and fails on mismatch — so it is
  impossible for these statistics to have been fit outside train and still pass.

The MSF-CBERT text model itself consumes NO Tier-B statistic (text + verbalized
categoricals + graph only); Tier B exists for the structured/XGBoost baseline.
"""
import argparse, json
import numpy as np, pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from src.util import load_config, log, ensure_dir, read_manifest, train_fingerprint

EXTRA_CATS = ["weapsubtype1_txt"]        # Tier-A-filled, XGB-only (not a graph relation)


def vif_prune(X, vif_max):
    """Iteratively drop the numeric column with the highest VIF > vif_max.
    X: (n, k) imputed, float. Returns kept column indices + per-round log."""
    keep = list(range(X.shape[1]))
    dropped = []
    while len(keep) > 1:
        vifs = []
        Z = X[:, keep]
        Zc = Z - Z.mean(0)
        for j in range(Z.shape[1]):
            y = Zc[:, j]
            A = np.delete(Zc, j, axis=1)
            denom = (y ** 2).sum()
            if denom < 1e-12:
                vifs.append(np.inf)      # zero-variance column: drop it
                continue
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            r2 = 1 - ((y - A @ coef) ** 2).sum() / denom
            vifs.append(1.0 / max(1e-12, 1 - r2))
        worst = int(np.argmax(vifs))
        if vifs[worst] <= vif_max:
            break
        dropped.append((keep[worst], float(vifs[worst])))
        keep.pop(worst)
    return keep, dropped


def main(cfg, split_name):
    p, d, pp = cfg["paths"], cfg["data"], cfg["preprocess"]
    df = pd.read_parquet(p["processed"])
    man = read_manifest(cfg, split_name)
    train_ids = man.loc[man.split == "train", "eventid"]
    tr = df[df["eventid"].isin(set(train_ids))]
    protected = set(pp["protected_cols"])

    num_cols = list(d["numeric_cols"])
    cat_cols = list(d["structured_cols"]) + EXTRA_CATS

    art = {"split": split_name,
           "train_fingerprint": train_fingerprint(train_ids),
           "max_train_date": str(tr["date"].max().date()),
           "n_train": int(len(tr)),
           "bins": {},                       # none retained: XGBoost consumes raw numerics
           "dropped": {"pcc": [], "nmi": [], "vif": []}}

    # ---- imputation values (train-only) ----
    art["numeric_medians"] = {c: (None if tr[c].dropna().empty else float(tr[c].median()))
                              for c in num_cols}
    art["categorical_modes"] = {c: (tr[c].mode().iloc[0] if not tr[c].mode().empty else "Unknown")
                                for c in cat_cols}

    # ---- redundancy elimination (XGB set only, train-only) ----
    kept_num = list(num_cols)
    corr = tr[kept_num].corr().abs()
    for i, a in enumerate(kept_num):
        for b in kept_num[i + 1:]:
            if a in kept_num and b in kept_num and corr.loc[a, b] > pp["pcc_max"]:
                victim = b if b not in protected else (a if a not in protected else None)
                if victim:
                    kept_num.remove(victim)
                    art["dropped"]["pcc"].append(
                        {"kept": a if victim == b else b, "dropped": victim,
                         "pcc": round(float(corr.loc[a, b]), 4)})

    kept_cat = list(cat_cols)
    codes = {c: tr[c].astype("category").cat.codes.values for c in cat_cols}
    for i, a in enumerate(cat_cols):
        for b in cat_cols[i + 1:]:
            if a in kept_cat and b in kept_cat:
                nmi = normalized_mutual_info_score(codes[a], codes[b])
                if nmi > pp["nmi_max"]:
                    victim = b if b not in protected else (a if a not in protected else None)
                    if victim:
                        kept_cat.remove(victim)
                        art["dropped"]["nmi"].append(
                            {"kept": a if victim == b else b, "dropped": victim,
                             "nmi": round(float(nmi), 4)})

    med = art["numeric_medians"]
    X = tr[kept_num].apply(lambda s: s.fillna(med[s.name])).values.astype(float)
    keep_idx, vif_dropped = vif_prune(X, pp["vif_max"])
    for j, v in vif_dropped:
        col = kept_num[j]
        if col not in protected:
            art["dropped"]["vif"].append({"dropped": col, "vif": round(v, 2)})
    vif_names = {kept_num[j] for j, _ in vif_dropped if kept_num[j] not in protected}
    kept_num = [c for c in kept_num if c not in vif_names]

    art["kept_numeric"] = kept_num
    art["kept_categorical"] = kept_cat

    ensure_dir(p["preprocess_dir"])
    out = f"{p['preprocess_dir']}/tierb_{split_name}.json"
    json.dump(art, open(out, "w"), indent=2)
    log(f"[{split_name}] Tier-B fit on {len(tr):,} train rows | kept numeric {kept_num} | "
        f"kept categorical {kept_cat}")
    for kind, drops in art["dropped"].items():
        if drops:
            log(f"[{split_name}]   dropped ({kind}): {drops}")
    log(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--split", required=True)
    a = ap.parse_args(); main(load_config(a.config), a.split)
