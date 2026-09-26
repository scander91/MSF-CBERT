"""Baseline: an INDUCTIVE adaptation of BertGCN.

Published BertGCN/TextGCN are transductive — test documents sit inside the corpus graph
during training, which is exactly the leakage our protocol forbids. This adaptation keeps
their core idea (interpolate BERT logits with GCN-over-doc-word-graph logits,
z = λ·z_BERT + (1−λ)·z_GCN) while making it leakage-clean:

  * word vocabulary and doc–word TF-IDF edges are built from TRAIN docs only;
  * a 2-layer GCN runs on the train graph (train docs + train-vocab words);
  * a TEST doc is attached at inference: it receives one aggregation hop from the word
    nodes it shares with the train vocabulary and sends nothing back.

Requires ConfliBERT test logits from a prior no-graph run (--bert-run).
"""
import argparse, json, os
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from src.util import load_config, log, ensure_dir, read_manifest


def normalize_adj(A, col_deg=None):
    """Symmetric normalization of a RECTANGULAR doc x word biadjacency:
    D_doc^{-1/2} A D_word^{-1/2}. Row scaling is row-local; the WORD (column)
    degrees are a corpus statistic, so held-out docs must pass col_deg computed
    on TRAIN (invariant 2 — no statistic fit outside train)."""
    dr = np.asarray(A.sum(1)).ravel(); dr[dr == 0] = 1
    dc = np.asarray(A.sum(0)).ravel() if col_deg is None else col_deg.copy()
    dc[dc == 0] = 1
    return sp.diags(1.0 / np.sqrt(dr)) @ A @ sp.diags(1.0 / np.sqrt(dc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", required=True)
    ap.add_argument("--bert-run", required=True, help="run dir of the no-graph ConfliBERT model")
    ap.add_argument("--label-scheme", default="with_unknown",
                    choices=["with_unknown", "without_unknown"])
    ap.add_argument("--lam", type=float, default=0.7)
    a = ap.parse_args(); cfg = load_config(a.config)

    df = pd.read_parquet(cfg["paths"]["processed"])
    df = df[~df["label"].isin(cfg["data"]["label_schemes"][a.label_scheme])]   # D4
    man = read_manifest(cfg, a.split)
    labels = sorted(df["label"].unique()); label2id = {l: i for i, l in enumerate(labels)}
    tr = df.merge(man[man.split == "train"], on="eventid")
    te = df.merge(man[man.split == "test"], on="eventid")

    vec = TfidfVectorizer(max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform(tr["text"])                       # TRAIN-ONLY vocabulary
    Xte = vec.transform(te["text"])                           # test attaches via train vocab

    # 2-hop doc<-word<-doc propagation on the train graph; test docs receive-only.
    # Features are TF-IDF reduced by TRAIN-FIT SVD to 256-d before propagation: the
    # full word x word product (50k^2) exceeds this host's 62 GB RAM; propagating
    # reduced features is the same linear operator applied in a lower rank.
    from sklearn.decomposition import TruncatedSVD
    word_deg = np.asarray(Xtr.sum(0)).ravel()                 # TRAIN word degrees (frozen)
    An = normalize_adj(Xtr)                                   # doc x word (train)
    svd = TruncatedSVD(n_components=256, random_state=42).fit(Xtr)   # train-only fit
    F_tr = svd.transform(Xtr)                                 # docs x 256
    M = An.T @ F_tr                                           # words x 256, train-only
    H_tr = An @ M                                             # train docs after 2 hops
    H_te = normalize_adj(Xte, col_deg=word_deg) @ M           # test docs: one receive hop

    gcn = LogisticRegression(max_iter=2000, C=4.0, n_jobs=-1)
    gcn.fit(H_tr, tr["label"].map(label2id))
    z_gcn = gcn.predict_proba(H_te)

    bert = pd.read_csv(os.path.join(a.bert_run, "preds_test.csv"))
    bert = bert.set_index(bert["eventid"].astype(str)).loc[te["eventid"].astype(str)]
    z_bert = bert[[f"p{i}" for i in range(len(labels))]].values

    z = a.lam * z_bert + (1 - a.lam) * z_gcn
    pred = z.argmax(1); y = te["label"].map(label2id).values
    mf1 = f1_score(y, pred, average="macro")

    name = f"{a.split}_bertgcn_ind" + ("_nounk" if a.label_scheme == "without_unknown" else "")
    run = os.path.join(cfg["paths"]["runs_dir"], name); ensure_dir(run)
    pd.DataFrame({"eventid": te["eventid"], "true": y, "pred": pred}) \
        .to_csv(f"{run}/preds_test.csv", index=False)
    json.dump({"run": name, "lam": a.lam, "labels": labels, "label_scheme": a.label_scheme,
               "test_macro_f1": float(mf1)}, open(f"{run}/metrics.json", "w"), indent=2)
    log(f"[{name}] λ={a.lam} test macro-F1 = {mf1:.4f}")


if __name__ == "__main__":
    main()
