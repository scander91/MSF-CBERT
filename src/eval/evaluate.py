"""Step 07 — Aggregate runs into per-class tables with uncertainty.

  per-class table:  P/R/F1 mean ± SD across the five seeds + 95% bootstrap CI on F1
                    + test support n
  significance:     paired bootstrap on macro-F1 difference + McNemar between two runs

Usage
  python -m src.eval.evaluate --runs seed_13 seed_21 seed_42 seed_87 seed_100
  python -m src.eval.evaluate --compare seed_42 seed_42_nograph
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, f1_score
from scipy.stats import chi2
from src.util import load_config, log, ensure_dir


def per_class(y, p, n_classes):
    P, R, F, S = precision_recall_fscore_support(y, p, labels=range(n_classes), zero_division=0)
    return P, R, F, S


def bootstrap_ci_f1(y, p, cls, n_boot, ci, rng):
    stats = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        stats.append(f1_score(y[b], p[b], labels=[cls], average=None, zero_division=0)[0])
    lo, hi = np.percentile(stats, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return lo, hi


def aggregate(cfg, run_names):
    rows, labels = [], None
    for rn in run_names:
        m = json.load(open(os.path.join(cfg["paths"]["runs_dir"], rn, "metrics.json")))
        labels = m["labels"]
        d = pd.read_csv(os.path.join(cfg["paths"]["runs_dir"], rn, "preds_test.csv"))
        rows.append(d)
    C = len(labels)
    P = np.zeros((len(rows), C)); R = np.zeros_like(P); F = np.zeros_like(P)
    for i, d in enumerate(rows):
        P[i], R[i], F[i], S = per_class(d["true"].values, d["pred"].values, C)
    rng = np.random.default_rng(0)
    y, p = rows[-1]["true"].values, rows[-1]["pred"].values     # CI on a representative seed
    lines = ["| Class | n | P (mean±SD) | R (mean±SD) | F1 (mean±SD) | F1 95% CI (seed "
             f"{run_names[-1].split('_')[-1]}) |", "|---|---|---|---|---|---|"]
    for c, lab in enumerate(labels):
        lo, hi = bootstrap_ci_f1(y, p, c, cfg["eval"]["bootstrap_n"], cfg["eval"]["ci"], rng)
        lines.append(f"| {lab} | {S[c]} | {P[:,c].mean()*100:.2f}±{P[:,c].std()*100:.2f} "
                     f"| {R[:,c].mean()*100:.2f}±{R[:,c].std()*100:.2f} "
                     f"| {F[:,c].mean()*100:.2f}±{F[:,c].std()*100:.2f} "
                     f"| [{lo*100:.2f}, {hi*100:.2f}] |")
    macro = F.mean(1)
    lines.append(f"\n**Macro-F1 across seeds: {macro.mean()*100:.2f} ± {macro.std()*100:.2f}** "
                 f"(runs: {', '.join(run_names)})")
    return "\n".join(lines)


def compare(cfg, run_a, run_b, n_boot=2000):
    da = pd.read_csv(os.path.join(cfg["paths"]["runs_dir"], run_a, "preds_test.csv"))
    db = pd.read_csv(os.path.join(cfg["paths"]["runs_dir"], run_b, "preds_test.csv"))
    m = da.merge(db, on="eventid", suffixes=("_a", "_b"))
    assert (m["true_a"] == m["true_b"]).all(), "runs are not on the same test set"
    y = m["true_a"].values; pa, pb = m["pred_a"].values, m["pred_b"].values
    # paired bootstrap on macro-F1 difference
    rng = np.random.default_rng(0); idx = np.arange(len(y)); diffs = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        diffs.append(f1_score(y[b], pa[b], average="macro")
                     - f1_score(y[b], pb[b], average="macro"))
    diffs = np.array(diffs)
    pval_boot = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    # McNemar with continuity correction on correctness
    ca, cb = (pa == y), (pb == y)
    n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) else 0.0
    pval_mc = float(chi2.sf(stat, 1))
    return {"run_a": run_a, "run_b": run_b,
            "macro_f1_a": f1_score(y, pa, average="macro"),
            "macro_f1_b": f1_score(y, pb, average="macro"),
            "diff_mean": float(diffs.mean()), "p_paired_bootstrap": pval_boot,
            "mcnemar_n01": n01, "mcnemar_n10": n10, "p_mcnemar": pval_mc}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--runs", nargs="+", help="aggregate these runs into the ±SD table")
    ap.add_argument("--compare", nargs=2, help="significance test between two runs")
    a = ap.parse_args(); cfg = load_config(a.config)
    ensure_dir(cfg["paths"]["tables_dir"])
    if a.runs:
        md = aggregate(cfg, a.runs)
        out = os.path.join(cfg["paths"]["tables_dir"], "per_class_uncertainty.md")
        open(out, "w").write(md); log(f"wrote {out}\n\n{md}")
    if a.compare:
        res = compare(cfg, *a.compare)
        log(json.dumps(res, indent=2))
