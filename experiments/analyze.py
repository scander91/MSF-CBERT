"""Statistics for the additional experiments (protocol in experiments/README.md, fixed before test scores were read).

Nadeau-Bengio corrected resampled t (two-sided, n_test/n_train = 0.15/0.70), Holm within
each family F1-F4, Cohen's d_z, seeds favouring each side. Everything outside F1-F4 is
descriptive (same statistics, unadjusted). Original-vs-group-disjoint change: difference
of ten-seed means, no test. Writes supplementary/results/processed/*.csv and
supplementary/results/statistical_tests/*.json. Requires the run directories under runs/.
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PUB = ROOT / "runs"  # published ten-seed runs: seed_{s}_structdense_nograph, seed_{s}_structdense_a100, seed_{s}_nograph, seed_{s}, seed_{s}_bertgcn_ind
OUT = ROOT / "supplementary" / "results"
SEEDS = [13, 21, 42, 87, 100, 7, 33, 55, 76, 91]
RATIO = 0.15 / 0.70
ACC = "test_accuracy_secondary_imbalance_caveat"


def metrics(path):
    m = json.loads((path / "metrics.json").read_text())
    f1 = 100 * m["test_macro_f1"]
    acc = m.get(ACC, m.get("test_accuracy"))
    if acc is None and (path / "preds_test.csv").exists():
        p = pd.read_csv(path / "preds_test.csv")
        acc = float((p["true"] == p["pred"]).mean())
    val = m.get("val_macro_f1")
    return f1, (100 * acc if acc is not None else np.nan), (100 * val if val is not None else np.nan)


def new(split, arm):
    return RUNS / f"{split}__{arm}"


def pub(name):
    return PUB / name


# arm -> function(seed) -> run dir
S = {a: (lambda a: lambda s: new(f"seed_{s}", a))(a) for a in
     ["gated", "text_cat", "text_only", "concat_direct", "concat_dense", "mlp_matched", "fixed_avg",
      "no_outcome", "vicinity_only", "text_num", "restricted", "restricted_first", "text_first",
      "no_weapon", "no_perp", "no_coded", "no_extended", "no_property", "no_casualty", "no_success",
      "graph_nocat", "struct_xgb_published", "struct_xgb_cat_parity", "struct_xgb_cat_no_outcome",
      "struct_xgb_cat_restricted", "struct_mlp_parity", "struct_mlp_no_outcome", "struct_mlp_restricted"]}
S.update({"pub_gated": lambda s: pub(f"seed_{s}_structdense_nograph"),
          "pub_gated_graph": lambda s: pub(f"seed_{s}_structdense_a100"),
          "pub_text_cat": lambda s: pub(f"seed_{s}_nograph"),
          "pub_text_graph": lambda s: pub(f"seed_{s}"),
          "pub_bertgcn": lambda s: pub(f"seed_{s}_bertgcn_ind")})
G = {a: (lambda a: lambda s: new(f"groupseed_{s}", a))(a) for a in
     ["gated", "text_cat", "text_only", "restricted", "concat_direct", "mlp_matched", "no_outcome",
      "graph_gated", "struct_xgb_cat_parity", "struct_xgb_cat_restricted", "struct_mlp_parity",
      "struct_mlp_restricted"]}
G["bertgcn"] = lambda s: RUNS / f"groupseed_{s}_bertgcn_ind"


def table(arms, part):
    rows = []
    for arm, f in arms.items():
        for s in SEEDS:
            d = f(s)
            if (d / "metrics.json").exists():
                f1, acc, val = metrics(d)
                rows.append(dict(partition=part, arm=arm, seed=s, test_macro_f1=f1, test_acc=acc, val_macro_f1=val))
    return pd.DataFrame(rows)


def corrected(a, b, df):
    x = df[df.arm == a].set_index("seed").test_macro_f1
    y = df[df.arm == b].set_index("seed").test_macro_f1
    common = [s for s in SEEDS if s in x.index and s in y.index]
    d = np.array([x[s] - y[s] for s in common]); J = len(d)
    if J < 2:
        return None
    m = d.mean(); sd = d.std(ddof=1)
    se = math.sqrt((1 / J + RATIO) * sd ** 2)
    t = m / se; p = 2 * stats.t.sf(abs(t), J - 1); h = stats.t.ppf(0.975, J - 1) * se
    return dict(a=a, b=b, n=J, mean_diff=m, ci_low=m - h, ci_high=m + h, t=t, p=p,
                d_z=m / sd if sd > 0 else np.nan, seeds_a_higher=int((d > 0).sum()),
                seeds_b_higher=int((d < 0).sum()))


def holm(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"]); prev = 0.0
    for k, i in enumerate(order):
        adj = min(1.0, max(prev, (len(rows) - k) * rows[i]["p"])); rows[i]["p_holm"] = adj; prev = adj
    return rows


def main():
    (OUT / "processed").mkdir(parents=True, exist_ok=True)
    (OUT / "statistical_tests").mkdir(parents=True, exist_ok=True)
    dS, dG = table(S, "S"), table(G, "G")
    chrono = []
    for arm in ["text_only", "no_outcome", "restricted", "restricted_first"]:
        d = new("chrono", arm)
        if (d / "metrics.json").exists():
            f1, acc, val = metrics(d); chrono.append(dict(partition="chrono", arm=arm, seed=0, test_macro_f1=f1, test_acc=acc, val_macro_f1=val))
    for name in ["chrono_structdense_nograph", "chrono_nograph"]:
        if (pub(name) / "metrics.json").exists():
            f1, acc, val = metrics(pub(name))
            chrono.append(dict(partition="chrono", arm="pub_" + name, seed=0, test_macro_f1=f1, test_acc=acc, val_macro_f1=val))
    allruns = pd.concat([dS, dG, pd.DataFrame(chrono)], ignore_index=True)
    allruns.to_csv(OUT / "processed" / "per_run_metrics.csv", index=False)
    summ = (allruns.groupby(["partition", "arm"])
            .agg(n=("seed", "size"), test_f1_mean=("test_macro_f1", "mean"), test_f1_sd=("test_macro_f1", "std"),
                 test_acc_mean=("test_acc", "mean"), test_acc_sd=("test_acc", "std"),
                 val_f1_mean=("val_macro_f1", "mean"), val_f1_sd=("val_macro_f1", "std"))
            .reset_index())
    summ.to_csv(OUT / "processed" / "arm_summary.csv", index=False)

    fam = {
        "F1_gating_S": (dS, [("gated", "concat_direct"), ("gated", "mlp_matched"), ("gated", "fixed_avg")]),
        "F2_features_S": (dS, [("gated", "no_outcome"), ("gated", "text_cat"), ("gated", "text_num"), ("gated", "text_only")]),
        "F3_group_disjoint_G": (dG, [("gated", "bertgcn"), ("gated", "text_cat"), ("gated", "text_only")]),
        "F4_graph": (None, [("G", "graph_gated", "gated"), ("S", "graph_nocat", "text_only")]),
    }
    res = {}
    for k, (df, pairs) in fam.items():
        rows = []
        for pr in pairs:
            if df is None:
                part, a, b = pr; r = corrected(a, b, dS if part == "S" else dG)
                if r: r["partition"] = part
            else:
                r = corrected(*pr, df)
            if r: rows.append(r)
        res[k] = holm(rows)
    desc_S = [("gated", a) for a in ["concat_dense", "vicinity_only", "restricted", "restricted_first", "text_first",
                                       "no_weapon", "no_perp", "no_coded", "no_extended", "no_property",
                                       "no_casualty", "no_success", "struct_xgb_published", "struct_xgb_cat_parity",
                                       "struct_mlp_parity"]] + \
             [("restricted", "struct_xgb_cat_restricted"), ("restricted", "struct_mlp_restricted"),
              ("no_outcome", "struct_xgb_cat_no_outcome"), ("restricted", "restricted_first"),
              ("gated", "pub_gated"), ("text_cat", "pub_text_cat"),
              ("pub_gated", "pub_bertgcn"), ("pub_gated", "pub_text_graph"), ("pub_gated", "pub_text_cat"),
              ("pub_gated_graph", "pub_gated")]
    desc_G = [("gated", a) for a in ["concat_direct", "mlp_matched", "no_outcome", "restricted",
                                       "struct_xgb_cat_parity", "struct_mlp_parity"]] + \
             [("restricted", "struct_xgb_cat_restricted"), ("restricted", "struct_mlp_restricted")]
    res["descriptive_unadjusted_S"] = [r for r in (corrected(a, b, dS) for a, b in desc_S) if r]
    res["descriptive_unadjusted_G"] = [r for r in (corrected(a, b, dG) for a, b in desc_G) if r]

    shift = []
    for arm in ["gated", "text_cat", "text_only", "restricted", "concat_direct", "mlp_matched", "no_outcome"]:
        a = dS[dS.arm == arm].test_macro_f1; b = dG[dG.arm == arm].test_macro_f1
        if len(a) and len(b):
            shift.append(dict(arm=arm, S_mean=a.mean(), S_sd=a.std(ddof=1), n_S=len(a), G_mean=b.mean(),
                              G_sd=b.std(ddof=1), n_G=len(b), G_minus_S=b.mean() - a.mean()))
    res["S_vs_G_descriptive_no_test"] = shift
    (OUT / "statistical_tests" / "contrasts.json").write_text(json.dumps(res, indent=2))
    flat = [dict(family=k, **r) for k, v in res.items() if k != "S_vs_G_descriptive_no_test" for r in v]
    pd.DataFrame(flat).to_csv(OUT / "statistical_tests" / "contrasts.csv", index=False)
    pd.DataFrame(shift).to_csv(OUT / "statistical_tests" / "original_vs_group_disjoint.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(2).to_string(index=False))
    for k, v in res.items():
        print("\n==", k)
        print(pd.DataFrame(v).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
