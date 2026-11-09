"""Secondary analyses from saved predictions (no retraining).

(a) Per-class accuracy of S-trained models on test events WITH vs WITHOUT a member of their
    group (primary definition g_90) in the training partition. Per class only; no subset macro-F1 claim.
(b) Group-balanced macro-F1: each test event weighted by 1 / (size of its group within the test set),
    so every test group has total weight 1. S and G; gated, text_cat, text_only (+ published gated).
(c) struct_xgb_cat_restricted: cause of the large accuracy SD across seeds (validation-selected settings).
(d) Group exposure of test events under each grouping definition (related+exact, 0.95, 0.90, 0.85, 0.80)
    for the original stratified splits (S) and the group-disjoint splits (G).
Also: descriptive text_cat - text_only on S (reference for the graph-redundancy contrast).
Outputs: supplementary/results/secondary/*.csv and secondary.json. Descriptive only; nothing is selected on test.
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PUB = ROOT / "runs"
PUB_SPLITS = ROOT / "artifacts" / "splits"
OUT = ROOT / "supplementary" / "results" / "secondary"
SEEDS = [13, 21, 42, 87, 100, 7, 33, 55, 76, 91]
GROUP_DEFS = ["g_rel_exact", "g_95", "g_90", "g_85", "g_80"]
PRIMARY = "g_90"
RATIO = 0.15 / 0.70


def split_manifest(part, s):
    f = PUB_SPLITS / f"seed_{s}.csv" if part == "S" else ROOT / "artifacts" / "splits" / f"groupseed_{s}.csv"
    return pd.read_csv(f, dtype={"eventid": str})


def run_dir(part, arm, s):
    if arm == "pub_gated":
        return PUB / f"seed_{s}_structdense_nograph"
    pre = "seed" if part == "S" else "groupseed"
    return RUNS / f"{pre}_{s}__{arm}"


def preds(d):
    return pd.read_csv(d / "preds_test.csv", dtype={"eventid": str})


def labels():
    return json.loads((RUNS / "seed_42__gated" / "metrics.json").read_text())["labels"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups = pd.read_parquet(ROOT / "artifacts" / "groups" / "groups.parquet")
    groups["eventid"] = groups.eventid.astype(str)
    lab = labels()
    res = {}

    # (d) exposure under each grouping definition
    rows = []
    for part in ["S", "G"]:
        for s in SEEDS:
            m = split_manifest(part, s).merge(groups, on="eventid")
            tr, te = m[m.split == "train"], m[m.split == "test"]
            r = dict(partition=part, seed=s, n_test=len(te))
            for g in GROUP_DEFS:
                r[f"exposed_pct_{g}"] = 100 * te[g].isin(set(tr[g])).mean()
            rows.append(r)
    exp_df = pd.DataFrame(rows)
    exp_df.to_csv(OUT / "group_exposure_by_definition.csv", index=False)
    summ = exp_df.groupby("partition")[[f"exposed_pct_{g}" for g in GROUP_DEFS]].agg(["min", "max", "mean"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(OUT / "group_exposure_summary.csv", index=False)
    res["exposure_summary"] = summ.round(3).reset_index().to_dict(orient="records")

    # (a) per-class accuracy exposed vs unexposed (S-trained models, primary grouping)
    rows = []
    for arm in ["gated", "pub_gated", "text_cat", "text_only"]:
        for s in SEEDS:
            m = split_manifest("S", s).merge(groups[["eventid", PRIMARY]], on="eventid")
            trg = set(m[m.split == "train"][PRIMARY])
            p = preds(run_dir("S", arm, s)).merge(m[m.split == "test"][["eventid", PRIMARY]], on="eventid", how="left")
            assert p[PRIMARY].notna().all(), (arm, s)
            p["exposed"] = p[PRIMARY].isin(trg)
            p["correct"] = p["true"] == p["pred"]
            for (c, e), x in p.groupby(["true", "exposed"]):
                rows.append(dict(arm=arm, seed=s, cls=lab[int(c)], exposed=bool(e), n=len(x), correct=int(x.correct.sum())))
    pc = pd.DataFrame(rows)
    pc.to_csv(OUT / "perclass_exposed_unexposed_by_seed.csv", index=False)
    pooled = pc.groupby(["arm", "cls", "exposed"])[["n", "correct"]].sum().reset_index()
    pooled["acc_pct"] = 100 * pooled.correct / pooled.n
    wide = pooled.pivot_table(index=["arm", "cls"], columns="exposed", values=["n", "acc_pct"]).reset_index()
    wide.columns = ["arm", "cls", "acc_unexposed", "acc_exposed", "n_unexposed_pooled", "n_exposed_pooled"]
    wide["exposed_minus_unexposed"] = wide.acc_exposed - wide.acc_unexposed
    tot = pc.groupby(["arm", "exposed"])[["n", "correct"]].sum().reset_index()
    tot["acc_pct"] = 100 * tot.correct / tot.n
    wide.to_csv(OUT / "perclass_exposed_unexposed_pooled.csv", index=False)
    tot.to_csv(OUT / "overall_accuracy_exposed_unexposed_pooled.csv", index=False)
    res["exposed_share_of_test_by_class_gated"] = (
        pooled[pooled.arm == "gated"].groupby("cls").apply(lambda x: 100 * x[x.exposed].n.sum() / x.n.sum(), include_groups=False).round(2).to_dict())
    res["overall_accuracy_exposed_unexposed"] = tot.round(3).to_dict(orient="records")

    # (b) group-balanced macro-F1
    rows = []
    for part, arms in [("S", ["gated", "pub_gated", "text_cat", "text_only"]), ("G", ["gated", "text_cat", "text_only"])]:
        for arm in arms:
            for s in SEEDS:
                m = split_manifest(part, s).merge(groups[["eventid", PRIMARY]], on="eventid")
                te = m[m.split == "test"].copy()
                te["w"] = 1.0 / te.groupby(PRIMARY)[PRIMARY].transform("size")
                p = preds(run_dir(part, arm, s)).merge(te[["eventid", "w"]], on="eventid", how="left")
                assert p.w.notna().all()
                rows.append(dict(partition=part, arm=arm, seed=s,
                                 macro_f1=100 * f1_score(p["true"], p["pred"], average="macro"),
                                 group_balanced_macro_f1=100 * f1_score(p["true"], p["pred"], average="macro", sample_weight=p.w),
                                 n_test=len(p), n_test_groups=float(p.w.sum())))
    gb = pd.DataFrame(rows)
    gb.to_csv(OUT / "group_balanced_macro_f1_by_seed.csv", index=False)
    gbs = gb.groupby(["partition", "arm"]).agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                                               gb_macro_f1_mean=("group_balanced_macro_f1", "mean"),
                                               gb_macro_f1_sd=("group_balanced_macro_f1", "std"),
                                               n_test_groups_mean=("n_test_groups", "mean")).reset_index()
    gbs["gb_minus_unweighted"] = gbs.gb_macro_f1_mean - gbs.macro_f1_mean
    gbs.to_csv(OUT / "group_balanced_macro_f1_summary.csv", index=False)
    res["group_balanced_summary"] = gbs.round(3).to_dict(orient="records")

    # (c) struct_xgb_cat_restricted diagnosis
    rows = []
    for part in ["S", "G"]:
        for s in SEEDS:
            d = run_dir(part, "struct_xgb_cat_restricted", s)
            mm = json.loads((d / "metrics.json").read_text())
            p = preds(d)
            share = p.pred.value_counts(normalize=True)
            rows.append(dict(partition=part, seed=s, class_balanced=mm["selected"]["class_balanced"],
                             max_depth=mm["selected"]["max_depth"], val_macro_f1=100 * mm["val_macro_f1"],
                             test_macro_f1=100 * mm["test_macro_f1"],
                             test_acc=100 * mm["test_accuracy_secondary_imbalance_caveat"],
                             n_pred_classes=int(p.pred.nunique()),
                             pred_share_bombing=100 * share.get(lab.index("Bombing/Explosion"), 0.0)))
    xr = pd.DataFrame(rows)
    xr.to_csv(OUT / "struct_xgb_cat_restricted_diagnosis.csv", index=False)
    res["xgb_restricted_by_class_balanced"] = (xr.groupby(["partition", "class_balanced"])
                                               .agg(n=("seed", "size"), acc_mean=("test_acc", "mean"),
                                                    f1_mean=("test_macro_f1", "mean"), val_f1_mean=("val_macro_f1", "mean"),
                                                    bombing_pred_share=("pred_share_bombing", "mean"))
                                               .round(3).reset_index().to_dict(orient="records"))

    # descriptive text_cat - text_only on S (unadjusted)
    pr = pd.read_csv(ROOT / "supplementary" / "results" / "processed" / "per_run_metrics.csv")
    x = pr[(pr.partition == "S") & (pr.arm == "text_cat")].set_index("seed").test_macro_f1
    y = pr[(pr.partition == "S") & (pr.arm == "text_only")].set_index("seed").test_macro_f1
    dd = np.array([x[s] - y[s] for s in SEEDS]); J = len(dd); sd = dd.std(ddof=1)
    se = math.sqrt((1 / J + RATIO) * sd ** 2); t = dd.mean() / se; h = stats.t.ppf(0.975, J - 1) * se
    res["descriptive_S_text_cat_minus_text_only"] = dict(mean_diff=dd.mean(), ci_low=dd.mean() - h, ci_high=dd.mean() + h,
                                                         p=2 * stats.t.sf(abs(t), J - 1), seeds_higher=int((dd > 0).sum()))
    (OUT / "secondary.json").write_text(json.dumps(res, indent=1, default=float))
    pd.set_option("display.width", 220)
    print(summ.round(2).to_string())
    print(wide[wide.arm == "gated"].round(2).to_string(index=False))
    print(tot.round(2).to_string(index=False))
    print(gbs.round(2).to_string(index=False))
    print(xr.round(2).to_string(index=False))
    print(json.dumps(res["descriptive_S_text_cat_minus_text_only"], default=float))


if __name__ == "__main__":
    main()
