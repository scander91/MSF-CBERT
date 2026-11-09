"""Result tables (CSV + LaTeX; LaTeX needs booktabs and adjustbox) for the additional experiments. Every number is read from
supplementary/results/processed/per_run_metrics.csv, supplementary/results/statistical_tests/contrasts.json,
supplementary/results/secondary/*, supplementary/results/param_counts.json, artifacts/groups/* and run metrics.json files
(per-class F1). Nothing is hard-coded except labels. Run analyze.py and secondary_analyses.py first.
Writes supplementary/tables/{csv,latex}/T*.{csv,tex}.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze as an  # noqa: E402  (corrected(), holm(), run paths)

ROOT = an.ROOT
RES = ROOT / "supplementary" / "results"
TAB = ROOT / "supplementary" / "tables"
SEEDS = an.SEEDS

NAME = {
    "pub_gated": "MSF-CBERT (published runs)", "pub_gated_graph": "MSF-CBERT+G (published runs)",
    "pub_text_cat": "Text+categoricals (published runs)", "pub_text_graph": "Text+graph (published runs)",
    "pub_bertgcn": "BertGCN-style (published runs)", "bertgcn": "BertGCN-style",
    "gated": "MSF-CBERT (gated fusion)", "text_cat": "Text+categoricals", "text_only": "Text only",
    "concat_direct": "Direct concatenation of standardised numerics", "concat_dense": "Concatenation of dense numeric vector",
    "mlp_matched": "MLP fusion, parameter-matched", "fixed_avg": "Fixed 0.5/0.5 average (gate removed)",
    "no_outcome": "Without outcome fields", "vicinity_only": "Numerics: vicinity only",
    "text_num": "Text+numerics (no categoricals)", "restricted": "Restricted input",
    "restricted_first": "Restricted input, first sentence", "text_first": "Text only, first sentence",
    "no_weapon": "Without weapon type", "no_perp": "Without perpetrator group", "no_coded": "Without analyst-coded fields",
    "no_extended": "Without extended", "no_property": "Without property damage", "no_casualty": "Without fatalities and injuries",
    "no_success": "Without success", "graph_nocat": "Text+graph (no categoricals)", "graph_gated": "MSF-CBERT+G",
    "struct_xgb_published": "XGBoost, earlier features (ordinal codes)", "struct_xgb_cat_parity": "XGBoost, native categoricals, same structured inputs",
    "struct_mlp_parity": "MLP (one-hot), same structured inputs", "struct_xgb_cat_no_outcome": "XGBoost, native categoricals, without outcome fields",
    "struct_mlp_no_outcome": "MLP (one-hot), without outcome fields", "struct_xgb_cat_restricted": "XGBoost, native categoricals, restricted input",
    "struct_mlp_restricted": "MLP (one-hot), restricted input",
}
REMOVED = {
    "gated": "none (text, 5 verbalised categoricals, 12 numerics)",
    "no_outcome": "nkill, nwound, success, extended, property", "vicinity_only": "11 numerics (keeps vicinity)",
    "text_cat": "all 12 numerics", "text_num": "all 5 categoricals", "text_only": "all categoricals and numerics",
    "no_weapon": "weaptype1_txt", "no_perp": "gname", "no_coded": "crit1-3, multiple, int_any",
    "no_extended": "extended", "no_property": "property", "no_casualty": "nkill, nwound", "no_success": "success",
    "restricted": "gname, weaptype1_txt, 11 numerics (keeps vicinity)",
    "restricted_first": "as restricted; narrative cut to date stamp + first sentence",
    "text_first": "all structured inputs; narrative cut to first sentence",
}


def pm(m, s, d=2):
    return "--" if pd.isna(m) else (f"{m:.{d}f}" if pd.isna(s) else f"{m:.{d}f} ± {s:.{d}f}")


def pfmt(p):
    return "--" if p is None or pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def ci(r):
    return f"{r['mean_diff']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"


def tex_escape(s):
    return (str(s).replace("±", "$\\pm$").replace("<0.001", "$<$0.001").replace("%", "\\%").replace("_", "\\_")
            .replace("&", "\\&").replace("≥", "$\\geq$").replace("→", "$\\rightarrow$").replace("−", "$-$"))


def write(df, name, caption, label, colspec=None, note=None, wide=False, size="\\footnotesize"):
    (TAB / "csv").mkdir(parents=True, exist_ok=True); (TAB / "latex").mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "csv" / f"{name}.csv", index=False)
    cols = list(df.columns)
    colspec = colspec or ("l" + "r" * (len(cols) - 1))
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[htbp]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\setlength{\\tabcolsep}{3pt}", "\\begin{adjustbox}{max width=\\linewidth}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             " & ".join(f"\\textbf{{{tex_escape(c)}}}" for c in cols) + "\\\\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(tex_escape(v) for v in r.values) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if note:
        lines.append(f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\scriptsize {note}}}")
    lines.append(f"\\end{{{env}}}")
    (TAB / "latex" / f"{name}.tex").write_text("\n".join(lines) + "\n")


def main():
    pr = pd.read_csv(RES / "processed" / "per_run_metrics.csv")
    summ = pd.read_csv(RES / "processed" / "arm_summary.csv").set_index(["partition", "arm"])
    con = json.loads((RES / "statistical_tests" / "contrasts.json").read_text())
    sec = RES / "secondary"
    params = json.loads((RES / "param_counts.json").read_text())
    dS = pr[pr.partition == "S"]; dG = pr[pr.partition == "G"]

    def s(part, arm, col="test_f1"):
        try:
            r = summ.loc[(part, arm)]
        except KeyError:
            return "--"
        return pm(r[f"{col}_mean"], r[f"{col}_sd"])

    def find(fam, a, b):
        for r in con[fam]:
            if r["a"] == a and r["b"] == b:
                return r
        return None

    # ---------- T1 complete baseline comparison
    rows = []
    for grp, arms in [("Published ten-seed runs (paper Table VI)", ["pub_gated", "pub_gated_graph", "pub_text_cat", "pub_text_graph", "pub_bertgcn"]),
                      ("Same-configuration retraining", ["gated", "text_cat", "text_only"]),
                      ("Neural fusion alternatives (same inputs)", ["concat_direct", "concat_dense", "mlp_matched", "fixed_avg"]),
                      ("Structured-only (no narrative)", ["struct_xgb_published", "struct_xgb_cat_parity", "struct_mlp_parity"])]:
        for a in arms:
            ga = {"pub_bertgcn": "bertgcn", "pub_gated_graph": "graph_gated"}.get(a, a)
            ga = None if a in ("pub_gated", "pub_text_cat", "pub_text_graph") else ga
            rows.append({"Group": grp, "System": NAME[a], "S test macro-F1": s("S", a), "S test acc.": s("S", a, "test_acc"),
                         "G test macro-F1": s("G", ga) if ga else "--", "G test acc.": s("G", ga, "test_acc") if ga else "--"})
    t1 = pd.DataFrame(rows)
    write(t1.drop(columns="Group"), "T1_baseline_comparison",
          "Complete baseline comparison over ten seeds (\\%, mean $\\pm$ SD). S: original stratified splits; G: group-disjoint splits.",
          "tab:s_baselines", "lrrrr",
          note="BertGCN-style and MSF-CBERT+G G-columns are the same systems evaluated on the group-disjoint splits. "
               "Published runs and same-configuration retraining use identical splits and settings on different GPUs.")

    # ---------- T2 parameter-matched fusion
    rows = []
    for a in ["gated", "concat_direct", "concat_dense", "mlp_matched", "fixed_avg", "text_cat"]:
        r = find("F1_gating_S", "gated", a) or next((x for x in con["descriptive_unadjusted_S"] if x["a"] == "gated" and x["b"] == a), None)
        rg = next((x for x in con["descriptive_unadjusted_G"] if x["a"] == "gated" and x["b"] == a), None)
        if a == "text_cat":
            r = find("F2_features_S", "gated", "text_cat"); rg = find("F3_group_disjoint_G", "gated", "text_cat")
        pc = params[a]
        rows.append({"Fusion": NAME[a], "Trainable params": f"{pc['total']:,}", "Non-encoder params": f"{pc['non_encoder']:,}",
                     "S macro-F1": s("S", a), "Gated − variant (S) [95% CI]": "--" if a == "gated" else ci(r),
                     "p (Holm)": "--" if a == "gated" else (f"{pfmt(r['p'])} ({pfmt(r.get('p_holm'))})" if r.get("p_holm") is not None else f"{pfmt(r['p'])} (unadj.)"),
                     "Seeds gated higher": "--" if a == "gated" else f"{r['seeds_a_higher']}/{r['n']}",
                     "G macro-F1": s("G", a),
                     "Gated − variant (G)": ("--" if a == "gated" or rg is None else f"{rg['mean_diff']:+.2f}, p{pfmt(rg["p"]) if rg["p"] < 0.001 else "=" + pfmt(rg["p"])}, {rg['seeds_a_higher']}/{rg['n']}")})
    write(pd.DataFrame(rows), "T2_parameter_matched_fusion",
          "Fusion of the same text and numeric inputs (ten seeds, \\%). Differences are paired over seeds; Nadeau--Bengio corrected resampled $t$.",
          "tab:s_fusion", "lrrrrrrrr", wide=True,
          note="All variants share the ConfliBERT encoder (109,081,344 parameters), inputs, splits, optimiser, epochs, early stopping and validation-based checkpoint selection. "
               "Holm adjustment within the three gating comparisons (concatenation, matched MLP, fixed average); concatenation of the dense vector is descriptive. "
               "Text+categoricals (no numeric pathway) is shown for reference; its S contrast belongs to the feature-dependence family and its G contrast to the group-disjoint family.")

    # ---------- T3 feature availability audit
    fa = pd.read_csv(RES / "feature_availability_audit.csv")
    t3 = fa[["feature", "gtd_column", "role_in_architecture", "available_at_initial_report", "post_event_outcome_or_label_defining", "arm_treatment"]].copy()
    t3.columns = ["Feature", "GTD column", "Role", "Available at first report", "Post-event / label-linked", "Treatment in experiments"]
    write(t3, "T3_feature_availability_audit", "Feature availability and target-leakage audit of all model inputs.",
          "tab:s_audit", "p{2.0cm}p{2.0cm}p{3.0cm}p{2.1cm}p{2.6cm}p{3.6cm}", wide=True, size="\\scriptsize",
          note="Full justification and uncertainty for each field: feature\\_availability\\_audit.csv. GTD gives no authorship time for summaries, so availability at first report cannot be established from the data.")
    fa.to_csv(TAB / "csv" / "T3_feature_availability_audit_full.csv", index=False)

    # ---------- T4 feature-group ablation
    rows = []
    order = ["gated", "no_outcome", "no_success", "no_extended", "no_property", "no_casualty", "no_coded", "vicinity_only",
             "text_cat", "no_perp", "no_weapon", "text_num", "text_only", "restricted", "restricted_first", "text_first"]
    for a in order:
        if a == "gated":
            r = None
        else:
            r = find("F2_features_S", "gated", a); fam = "F2 (Holm)"
            if r is None:
                r = next(x for x in con["descriptive_unadjusted_S"] if x["a"] == "gated" and x["b"] == a); fam = "unadjusted"
        rg = None
        if a in ("no_outcome", "restricted"):
            rg = next(x for x in con["descriptive_unadjusted_G"] if x["a"] == "gated" and x["b"] == a)
        elif a in ("text_cat", "text_only"):
            rg = find("F3_group_disjoint_G", "gated", a)
        ch = pr[(pr.partition == "chrono") & (pr.arm == a)]
        rows.append({"Setting": NAME[a] if a != "gated" else "Full input (MSF-CBERT, gated)", "Removed inputs": REMOVED[a],
                     "S macro-F1": s("S", a), "Variant − full (S) [95% CI]": "--" if r is None else
                     f"{-r['mean_diff']:+.2f} [{-r['ci_high']:+.2f}, {-r['ci_low']:+.2f}]",
                     "p": "--" if r is None else (f"{pfmt(r['p_holm'])} (Holm)" if fam.startswith("F2") else f"{pfmt(r['p'])} (unadj.)"),
                     "Seeds variant higher": "--" if r is None else f"{r['seeds_b_higher']}/{r['n']}",
                     "G macro-F1": s("G", a), "Variant − full (G)": "--" if rg is None else f"{-rg['mean_diff']:+.2f}",
                     "Chrono macro-F1": (f"{ch.test_macro_f1.iloc[0]:.2f}" if len(ch) else
                                         (f"{pr[(pr.partition=='chrono')&(pr.arm=='pub_chrono_structdense_nograph')].test_macro_f1.iloc[0]:.2f}" if a == "gated" else "--"))})
    write(pd.DataFrame(rows), "T4_feature_group_ablation",
          "Feature-group exclusion and restricted-input sensitivity (ten seeds; chronological split: one run).",
          "tab:s_ablation", "p{3.3cm}p{3.6cm}rrrrrrr", wide=True, size="\\scriptsize",
          note="Reference: same-configuration retraining of MSF-CBERT. Holm adjustment over the four feature-dependence comparisons (outcome fields, numerics, categoricals, all structured inputs); all other rows unadjusted. "
               "Restricted input keeps the narrative, region, country, target type and vicinity; it is a reduced-input sensitivity study, not an estimate of performance at first report, because GTD summaries are written after the event. "
               "The chronological full-input value is the published MSF-CBERT chronological run.")

    # ---------- T5 original vs group-disjoint
    gr = json.loads((ROOT / "artifacts" / "groups" / "groups_report.json").read_text())
    exp_s = pd.read_csv(sec / "group_exposure_summary.csv").set_index("partition")
    gbs = pd.read_csv(sec / "group_balanced_macro_f1_summary.csv").set_index(["partition", "arm"])
    sz = []
    for sd in SEEDS:
        rep = json.loads((ROOT / "artifacts" / "groups" / f"groupseed_split_report_{sd}.json").read_text())[0]
        sz.append(dict(seed=sd, **{f"n_{k}": v for k, v in rep["sizes"].items()}, **{f"groups_{k}": v for k, v in rep["groups"].items()},
                       max_share_gap=rep["max_abs_class_share_diff_test_vs_corpus"]))
    sz = pd.DataFrame(sz); sz.to_csv(TAB / "csv" / "T5b_group_split_sizes.csv", index=False)
    szt = pd.DataFrame({"Seed": sz.seed, "Train events": sz.n_train.map("{:,}".format), "Val events": sz.n_val.map("{:,}".format),
                        "Test events": sz.n_test.map("{:,}".format), "Train groups": sz.groups_train.map("{:,}".format),
                        "Val groups": sz.groups_val.map("{:,}".format), "Test groups": sz.groups_test.map("{:,}".format),
                        "Max class-share gap (pts)": (100 * sz.max_share_gap).map("{:.3f}".format)})
    write(szt, "T5b_group_split_sizes", "Group-disjoint partitions: sizes, groups per partition and largest test-vs-corpus class-share difference.",
          "tab:s_splitsizes", "rrrrrrrr", note="Twenty StratifiedGroupKFold folds (label-stratified, grouped by the primary 0.90 groups) assigned 14/3/3 to train/validation/test with the split seed. No group crosses partitions (asserted). Class counts per partition: groupseed\\_split\\_report\\_*.json.")
    rows = []
    for a in ["gated", "text_cat", "text_only", "concat_direct", "mlp_matched", "no_outcome", "restricted"]:
        x = next(r for r in con["S_vs_G_descriptive_no_test"] if r["arm"] == a)
        gb = lambda p: pm(gbs.loc[(p, a)].gb_macro_f1_mean, gbs.loc[(p, a)].gb_macro_f1_sd) if (p, a) in gbs.index else "--"
        rows.append({"System": NAME[a], "S macro-F1": pm(x["S_mean"], x["S_sd"]), "G macro-F1": pm(x["G_mean"], x["G_sd"]),
                     "G − S": f"{x['G_minus_S']:+.2f}", "S acc.": s("S", a, "test_acc"), "G acc.": s("G", a, "test_acc"),
                     "S group-balanced F1": gb("S"), "G group-balanced F1": gb("G")})
    t5 = pd.DataFrame(rows)
    write(t5, "T5_original_vs_group_disjoint",
          "Original stratified (S) versus group-disjoint (G) evaluation, same configuration, ten seeds each (\\%).",
          "tab:s_svsg", "lrrrrrrr", wide=True,
          note=(f"Groups: connected components of GTD related links, identical summaries (date stamp removed, whitespace-normalised, lower-case) and TF-IDF cosine $\\geq$0.90 (exhaustive pairwise comparison); "
                f"{gr['all_0.9']['groups']:,} groups, {gr['all_0.9']['singletons']:,} singletons, {gr['all_0.9']['events_in_multi']:,} events in multi-event groups, largest {gr['all_0.9']['largest']}. "
                f"G test partitions: {sz.n_test.min():,}--{sz.n_test.max():,} events in {sz.groups_test.min():,}--{sz.groups_test.max():,} groups; maximum class-share gap test vs corpus {100*sz.max_share_gap.max():.3f} points. "
                f"In S, {exp_s.loc['S','exposed_pct_g_90_min']:.2f}--{exp_s.loc['S','exposed_pct_g_90_max']:.2f}\\% of test events have a group member in training (0\\% in G). "
                "G $-$ S is a difference of ten-seed means without a test (the splits overlap, so seeds are neither paired nor independent); no events are removed and class shares are held equal by stratification. "
                "Group-balanced macro-F1 weights each test event by 1/(size of its group in the test set)."))

    # exposure sensitivity table
    rows = []
    defs = [("g_rel_exact", "related + identical summary"), ("g_95", "+ cosine $\\geq$0.95"), ("g_90", "+ cosine $\\geq$0.90 (primary)"),
            ("g_85", "+ cosine $\\geq$0.85"), ("g_80", "+ cosine $\\geq$0.80")]
    key = {"g_rel_exact": "related+exact", "g_95": "all_0.95", "g_90": "all_0.9", "g_85": "all_0.85", "g_80": "all_0.8"}
    for g, lab in defs:
        rows.append({"Grouping": lab, "Groups": f"{gr[key[g]]['groups']:,}", "Events in multi-event groups": f"{gr[key[g]]['events_in_multi']:,}",
                     "Largest group": gr[key[g]]["largest"],
                     "S: test exposed % (min--max)": f"{exp_s.loc['S', f'exposed_pct_{g}_min']:.2f}--{exp_s.loc['S', f'exposed_pct_{g}_max']:.2f}",
                     "G: test exposed % (min--max)": f"{exp_s.loc['G', f'exposed_pct_{g}_min']:.2f}--{exp_s.loc['G', f'exposed_pct_{g}_max']:.2f}"})
    t5c = pd.DataFrame(rows)
    (TAB / "csv").mkdir(parents=True, exist_ok=True)
    write(t5c, "T5c_group_exposure_sensitivity",
          "Grouping thresholds and exposure: share of test events with a member of their group in the training partition.",
          "tab:s_exposure", "lrrrrr",
          note="Edges accumulate: each row adds near-duplicate pairs at the stated TF-IDF cosine to the related and identical-summary links. "
               "G splits are built from the primary (0.90) groups, so their exposure is zero by construction at $\\geq$0.90 and non-zero only at looser thresholds. "
               "At 0.80, chaining produces a component of 1,076 events, so 0.80 is used for sensitivity only.")

    # per-class exposed / unexposed
    pc = pd.read_csv(sec / "perclass_exposed_unexposed_pooled.csv")
    pc = pc[pc.arm == "gated"]
    t5d = pd.DataFrame({"Attack type": pc.cls, "Unexposed n (pooled)": pc.n_unexposed_pooled.astype(int).map("{:,}".format),
                        "Exposed n (pooled)": pc.n_exposed_pooled.astype(int).map("{:,}".format),
                        "Acc. unexposed": pc.acc_unexposed.map("{:.2f}".format), "Acc. exposed": pc.acc_exposed.map("{:.2f}".format),
                        "Exposed − unexposed": pc.exposed_minus_unexposed.map("{:+.2f}".format)})
    tot = pd.read_csv(sec / "overall_accuracy_exposed_unexposed_pooled.csv")
    write(t5d, "T5d_perclass_exposed_unexposed",
          "Per-class test accuracy of MSF-CBERT trained on the original splits, for test events with and without a group member in training (pooled over ten seeds, \\%).",
          "tab:s_exposed", "lrrrrr",
          note="Same-configuration retraining. Pooled overall accuracy: " + "; ".join(
              f"{NAME[a] if a!='pub_gated' else 'published MSF-CBERT runs'} {t.loc[t.exposed==False,'acc_pct'].iloc[0]:.2f} unexposed vs {t.loc[t.exposed==True,'acc_pct'].iloc[0]:.2f} exposed"
              for a, t in tot.groupby("arm", sort=False)) + ". Descriptive; exposed and unexposed events differ in class mix and difficulty, so no subset macro-F1 is claimed.")

    # ---------- T6 statistics (all families)
    rows = []
    fams = {"F1_gating_S": "Fusion (S)", "F2_features_S": "Feature dependence (S)", "F3_group_disjoint_G": "Group-disjoint (G)",
            "F4_graph": "Graph", "descriptive_unadjusted_S": "descriptive (S)", "descriptive_unadjusted_G": "descriptive (G)"}
    for k, lab in fams.items():
        for r in con[k]:
            rows.append({"Family": lab, "Comparison": f"{NAME.get(r['a'], r['a'])} vs {NAME.get(r['b'], r['b'])}",
                         "Mean diff. [95% CI]": ci(r), "d_z": f"{r['d_z']:.2f}", "p": pfmt(r["p"]),
                         "p Holm": pfmt(r.get("p_holm")), "Seeds first higher": f"{r['seeds_a_higher']}/{r['n']}"})
    write(pd.DataFrame(rows), "T6_statistical_tests",
          "Paired comparisons of test macro-F1 (points): Nadeau--Bengio corrected resampled $t$ (two-sided, $n_{test}/n_{train}=0.15/0.70$), Holm within each confirmatory family (fusion, feature dependence, group-disjoint, graph).",
          "tab:s_stats", "llrrrrr", wide=True, size="\\scriptsize",
          note="Effect size d\\_z = mean paired difference / SD of differences. Descriptive rows are unadjusted. The corrected test accounts for training-set overlap between random splits but remains approximate with ten seeds.")

    # ---------- T7 graph contribution
    pub = an.table({k: v for k, v in an.S.items() if k.startswith("pub_")}, "S")
    rows = []
    for lab, a, b, df, fam in [("MSF-CBERT+G − MSF-CBERT (S, published runs)", "pub_gated_graph", "pub_gated", pub, "published"),
                               ("Text+graph − text+categoricals (S, published runs)", "pub_text_graph", "pub_text_cat", pub, "published"),
                               ("MSF-CBERT+G − MSF-CBERT (G)", "graph_gated", "gated", dG, "F4"),
                               ("Text+graph (no categoricals) − text only (S)", "graph_nocat", "text_only", dS, "F4")]:
        r = an.corrected(a, b, df)
        if fam == "F4":
            r = next(x for x in con["F4_graph"] if x["a"] == a)
        rows.append({"Contrast": lab, "Mean diff. [95% CI]": ci(r), "p": pfmt(r["p"]), "p Holm": pfmt(r.get("p_holm")),
                     "Seeds first higher": f"{r['seeds_a_higher']}/{r['n']}"})
    sc = json.loads((sec / "secondary.json").read_text())["descriptive_S_text_cat_minus_text_only"]
    rows.append({"Contrast": "Text+categoricals − text only (S, reference)", "Mean diff. [95% CI]": f"{sc['mean_diff']:+.2f} [{sc['ci_low']:+.2f}, {sc['ci_high']:+.2f}]",
                 "p": pfmt(sc["p"]), "p Holm": "--", "Seeds first higher": f"{sc['seeds_higher']}/10"})
    write(pd.DataFrame(rows), "T7_graph_contribution", "Graph contribution (test macro-F1, points, ten seeds).", "tab:s_graph", "lrrrr",
          note="The graph links perpetrator, target type, weapon type and country, the same fields that are verbalised as categoricals. Published-run rows reproduce the paper's contrasts (unadjusted here).")

    # ---------- T8 compact existing vs new
    pubv = lambda a: summ.loc[("S", a)]
    rpub = an.corrected("gated", "pub_gated", dS)
    rows = [
        {"Claim": "MSF-CBERT test macro-F1 (S)", "Manuscript": "86.01 ± 0.67", "Verified / new": f"{pm(pubv('pub_gated').test_f1_mean, pubv('pub_gated').test_f1_sd)} recomputed; retraining {s('S','gated')}", "Source": "per_run_metrics.csv"},
        {"Claim": "MSF-CBERT accuracy (S)", "Manuscript": "93.88 ± 0.16", "Verified / new": pm(pubv('pub_gated').test_acc_mean, pubv('pub_gated').test_acc_sd), "Source": "per_run_metrics.csv"},
        {"Claim": "MSF-CBERT+G (S)", "Manuscript": "85.84 ± 0.89", "Verified / new": pm(pubv('pub_gated_graph').test_f1_mean, pubv('pub_gated_graph').test_f1_sd), "Source": "per_run_metrics.csv"},
        {"Claim": "Text+categoricals (S)", "Manuscript": "84.68 ± 0.81", "Verified / new": f"{pm(pubv('pub_text_cat').test_f1_mean, pubv('pub_text_cat').test_f1_sd)}; retraining {s('S','text_cat')}", "Source": "per_run_metrics.csv"},
        {"Claim": "Text+graph (S)", "Manuscript": "84.81 ± 1.00", "Verified / new": pm(pubv('pub_text_graph').test_f1_mean, pubv('pub_text_graph').test_f1_sd), "Source": "per_run_metrics.csv"},
        {"Claim": "BertGCN-style (S)", "Manuscript": "84.99 ± 0.84", "Verified / new": pm(pubv('pub_bertgcn').test_f1_mean, pubv('pub_bertgcn').test_f1_sd), "Source": "per_run_metrics.csv"},
        {"Claim": "Chronological MSF-CBERT", "Manuscript": "83.30", "Verified / new": f"{pr[(pr.partition=='chrono')&(pr.arm=='pub_chrono_structdense_nograph')].test_macro_f1.iloc[0]:.2f}", "Source": "per_run_metrics.csv"},
        {"Claim": "Retraining − published MSF-CBERT (run-to-run)", "Manuscript": "--", "Verified / new": f"{rpub['mean_diff']:+.2f}, p={pfmt(rpub["p"])}, {rpub['seeds_a_higher']}/10", "Source": "contrasts.csv"},
        {"Claim": "Test events sharing a group with training (S)", "Manuscript": "3.05% verbatim, 3.30% near-dup.",
         "Verified / new": f"{exp_s.loc['S','exposed_pct_g_90_min']:.2f}--{exp_s.loc['S','exposed_pct_g_90_max']:.2f}% (related, identical, cosine ≥0.90)", "Source": "group_exposure_summary.csv"},
        {"Claim": "MSF-CBERT on group-disjoint splits", "Manuscript": "--", "Verified / new": s("G", "gated"), "Source": "arm_summary.csv"},
        {"Claim": "Gate vs simpler fusion", "Manuscript": "gated branch (no comparison)", "Verified / new": "no measurable difference (Holm p ≥ 0.46)", "Source": "contrasts.csv (fusion family)"},
        {"Claim": "Graph redundancy", "Manuscript": "not isolated", "Verified / new": "graph without categoricals +3.89 over text only; +G on G −0.56 (p=0.075)", "Source": "contrasts.csv (graph family)"},
        {"Claim": "Unknown categorical values", "Manuscript": "explicit Unknown states", "Verified / new": "kept as Unknown in data; verbalisation omits the phrase", "Source": "src/features/verbalize.py"},
    ]
    write(pd.DataFrame(rows), "T8_existing_vs_new", "Manuscript claims versus verified and new results.", "tab:s_compact",
          "p{4.2cm}p{3.2cm}p{6.0cm}p{3.2cm}", wide=True, size="\\scriptsize")

    # ---------- T9 chronological arms (single run each)
    ch = pr[pr.partition == "chrono"].copy()
    ch["System"] = ch.arm.map({"pub_chrono_structdense_nograph": "MSF-CBERT (published run)", "pub_chrono_nograph": "Text+categoricals (published run)",
                               "no_outcome": NAME["no_outcome"], "restricted": NAME["restricted"], "restricted_first": NAME["restricted_first"], "text_only": NAME["text_only"]})
    t9 = ch[["System", "val_macro_f1", "test_macro_f1", "test_acc"]].rename(columns={"val_macro_f1": "Val macro-F1", "test_macro_f1": "Test macro-F1", "test_acc": "Test acc."})
    for c in ["Val macro-F1", "Test macro-F1", "Test acc."]:
        t9[c] = t9[c].map("{:.2f}".format)
    write(t9, "T9_chronological", "Chronological split (train to 2016-08-18, test from 2018-07-09), one run per system (\\%).", "tab:s_chrono", "lrrr",
          note="Single split: descriptive only, no test.")

    # ---------- T10 structured-only diagnosis
    xr = pd.read_csv(sec / "struct_xgb_cat_restricted_diagnosis.csv")
    t10 = xr.groupby(["partition", "class_balanced"]).agg(n=("seed", "size"), val=("val_macro_f1", "mean"), f1=("test_macro_f1", "mean"),
                                                          acc=("test_acc", "mean"), share=("pred_share_bombing", "mean")).reset_index()
    t10 = pd.DataFrame({"Split": t10.partition, "Validation-selected class weighting": t10.class_balanced.map({True: "balanced", False: "none"}),
                        "Seeds": t10.n, "Val macro-F1": t10.val.map("{:.2f}".format), "Test macro-F1": t10.f1.map("{:.2f}".format),
                        "Test acc.": t10.acc.map("{:.2f}".format), "Predicted Bombing share %": t10.share.map("{:.1f}".format)})
    write(t10, "T10_xgb_restricted_diagnosis", "XGBoost on restricted structured inputs: source of the large accuracy SD across seeds.", "tab:s_xgbdiag", "llrrrrr",
          note="Validation macro-F1 is nearly tied between class-balanced and unweighted training (differences below one point), so the validation-selected setting alternates across seeds. "
               "Balanced weighting predicts the majority class far less often and lowers accuracy by about 19 points at equal macro-F1. Reported as selected on validation; not re-selected on test.")

    # ---------- T11 per-class F1 S vs G (gated, text_cat)
    lab = json.loads((an.RUNS / "seed_42__gated" / "metrics.json").read_text())["labels"]
    rows = []
    for c in lab:
        r = {"Attack type": c}
        for part, pre in [("S", "seed"), ("G", "groupseed")]:
            for a in ["gated", "text_cat"]:
                v = [100 * json.loads((an.RUNS / f"{pre}_{sd}__{a}" / "metrics.json").read_text())["report"][c]["f1-score"] for sd in SEEDS]
                r[f"{part} {('MSF-CBERT' if a=='gated' else 'Text+cat.')}"] = pm(np.mean(v), np.std(v, ddof=1))
        rows.append(r)
    write(pd.DataFrame(rows), "T11_perclass_f1_S_vs_G", "Per-class test F1 on original (S) and group-disjoint (G) splits, same configuration, ten seeds (\\%, mean $\\pm$ SD).",
          "tab:s_perclass", "lrrrr")
    print("tables written to", TAB)


if __name__ == "__main__":
    main()
