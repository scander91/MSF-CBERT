"""Figures for the additional experiments (PDF/PNG/SVG + plotting-data CSV per figure).

Every value is computed from supplementary/results/processed/per_run_metrics.csv (paired Nadeau-Bengio
corrected intervals, as in analyze.py) or supplementary/results/secondary/*.csv. Style follows the paper's
existing figures (Okabe-Ito blue/amber/green, DejaVu Sans, no red).
Run: python experiments/make_figures.py   (after analyze.py and secondary_analyses.py)
Main-paper figure: fig_contrasts_panels (replaces the single-panel fig_contrast_forest).
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze as an  # noqa: E402

RES = an.ROOT / "supplementary" / "results"
FIG = an.ROOT / "supplementary" / "figures"
BLUE, AMBER, GREEN = "#0072B2", "#E69F00", "#009E73"
INK, MUTED, GRID = "#1a1a1a", "#5b5b5b", "#d9d9d9"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})
STYLE = {"sys": (GREEN, "s"), "comp": (BLUE, "o"), "g": (AMBER, "D")}


def save(fig, name, data):
    for sub in ("pdf", "png", "svg", "data"):
        (FIG / sub).mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / ext / f"{name}.{ext}")
    pd.DataFrame(data).to_csv(FIG / "data" / f"{name}.csv", index=False)
    plt.close(fig)
    print("wrote", name)


def despine(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def contrast(df, a, b):
    r = an.corrected(a, b, df)
    return r["mean_diff"], r["ci_low"], r["ci_high"], r["seeds_a_higher"], r["n"], r["p"]


def forest(ax, rows, title, xlabel, xticks=None, ann_x=1.04):
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        c, mk = STYLE[r["kind"]]
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=c, lw=1.6, solid_capstyle="round", zorder=3)
        ax.plot(r["mean"], yi, mk, color=c, ms=4.3, mec="white", mew=0.7, zorder=4)
        ax.annotate(f"{r['mean']:+.2f} ({r['k']}/{r['n']})", (ann_x, yi), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=5.6, color=INK, annotation_clip=False)
    ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
    ax.set_yticks(y, [r["label"] for r in rows], fontsize=6.3)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.tick_params(axis="y", length=0, pad=2)
    lo = min(min(r["lo"] for r in rows), 0); hi = max(max(r["hi"] for r in rows), 0); span = hi - lo
    ax.set_xlim(lo - 0.08 * span, hi + 0.08 * span)
    if xticks is not None:
        ax.set_xticks(xticks)
    else:
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel(xlabel, fontsize=6, labelpad=1.5)
    ax.tick_params(axis="x", labelsize=5.8, pad=1.5)
    if title:
        ax.set_title(title, loc="left", fontsize=6.8, pad=3, x=-0.9)
    despine(ax)


def build_rows(spec, dfs):
    rows = []
    for label, part, a, b, kind in spec:
        m, lo, hi, k, n, p = contrast(dfs[part], a, b)
        rows.append(dict(panel=None, label=label, partition=part, a=a, b=b, kind=kind, mean=m, lo=lo, hi=hi, k=k, n=n, p=p))
    span = max(max(r["hi"], 0) for r in rows) - min(min(r["lo"] for r in rows), 0)
    for r in rows:
        r["span"] = span
    return rows


def fig_panels(dfs):
    specs = {
        "(a) Stratified splits, primary runs": [
            ("BertGCN-style", "S", "pub_gated", "pub_bertgcn", "sys"),
            ("text+graph", "S", "pub_gated", "pub_text_graph", "sys"),
            ("text+cat.", "S", "pub_gated", "pub_text_cat", "sys"),
            ("+ graph", "S", "pub_gated_graph", "pub_gated", "comp"),
            ("text+cat. + graph", "S", "pub_text_graph", "pub_text_cat", "comp")],
        "(b) Group-disjoint splits": [
            ("BertGCN-style", "G", "gated", "bertgcn", "sys"),
            ("text+cat.", "G", "gated", "text_cat", "sys"),
            ("text only", "G", "gated", "text_only", "sys"),
            ("+ graph", "G", "graph_gated", "gated", "comp")],
        "(c) Gate vs. simpler fusion (S)": [
            ("concat.", "S", "gated", "concat_direct", "sys"),
            ("concat. (G)", "G", "gated", "concat_direct", "g"),
            ("dense concat.", "S", "gated", "concat_dense", "sys"),
            ("MLP", "S", "gated", "mlp_matched", "sys"),
            ("MLP (G)", "G", "gated", "mlp_matched", "g"),
            ("fixed avg.", "S", "gated", "fixed_avg", "sys")],
        "(d) Feature-group exclusion (S)": [
            ("− outcome", "S", "no_outcome", "gated", "comp"),
            ("vicinity only", "S", "vicinity_only", "gated", "comp"),
            ("− numerics", "S", "text_cat", "gated", "comp"),
            ("− weapon", "S", "no_weapon", "gated", "comp"),
            ("− categoricals", "S", "text_num", "gated", "comp"),
            ("text only", "S", "text_only", "gated", "comp"),
            ("restricted", "S", "restricted", "gated", "comp"),
            ("restr., 1st sent.", "S", "restricted_first", "gated", "comp")],
    }
    W, H = 7.16, 1.36
    fig = plt.figure(figsize=(W, H))
    lab_w, plot_w, ann_w = 0.80, 0.50, 0.48
    axes = [fig.add_axes([(i * (lab_w + plot_w + ann_w) + lab_w) / W, 0.2, plot_w / W, 0.70]) for i in range(4)]
    data = []
    xl = {0: "$\\Delta$ macro-F1 (points)", 1: "$\\Delta$ macro-F1 (points)", 2: "$\\Delta$ macro-F1 (points)",
          3: "$\\Delta$ when removed (points)"}
    for i, (ax, (title, spec)) in enumerate(zip(axes, specs.items())):
        rows = build_rows(spec, dfs)
        forest(ax, rows, title, xl[i], xticks=[-12, -6, 0] if i == 3 else None)
        for r in rows:
            r["panel"] = title
        data += rows
    # colour key is given in the paper caption (green: system contrast, blue: component, amber: G splits)
    save(fig, "fig_contrasts_panels", [{k: v for k, v in r.items() if k != "span"} for r in data])
    return data


def fig_s_vs_g(summ):
    arms = [("gated", "MSF-CBERT"), ("mlp_matched", "MLP fusion"), ("text_cat", "Text+categoricals"), ("no_outcome", "w/o outcome fields"),
            ("concat_direct", "Direct concatenation"), ("text_only", "Text only"), ("restricted", "Restricted input")]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    y = np.arange(len(arms))[::-1]; data = []
    for yi, (a, lab) in zip(y, arms):
        for part, c, mk, off in (("S", BLUE, "o", 0.13), ("G", AMBER, "D", -0.13)):
            r = summ.loc[(part, a)]
            ax.errorbar(r.test_f1_mean, yi + off, xerr=r.test_f1_sd, fmt=mk, color=c, ms=4, mec="white", mew=0.6, elinewidth=1.4, capsize=0)
            data.append(dict(arm=a, partition=part, mean=r.test_f1_mean, sd=r.test_f1_sd, n=r.n))
    ax.set_yticks(y, [a[1] for a in arms]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("test macro-F1 (%), mean $\\pm$ SD over ten seeds")
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0); despine(ax)
    ax.legend([Line2D([], [], color=BLUE, marker="o", ls="none"), Line2D([], [], color=AMBER, marker="D", ls="none")],
              ["original stratified splits", "group-disjoint splits"], loc="lower right", fontsize=6.3)
    save(fig, "fig_original_vs_group_disjoint", data)


def fig_baselines(summ):
    arms = [("S", "pub_gated", "MSF-CBERT (published)"), ("S", "pub_gated_graph", "MSF-CBERT+G (published)"),
            ("S", "pub_bertgcn", "BertGCN-style (published)"), ("S", "pub_text_graph", "Text+graph (published)"),
            ("S", "pub_text_cat", "Text+categoricals (published)"), ("S", "gated", "MSF-CBERT (retraining)"),
            ("S", "text_only", "Text only"), ("S", "struct_xgb_cat_parity", "XGBoost, structured only"),
            ("S", "struct_mlp_parity", "MLP, structured only")]
    fig, ax = plt.subplots(figsize=(3.5, 2.5)); y = np.arange(len(arms))[::-1]; data = []
    for yi, (p, a, lab) in zip(y, arms):
        r = summ.loc[(p, a)]
        c = GREEN if "MSF" in lab else (AMBER if "structured" in lab else BLUE)
        ax.errorbar(r.test_f1_mean, yi, xerr=r.test_f1_sd, fmt="o", color=c, ms=4, mec="white", mew=0.6, elinewidth=1.4)
        ax.annotate(f"{r.test_f1_mean:.2f}", (r.test_f1_mean + r.test_f1_sd + 0.6, yi), va="center", fontsize=6)
        data.append(dict(partition=p, arm=a, label=lab, mean=r.test_f1_mean, sd=r.test_f1_sd, n=r.n))
    ax.set_yticks(y, [a[2] for a in arms]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("test macro-F1 (%), original splits, mean $\\pm$ SD"); ax.set_xlim(62, 92)
    ax.xaxis.grid(True, color=GRID, lw=0.5); despine(ax)
    save(fig, "fig_baseline_comparison", data)


def fig_perclass_exposed():
    d = pd.read_csv(RES / "secondary" / "perclass_exposed_unexposed_pooled.csv")
    d = d[d.arm == "gated"].sort_values("n_unexposed_pooled", ascending=False)
    short = {"Facility/Infrastructure Attack": "Facility/Infra.", "Hostage Taking (Barricade Incident)": "Barricade",
             "Hostage Taking (Kidnapping)": "Kidnapping"}
    fig, ax = plt.subplots(figsize=(3.5, 2.3)); y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.plot([r.acc_unexposed, r.acc_exposed], [yi, yi], color=GRID, lw=1.5, zorder=1)
        ax.plot(r.acc_unexposed, yi, "o", color=BLUE, ms=4.5, mec="white", mew=0.6, zorder=3)
        ax.plot(r.acc_exposed, yi, "D", color=AMBER, ms=4, mec="white", mew=0.6, zorder=3)
    ax.set_yticks(y, [short.get(c, c) for c in d.cls]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("test accuracy within class (%), pooled over ten seeds")
    ax.legend([Line2D([], [], color=BLUE, marker="o", ls="none"), Line2D([], [], color=AMBER, marker="D", ls="none")],
              ["no group member in training", "group member in training"], loc="lower left", fontsize=6.3)
    ax.xaxis.grid(True, color=GRID, lw=0.5); despine(ax)
    save(fig, "fig_perclass_exposed_vs_unexposed", d)


def fig_perclass_s_g():
    lab = json.loads((an.RUNS / "seed_42__gated" / "metrics.json").read_text())["labels"]
    short = {"Facility/Infrastructure Attack": "Facility/Infra.", "Hostage Taking (Barricade Incident)": "Barricade",
             "Hostage Taking (Kidnapping)": "Kidnapping"}
    rows = []
    for c in lab:
        for part, pre in (("S", "seed"), ("G", "groupseed")):
            v = [100 * json.loads((an.RUNS / f"{pre}_{s}__gated" / "metrics.json").read_text())["report"][c]["f1-score"] for s in an.SEEDS]
            rows.append(dict(cls=c, partition=part, mean=np.mean(v), sd=np.std(v, ddof=1)))
    d = pd.DataFrame(rows); order = d[d.partition == "S"].sort_values("mean", ascending=False).cls.tolist()
    fig, ax = plt.subplots(figsize=(3.5, 2.3)); y = np.arange(len(order))[::-1]
    for yi, c in zip(y, order):
        for part, col, mk, off in (("S", BLUE, "o", 0.13), ("G", AMBER, "D", -0.13)):
            r = d[(d.cls == c) & (d.partition == part)].iloc[0]
            ax.errorbar(r["mean"], yi + off, xerr=r["sd"], fmt=mk, color=col, ms=4, mec="white", mew=0.6, elinewidth=1.3)
    ax.set_yticks(y, [short.get(c, c) for c in order]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("MSF-CBERT test F1 (%), mean $\\pm$ SD")
    ax.legend([Line2D([], [], color=BLUE, marker="o", ls="none"), Line2D([], [], color=AMBER, marker="D", ls="none")],
              ["original stratified splits", "group-disjoint splits"], loc="lower left", fontsize=6.3)
    ax.xaxis.grid(True, color=GRID, lw=0.5); despine(ax)
    save(fig, "fig_perclass_f1_S_vs_G", d)


def fig_exposure():
    e = pd.read_csv(RES / "secondary" / "group_exposure_by_definition.csv")
    defs = [("g_rel_exact", "related+\nidentical"), ("g_95", "+cos\n$\\geq$0.95"), ("g_90", "+cos\n$\\geq$0.90"),
            ("g_85", "+cos\n$\\geq$0.85"), ("g_80", "+cos\n$\\geq$0.80")]
    fig, ax = plt.subplots(figsize=(3.5, 2.1)); x = np.arange(len(defs)); data = []
    for part, c, mk, off in (("S", BLUE, "o", -0.12), ("G", AMBER, "D", 0.12)):
        sub = e[e.partition == part]
        m = [sub[f"exposed_pct_{g}"].mean() for g, _ in defs]
        lo = [sub[f"exposed_pct_{g}"].min() for g, _ in defs]; hi = [sub[f"exposed_pct_{g}"].max() for g, _ in defs]
        ax.errorbar(x + off, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], fmt=mk, color=c, ms=4.5, mec="white", mew=0.6,
                    elinewidth=1.3, label="original stratified splits" if part == "S" else "group-disjoint splits (built at 0.90)")
        for g, mm, l, h in zip(defs, m, lo, hi):
            data.append(dict(partition=part, grouping=g[0], mean=mm, min=l, max=h))
    ax.set_xticks(x, [d[1] for d in defs]); ax.set_ylabel("test events with a group\nmember in training (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.legend(loc="upper left", fontsize=6.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig_group_exposure_sensitivity", data)


def fig_graph(dfs):
    spec = [("+G vs. MSF-CBERT (S, published)", "S", "pub_gated_graph", "pub_gated", "comp"),
            ("text+graph vs. text+cat. (S, published)", "S", "pub_text_graph", "pub_text_cat", "comp"),
            ("+G vs. MSF-CBERT (G)", "G", "graph_gated", "gated", "comp"),
            ("graph, no categoricals, vs. text only (S)", "S", "graph_nocat", "text_only", "sys"),
            ("text+cat. vs. text only (S, reference)", "S", "text_cat", "text_only", "sys")]
    rows = build_rows(spec, dfs)
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    forest(ax, rows, "", "$\\Delta$ test macro-F1 (points), mean, 95% CI")
    save(fig, "fig_graph_contribution", [{k: v for k, v in r.items() if k != "span"} for r in rows])


def main():
    pr = pd.read_csv(RES / "processed" / "per_run_metrics.csv")
    dfs = {"S": pr[pr.partition == "S"], "G": pr[pr.partition == "G"]}
    summ = pd.read_csv(RES / "processed" / "arm_summary.csv").set_index(["partition", "arm"])
    for r in fig_panels(dfs):
        print(f"  {r['panel'][:4]} {r['label']:28s} {r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] {r['k']}/{r['n']} p={r['p']:.4f}")
    fig_s_vs_g(summ); fig_baselines(summ); fig_perclass_exposed(); fig_perclass_s_g(); fig_exposure(); fig_graph(dfs)


if __name__ == "__main__":
    main()
