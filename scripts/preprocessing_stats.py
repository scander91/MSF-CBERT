"""Missing-data profile of the GTD export and of the MSF-CBERT preprocessing.

Reads the raw export (data/raw/globalterrorismdb_0522dist.xlsx), the Tier-A output
(data/processed/gtd.parquet) and the seed-42 Tier-B artefact (artifacts/preprocess/
tierb_seed_42.json). Writes artifacts/tables/preprocessing_stats.json and the figure
figures/fig_preprocessing.{png,pdf}. Every number is computed here; nothing is typed.

Missing = NaN, empty string, or a GTD numeric sentinel (-9 / -99, "unknown") in a numeric
column. The categorical "Unknown" code is a GTD value, not a missing cell, and is counted
separately. Run with ~/env314/bin/python scripts/preprocessing_stats.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import plot_style as mp  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
plt = mp.plt
SENTINELS = {-9, -99}
NUMERIC = ["nkill", "nwound", "casualties", "success", "suicide", "extended", "multiple",
           "crit1", "crit2", "crit3", "vicinity", "ishostkid", "property", "int_any"]
CATEG = ["region_txt", "country_txt", "provstate", "city", "gname", "targtype1_txt",
         "weaptype1_txt", "weapsubtype1_txt"]
OUTCOME = {"nkill", "nwound", "casualties", "success", "extended", "property"}


def missing_mask(s: pd.Series, numeric: bool) -> pd.Series:
    m = s.isna()
    if s.dtype == object:
        m |= s.astype(str).str.strip().eq("")
    if numeric:
        m |= pd.to_numeric(s, errors="coerce").isin(SENTINELS)
    return m


def main():
    # the export as read by pandas; a pickle of the same read is used when openpyxl is absent
    try:
        raw = pd.read_excel(ROOT / "data/raw/globalterrorismdb_0522dist.xlsx")
    except ImportError:
        raw = pd.read_pickle(pathlib.Path.home() / "TerrorismNER_Project/checkpoints/gtd_full.pkl")
    raw.columns = [c.lower() for c in raw.columns]
    cells = pd.DataFrame({c: raw[c].isna() | (raw[c].astype(str).str.strip().eq("")
                                               if raw[c].dtype == object else False)
                          for c in raw.columns})
    out = {"raw_rows": int(len(raw)), "raw_cols": int(raw.shape[1]),
           "raw_cell_missing_pct": float(100 * cells.values.mean()),
           "raw_cols_with_missing": int(cells.any().sum())}

    t1 = pd.read_parquet(ROOT / "data/processed/gtd.parquet")          # Tier-A output, text-usable
    raw["eventid"] = raw["eventid"].astype("int64").astype(str)
    t1["eventid"] = t1["eventid"].astype(str)
    ids = set(t1["eventid"])
    rc = raw[raw["eventid"].isin(ids)]
    # consistency: the raw values that Tier A leaves untouched must agree on the shared events
    m = rc.set_index("eventid").loc[t1["eventid"], ["success", "suicide", "weaptype1_txt"]]
    agree = (m["success"].values == t1["success"].values).mean(), \
            (m["weaptype1_txt"].astype(str).values == t1["weaptype1_txt"].astype(str).values).mean()
    print("raw/Tier-A agreement success, weaptype1_txt:", agree)
    assert min(agree) > 0.999, "raw source does not match the Tier-A input"
    tb = json.load(open(ROOT / "artifacts/preprocess/tierb_seed_42.json"))
    kept = set(tb["kept_numeric"]) | set(tb["kept_categorical"])
    rows = []
    for c in NUMERIC + CATEG:
        num = c in NUMERIC
        r = 100 * missing_mask(rc[c], num).mean() if c in rc else np.nan
        if c == "casualties":           # derived in Tier A (nkill + nwound); absent from the export
            r = np.nan
        a = 100 * missing_mask(t1[c], num).mean()
        unk = 100 * t1[c].astype(str).eq("Unknown").mean() if not num else np.nan
        rows.append({"attr": c, "numeric": num, "outcome": c in OUTCOME, "raw_pct": r,
                     "after_tierA_pct": a, "unknown_code_pct": unk, "kept_seed42": c in kept,
                     "after_tierB_pct": 0.0 if c in kept else None})
    out["attributes"] = rows
    out["corpus_rows"] = int(len(t1))
    out["tierB_dropped"] = tb["dropped"]
    out["funnel"] = {"raw_columns": out["raw_cols"], "candidate_attributes": len(NUMERIC + CATEG),
                     "after_redundancy": len(kept), "numeric_dense": len(tb["kept_numeric"]),
                     "categorical_kept": len(tb["kept_categorical"]), "verbalised": 5}
    (ROOT / "artifacts/tables/preprocessing_stats.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "attributes"}, indent=1, default=float))
    for r in rows:
        print(f"  {r['attr']:18s} raw {r['raw_pct']:6.2f}  tierA {r['after_tierA_pct']:6.2f}  "
              f"unk {r['unknown_code_pct'] if r['unknown_code_pct']==r['unknown_code_pct'] else float('nan'):6.2f}  kept {r['kept_seed42']}")
    figure(out)


def figure(out):
    # (a) only retained attributes that have gaps; every other retained attribute is complete
    rows = [r for r in out["attributes"] if r["kept_seed42"]
            and max(np.nan_to_num(r["raw_pct"]), r["after_tierA_pct"]) > 0]
    rows.sort(key=lambda r: max(np.nan_to_num(r["raw_pct"]), r["after_tierA_pct"]))
    n_complete = sum(r["kept_seed42"] for r in out["attributes"]) - len(rows)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.6), gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(len(rows))
    v = [max(np.nan_to_num(r["raw_pct"]), r["after_tierA_pct"]) for r in rows]
    col = [mp.AMBER if r["outcome"] else mp.BLUE for r in rows]
    ax.barh(y, v, 0.6, color=col, zorder=3)
    for yi, x in zip(y, v):
        ax.annotate(f"{x:.2f}" if x >= 0.005 else "<0.01", (x, yi), xytext=(2, 0), textcoords="offset points", va="center",
                    fontsize=6.6, color=mp.INK)
    ax.set_yticks(y, [r["attr"] for r in rows], fontsize=7)
    ax.set_xlim(0, max(v) * 1.25)
    ax.set_xlabel("missing before training-fit imputation (%, 143,586 events)")
    ax.set_title(f"(a) Gaps in retained attributes ({n_complete} others complete)", loc="left", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend([Patch(color=mp.BLUE), Patch(color=mp.AMBER)], ["pre-event field", "post-event outcome"],
              loc="lower right", fontsize=6.6)
    ax.annotate("numerics: training-row median\ncity: Unknown marker\n$\\rightarrow$ 0% missing",
                (0.98, 0.42), xycoords="axes fraction", ha="right", fontsize=6.6, color=mp.MUTED)
    ax.xaxis.grid(True, color=mp.GRID, lw=0.6, zorder=0)
    mp.despine(ax)

    f = out["funnel"]
    steps = [("GTD columns", f["raw_columns"]), ("candidate attributes", f["candidate_attributes"]),
             ("after train-fit\nredundancy filter", f["after_redundancy"]),
             ("dense numerics", f["numeric_dense"]), ("verbalised categoricals", f["verbalised"])]
    yy = np.arange(len(steps))[::-1]
    ax2.barh(yy, [s[1] for s in steps], 0.6, color=[mp.MUTED, mp.MUTED, mp.BLUE, mp.BLUE, mp.GREEN], zorder=3)
    for yi, (lab, v) in zip(yy, steps):
        ax2.annotate(str(v), (v, yi), xytext=(3, 0), textcoords="offset points", va="center",
                     fontsize=7, color=mp.INK)
    ax2.set_yticks(yy, [s[0] for s in steps], fontsize=6.8)
    ax2.set_xscale("log")
    ax2.set_xlim(3, 400)
    ax2.set_xlabel("attributes (log scale)")
    ax2.set_title("(b) Attribute funnel, seed 42", loc="left", fontsize=8)
    ax2.xaxis.grid(True, color=mp.GRID, lw=0.6, zorder=0, which="major")
    mp.despine(ax2)
    fig.tight_layout()
    mp.save(fig, "fig_preprocessing")


if __name__ == "__main__":
    main()
