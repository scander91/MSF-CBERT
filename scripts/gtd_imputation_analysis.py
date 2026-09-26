"""GTD data analysis: temporal/class profile and what common imputation practice
does to sensitive fields. Descriptive only (no model is trained here).

Corpus = the 143,586 events with a usable summary (data/processed/gtd.parquet ids), values
taken from the RAW export so that GTD "unknown" codes (-9/-99) and blanks are still visible.
Writes artifacts/tables/gtd_imputation_analysis.json and figures/fig_gtd_analysis.{png,pdf}.
    ~/env314/bin/python scripts/gtd_imputation_analysis.py
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
SENT = {-9, -99}
SHORT = {"Bombing/Explosion": "Bombing", "Armed Assault": "Armed assault",
         "Hostage Taking (Kidnapping)": "Kidnapping", "Assassination": "Assassination",
         "Unknown": "Unknown", "Facility/Infrastructure Attack": "Facility/infra.",
         "Unarmed Assault": "Unarmed assault", "Hostage Taking (Barricade Incident)": "Barricade",
         "Hijacking": "Hijacking"}


def cramers_v(x, y):
    t = pd.crosstab(x, y).values.astype(float)
    n = t.sum()
    e = t.sum(1, keepdims=True) @ t.sum(0, keepdims=True) / n
    chi2 = ((t - e) ** 2 / np.where(e == 0, 1, e)).sum()
    k = min(t.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def main():
    try:
        raw = pd.read_excel(ROOT / "data/raw/globalterrorismdb_0522dist.xlsx")
    except ImportError:
        raw = pd.read_pickle(pathlib.Path.home() / "TerrorismNER_Project/checkpoints/gtd_full.pkl")
    raw.columns = [c.lower() for c in raw.columns]
    raw["eventid"] = raw["eventid"].astype("int64").astype(str)
    ids = set(pd.read_parquet(ROOT / "data/processed/gtd.parquet", columns=["eventid"])["eventid"].astype(str))
    df = raw[raw["eventid"].isin(ids)].copy()
    y = df["attacktype1_txt"]
    n = len(df)
    out = {"n_events": n}

    # ---------- numeric sensitive fields: unknown = blank or GTD code -9/-99
    num = {"nkill": "fatalities", "nwound": "injured", "property": "property damage",
           "ishostkid": "hostages/kidnap victims"}
    rows = {}
    for c, lab in num.items():
        v = pd.to_numeric(df[c], errors="coerce")
        blank, code = v.isna(), v.isin(SENT)
        unk = blank | code
        known = v[~unk]
        r = {"label": lab, "blank": int(blank.sum()), "gtd_unknown_code": int(code.sum()),
             "unknown_total": int(unk.sum()), "unknown_pct": 100 * float(unk.mean()),
             "known_zero_pct": 100 * float((known == 0).mean()),
             "corpus_mean_known": float(known.mean()), "corpus_median_known": float(known.median()),
             # zero-fill: every unknown becomes "0"; code-as-value: -9/-99 read as numbers
             "zero_fill_rewrites_unknown_as_zero": int(unk.sum()),
             "code_as_value_negative_entries": int(code.sum()),
             "missingness_vs_label_cramers_v": cramers_v(unk.astype(int), y)}
        # share of events recorded as "0" after zero-fill vs among known values
        zf = v.where(~unk, 0)
        r["zero_pct_after_zero_fill"] = 100 * float((zf == 0).mean())
        rows[c] = r
    out["numeric"] = rows

    # ---------- per-class unknown rate of fatalities and property damage
    per_class = {}
    for c in ("nkill", "property"):
        v = pd.to_numeric(df[c], errors="coerce")
        unk = v.isna() | v.isin(SENT)
        per_class[c] = {cl: 100 * float(unk[y == cl].mean()) for cl in y.unique()}
    out["unknown_rate_by_class"] = per_class

    # ---------- categorical identity/location fields: mode-fill would assign the modal value
    cat = {"gname": "perpetrator group", "city": "city", "provstate": "province/state",
           "corp1": "target entity"}
    crow = {}
    for c, lab in cat.items():
        s = df[c].astype("string").str.strip()
        blank = s.isna() | s.eq("")
        unkn = s.str.lower().eq("unknown").fillna(False)
        known = s[~blank & ~unkn]
        mode = known.mode().iloc[0] if len(known) else None
        crow[c] = {"label": lab, "blank": int(blank.sum()), "coded_unknown": int(unkn.sum()),
                   "unknown_pct": 100 * float((blank | unkn).mean()),
                   "mode_of_known": None if mode is None else str(mode),
                   "mode_fill_would_assign": int((blank | unkn).sum()),
                   "mode_known_share_pct": 100 * float((known == mode).mean()) if mode is not None else None}
    out["categorical"] = crow

    # ---------- temporal profile
    yr = df.groupby([df["iyear"], y]).size().unstack(fill_value=0)
    out["events_per_year_min_max"] = [int(yr.sum(1).min()), int(yr.sum(1).max())]
    out["peak_year"] = int(yr.sum(1).idxmax())
    out["years"] = [int(yr.index.min()), int(yr.index.max())]
    (ROOT / "artifacts/tables/gtd_imputation_analysis.json").write_text(json.dumps(out, indent=2))

    # ---------- figure: (a) events per year by attack type, (b) unknown rate by class
    order = y.value_counts().index.tolist()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.35, 1]})
    cols = ["#08306b", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#E69F00", "#009E73",
            "#CC79A7", "#999999"]
    a1.stackplot(yr.index, *[yr[c] / 1000 for c in order], labels=[SHORT[c] for c in order],
                 colors=cols[:len(order)], lw=0)
    a1.set_xlabel("year"); a1.set_ylabel("events (thousands)")
    a1.set_title("(a) Corpus events per year by attack type", loc="left")
    a1.legend(fontsize=6, ncols=2, loc="upper left", frameon=False)
    mp.despine(a1)
    cls = order[::-1]
    yy = np.arange(len(cls))
    h = 0.38
    a2.barh(yy + h / 2, [per_class["nkill"][c] for c in cls], h, color=mp.BLUE, label="fatalities")
    a2.barh(yy - h / 2, [per_class["property"][c] for c in cls], h, color=mp.AMBER,
            label="property damage")
    a2.set_yticks(yy, [SHORT[c] for c in cls])
    a2.set_xlabel("unknown in GTD (% of events)")
    a2.set_title("(b) Unknown sensitive fields by class", loc="left")
    a2.legend(fontsize=6.5, frameon=False, loc="lower right")
    a2.xaxis.grid(True, color=mp.GRID, lw=0.6); a2.set_axisbelow(True)
    mp.despine(a2)
    fig.tight_layout()
    mp.save(fig, "fig_gtd_analysis")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
