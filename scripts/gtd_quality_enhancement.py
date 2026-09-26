"""GTD dataset-quality enhancement report (descriptive; no model is trained here).

Operations measured on the 143,586-event usable-narrative corpus:
  1. day recovered from the event identifier (Tier A; counts from tier_a_report.json)
  2. GTD unknown codes (-9 / -99) normalised to missing, per numeric field
  3. categorical gaps kept as an explicit "Unknown" (never mode-filled)
  4. city recovered from coordinates: nearest GTD record with a known city in the same country,
     accepted only within MAX_KM; validated by hiding the city of known records. Descriptive data
     quality only: city is not a model input.
  5. weapon sub-type filled from weapon type ("<type> (unspecified)")
  6. verbatim duplicate summaries (sibling records of multi-part incidents)
Writes artifacts/tables/gtd_quality_enhancement.json.
    ~/env314/bin/python scripts/gtd_quality_enhancement.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = pathlib.Path(__file__).resolve().parents[1]
SENT = [-9, -99]
MAX_KM = 0.1   # chosen on the hold-out check: 84% correct at 72% coverage (10 km: 70%)
R_EARTH = 6371.0


def unknown_city(s):
    s = s.astype("string").str.strip()
    return s.isna() | s.eq("") | s.str.lower().eq("unknown").fillna(False)


def nearest_city(ref, qry):
    """For each query row, the city of the nearest reference record in the SAME country."""
    out_city = pd.Series(pd.NA, index=qry.index, dtype="string")
    out_km = pd.Series(np.nan, index=qry.index)
    for country, q in qry.groupby("country_txt"):
        r = ref[ref["country_txt"] == country]
        if r.empty:
            continue
        tree = BallTree(np.radians(r[["latitude", "longitude"]].values), metric="haversine")
        d, i = tree.query(np.radians(q[["latitude", "longitude"]].values), k=1)
        out_city.loc[q.index] = r["city"].values[i[:, 0]]
        out_km.loc[q.index] = d[:, 0] * R_EARTH
    return out_city, out_km


def main():
    raw = pd.read_pickle(pathlib.Path.home() / "TerrorismNER_Project/checkpoints/gtd_full.pkl")
    raw.columns = [c.lower() for c in raw.columns]
    raw["eventid"] = raw["eventid"].astype("int64").astype(str)
    ids = set(pd.read_parquet(ROOT / "data/processed/gtd.parquet", columns=["eventid"])["eventid"].astype(str))
    df = raw[raw["eventid"].isin(ids)].copy()
    rep = {"n_events": len(df)}

    ta = json.loads((ROOT / "artifacts/preprocess/tier_a_report.json").read_text())
    rep["day_recovered_from_eventid"] = ta["day_recovered_from_eventid"]
    rep["month_recovered_from_eventid"] = ta["month_recovered_from_eventid"]
    rep["raw_events"] = ta["raw_rows"]
    rep["events_without_usable_narrative"] = ta["events_without_usable_text"]

    codes = {}
    for c in ["nkill", "nwound", "property", "ishostkid", "nhostkid", "nperps", "nperpcap",
              "propvalue", "ransomamt", "nreleased"]:
        if c in df:
            codes[c] = int(pd.to_numeric(df[c], errors="coerce").isin(SENT).sum())
    rep["unknown_codes_normalised"] = codes
    rep["unknown_codes_total"] = int(sum(codes.values()))

    cats = {}
    for c in ["gname", "city", "provstate", "targtype1_txt", "weaptype1_txt", "corp1", "target1"]:
        s = df[c].astype("string").str.strip()
        cats[c] = {"blank": int((s.isna() | s.eq("")).sum()),
                   "coded_unknown": int(s.str.lower().eq("unknown").fillna(False).sum())}
    rep["categorical_kept_unknown"] = cats

    # ---- city recovery from coordinates (GTD-internal, same-country nearest known city)
    has_ll = df["latitude"].notna() & df["longitude"].notna()
    known = df[has_ll & ~unknown_city(df["city"])]
    target = df[has_ll & unknown_city(df["city"])]
    city, km = nearest_city(known, target)
    ok = km <= MAX_KM
    rep["city_unknown"] = int(unknown_city(df["city"]).sum())
    rep["city_unknown_with_coordinates"] = int(len(target))
    rep["city_recovered_within_km"] = MAX_KM
    rep["city_recovered"] = int(ok.sum())
    rep["city_recovered_distance_km_median"] = float(km[ok].median()) if ok.any() else None

    # validation: hide the city of 10,000 known records; reference = all OTHER known records
    rng = np.random.default_rng(42)
    val_idx = rng.choice(known.index.values, size=min(10000, len(known)), replace=False)
    val = known.loc[val_idx]
    ref = known.drop(index=val_idx)
    vc, vkm = nearest_city(ref, val)
    norm = lambda s: s.astype("string").str.lower().str.replace(r"[^a-z]", "", regex=True)
    acc_all = (norm(vc) == norm(val["city"])) & (vkm <= MAX_KM)
    within = vkm <= MAX_KM
    rep["city_validation"] = {
        "n": int(len(val)), "accepted_within_km": int(within.sum()),
        "accuracy_on_accepted_pct": 100 * float(acc_all[within].mean()),
        "coverage_pct": 100 * float(within.mean())}

    ws = df["weapsubtype1_txt"].astype("string")
    rep["weapon_subtype_filled_from_type"] = int((ws.isna() | ws.str.strip().eq("")).sum())

    s = df["summary"].astype(str).str.strip()
    dup = s.duplicated(keep=False)
    rep["events_sharing_verbatim_summary"] = int(dup.sum())
    rep["events_sharing_verbatim_summary_pct"] = 100 * float(dup.mean())
    rep["multi_part_incident_share_of_those_pct"] = 100 * float((pd.to_numeric(df.loc[dup, "multiple"], errors="coerce") == 1).mean())

    (ROOT / "artifacts/tables/gtd_quality_enhancement.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))




# ---------------------------------------------------------------- before/after figure
def figure():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import plot_style as mp
    plt = mp.plt
    raw = pd.read_pickle(pathlib.Path.home() / "TerrorismNER_Project/checkpoints/gtd_full.pkl")
    raw.columns = [c.lower() for c in raw.columns]
    raw["eventid"] = raw["eventid"].astype("int64").astype(str)
    ids = set(pd.read_parquet(ROOT / "data/processed/gtd.parquet", columns=["eventid"])["eventid"].astype(str))
    df = raw[raw["eventid"].isin(ids)].copy()
    rep = json.loads((ROOT / "artifacts/tables/gtd_quality_enhancement.json").read_text())

    fields = {"nperps": "perpetrators", "propvalue": "property value", "property": "property flag",
              "nreleased": "hostages released", "nperpcap": "perpetrators captured",
              "nhostkid": "hostages", "ishostkid": "hostage flag"}
    means = {}
    for c, lab in fields.items():
        v = pd.to_numeric(df[c], errors="coerce")
        means[c] = {"before_mean_codes_as_values": float(v.mean()),
                    "after_mean_known_only": float(v[~v.isin(SENT)].mean())}

    # validation curve
    has_ll = df["latitude"].notna() & df["longitude"].notna()
    known = df[has_ll & ~unknown_city(df["city"])]
    rng = np.random.default_rng(42)
    vi = rng.choice(known.index.values, size=10000, replace=False)
    val, ref = known.loc[vi], known.drop(index=vi)
    vc, vkm = nearest_city(ref, val)
    norm = lambda s: s.astype("string").str.lower().str.replace(r"[^a-z]", "", regex=True)
    hit = norm(vc) == norm(val["city"])
    th = [0.01, 0.1, 0.5, 1, 2, 5, 10]
    curve = [(t, 100 * float((vkm <= t).mean()), 100 * float(hit[vkm <= t].mean())) for t in th]

    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.25), gridspec_kw={"width_ratios": [1.25, 1.1, 0.9, 1.0]})
    # (a) codes read as numbers, before vs after
    codes = {k: v for k, v in rep["unknown_codes_normalised"].items() if v > 0}
    order = sorted(codes, key=codes.get)
    y = np.arange(len(order))
    ax[0].barh(y, [codes[c] / 1000 for c in order], color=mp.AMBER, label="before")
    ax[0].plot([0] * len(order), y, "o", color=mp.BLUE, ms=3.5, label="after")
    nice = {"nperps": "perpetrators", "propvalue": "property value", "property": "property flag",
            "nreleased": "released", "nperpcap": "captured", "nhostkid": "hostages",
            "ishostkid": "hostage flag", "ransomamt": "ransom"}
    ax[0].set_yticks(y, [nice.get(c, c) for c in order], fontsize=6.3)
    ax[0].set_xlabel("codes read as numbers (k)", fontsize=6.8)
    ax[0].set_title("(a) Unknown codes", loc="left", fontsize=7.5)
    ax[0].legend(fontsize=6, frameon=False, loc="lower right")
    # (b) distortion of means
    mo = ["nperps", "nreleased"]
    x = np.arange(len(mo)); w = 0.38
    ax[1].bar(x - w / 2, [means[c]["before_mean_codes_as_values"] for c in mo], w, color=mp.AMBER, label="before")
    ax[1].bar(x + w / 2, [means[c]["after_mean_known_only"] for c in mo], w, color=mp.BLUE, label="after")
    ax[1].axhline(0, color=mp.MUTED, lw=0.7)
    ax[1].set_xticks(x, ["perpetrators", "hostages\nreleased"], fontsize=6.3)
    ax[1].set_ylabel("mean per event", fontsize=6.8)
    ax[1].set_title("(b) Field means", loc="left", fontsize=7.5)
    ax[1].legend(fontsize=6, frameon=False)
    # (c) unknown city before/after
    cu, rec = rep["city_unknown"], rep["city_recovered"]
    nocoord = cu - rep["city_unknown_with_coordinates"]
    left = rep["city_unknown_with_coordinates"] - rec
    ax[2].bar([0], [cu], color=mp.AMBER, width=0.6)
    ax[2].bar([1], [nocoord], color="#999999", width=0.6, label="no coordinates")
    ax[2].bar([1], [left], bottom=[nocoord], color=mp.AMBER, alpha=0.55, width=0.6, label="no record ≤0.1 km")
    ax[2].bar([2], [rec], color=mp.BLUE, width=0.6, label="recovered")
    ax[2].set_xticks([0, 1, 2], ["before", "after", "recov."], fontsize=6.3)
    ax[2].set_ylabel("unknown city (events)", fontsize=6.8)
    ax[2].set_title("(c) City", loc="left", fontsize=7.5)
    ax[2].legend(fontsize=5.4, frameon=False, loc="upper center", bbox_to_anchor=(0.55, 1.0))
    ax[2].set_ylim(0, cu * 1.45)
    # (d) validation trade-off
    cov = [c[1] for c in curve]; acc = [c[2] for c in curve]
    ax[3].plot(cov, acc, "-o", color=mp.BLUE, ms=3)
    for t, cv, ac in curve:
        if t in (0.1, 1, 10):
            ax[3].annotate(f"{t:g} km", (cv, ac), textcoords="offset points", xytext=(3, 3), fontsize=5.8)
    sel = [c for c in curve if c[0] == 0.1][0]
    ax[3].plot(sel[1], sel[2], "s", color=mp.GREEN, ms=6, zorder=5)
    ax[3].set_xlabel("coverage (%)", fontsize=6.8); ax[3].set_ylabel("correct city (%)", fontsize=6.8)
    ax[3].set_title("(d) Recovery check", loc="left", fontsize=7.5)
    for a in ax:
        a.tick_params(labelsize=6.3)
        mp.despine(a)
    fig.tight_layout(w_pad=0.6)
    mp.save(fig, "fig_quality_before_after")
    extra = {"field_means": means, "city_validation_curve": [
        {"km": t, "coverage_pct": cv, "accuracy_pct": ac} for t, cv, ac in curve]}
    p = ROOT / "artifacts/tables/gtd_quality_enhancement.json"
    j = json.loads(p.read_text()); j.update(extra); p.write_text(json.dumps(j, indent=2))
    print(json.dumps(extra, indent=2))


if __name__ == "__main__":
    import sys as _s
    figure() if "--figure" in _s.argv else main()
