"""Step 00 — Load raw GTD and apply TIER-A preprocessing.

Tier A = row-local + deterministic ONLY. Nothing here depends on any other row,
so it runs ONCE before splitting without leakage risk. Distribution-dependent
steps (imputation values, bins, redundancy elimination) are Tier B:
src/data/preprocess_tierb.py, train-fit per split, fingerprinted, audited (A7).

Tier-A steps (each row-local, each deterministic):
  1. date recovery: yyyymmdd prefix of eventid when imonth/iday == 0; clamp fallback
  2. fixed-value marker fills ("Unknown") for categorical structured columns
  3. row-local numerics: sentinel (-9/-99) -> NaN; casualties = nkill + nwound;
     weapsubtype1_txt filled from weaptype1_txt ("<weaptype> (unspecified)")
  4. text: whitespace-normalized summary; usable = len >= min_summary_chars;
     NO text imputation — unusable-text rows are dropped from the corpus and the
     before/after per-class support is REPORTED
  5. deduplication on eventid
Labels: ALL 9 attack types kept, incl. "Unknown" — label schemes are a runtime
axis (D4) so both schemes share the same split manifests.

Deterministic: same input file -> byte-identical output.
"""
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd
from src.util import load_config, log, ensure_dir

# GTD sentinel codes meaning "unknown" in numeric columns (Tier A: -> NaN, Tier B imputes)
SENTINELS = {-9, -99}


def recover_dates(df, d):
    """Tier A-1. GTD uses 0 for unknown month/day. The eventid prefix encodes yyyymmdd
    of the incident record; use it to recover month/day when flagged 0, else clamp to 1."""
    y = df[d["date_cols"]["year"]].astype(int)
    m = df[d["date_cols"]["month"]].astype(int)
    dy = df[d["date_cols"]["day"]].astype(int)

    eid = df["eventid"].astype(str).str.slice(0, 8)
    ok8 = eid.str.fullmatch(r"\d{8}")
    em = pd.to_numeric(eid.str.slice(4, 6), errors="coerce").where(ok8)
    ed = pd.to_numeric(eid.str.slice(6, 8), errors="coerce").where(ok8)
    ey = pd.to_numeric(eid.str.slice(0, 4), errors="coerce").where(ok8)

    # recover only where the recorded field is 0, the eventid year agrees, and the
    # recovered value is a plausible calendar value
    rec_m = (m == 0) & (ey == y) & em.between(1, 12)
    rec_d = (dy == 0) & (ey == y) & ed.between(1, 31)
    m = m.mask(rec_m, em).clip(lower=1)
    dy = dy.mask(rec_d, ed).clip(lower=1)

    date = pd.to_datetime(dict(year=y, month=m, day=dy), errors="coerce")
    # invalid combos like Feb 30 after recovery: clamp day to 1 (deterministic fallback)
    bad = date.isna()
    if bad.any():
        date = date.mask(bad, pd.to_datetime(dict(year=y, month=m.where(~bad, m.clip(upper=12)),
                                                  day=dy.where(~bad, 1)), errors="coerce"))
    stats = {"month_recovered_from_eventid": int(rec_m.sum()),
             "day_recovered_from_eventid": int(rec_d.sum()),
             "date_unparseable_dropped": int(date.isna().sum())}
    return date, stats


def support_table(series):
    t = series.value_counts()
    return {str(k): int(v) for k, v in t.items()}


def main(cfg):
    p, d = cfg["paths"], cfg["data"]
    log(f"reading {p['gtd_raw']}")
    if str(p["gtd_raw"]).endswith((".xlsx", ".xls")):
        df = pd.read_excel(p["gtd_raw"])
    else:
        df = pd.read_csv(p["gtd_raw"], encoding="latin-1", low_memory=False)
    df.columns = [c.lower() for c in df.columns]      # GTD mixes case (INT_ANY etc.)
    n_raw = len(df)
    log(f"raw export: {n_raw:,} rows x {len(df.columns)} columns")

    need = [d["id_col"], d["label_col"], *d["text_cols"],
            d["date_cols"]["year"], d["date_cols"]["month"], d["date_cols"]["day"],
            *d["structured_cols"], "weapsubtype1_txt", "nkill", "nwound",
            "success", "suicide", "extended", "multiple", "crit1", "crit2", "crit3",
            "vicinity", "ishostkid", "property", "int_any"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"FATAL: raw GTD is missing columns: {missing}")
    df = df[list(dict.fromkeys(need))].copy()
    df = df.rename(columns={d["id_col"]: "eventid", d["label_col"]: "label"})
    df["eventid"] = df["eventid"].astype(str)          # canonical id type everywhere

    report = {"raw_rows": n_raw}

    # ---- Tier A-5 (early, so every later count is per unique event) ----
    df = df.drop_duplicates(subset="eventid").reset_index(drop=True)
    report["after_dedup"] = len(df)

    # ---- Tier A-1: dates ----
    df["date"], dstats = recover_dates(df, d)
    report.update(dstats)
    df = df.dropna(subset=["date"])

    # ---- labels: keep ALL 9 (D4); drop only rows with no label at all ----
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(str)
    report["after_date_and_label"] = len(df)

    # ---- Tier A-2: categorical marker fills ----
    for col in d["structured_cols"]:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA).fillna("Unknown").astype(str)

    # ---- Tier A-3: row-local numerics ----
    for col in ["nkill", "nwound", "vicinity", "ishostkid", "property", "int_any"]:
        v = pd.to_numeric(df[col], errors="coerce")
        df[col] = v.mask(v.isin(list(SENTINELS)))
    df["casualties"] = df["nkill"] + df["nwound"]       # NaN-propagating, row-local
    for col in ["success", "suicide", "extended", "multiple", "crit1", "crit2", "crit3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    ws = df["weapsubtype1_txt"].astype("string").str.strip().replace("", pd.NA)
    df["weapsubtype1_txt"] = ws.fillna(df["weaptype1_txt"] + " (unspecified)").astype(str)

    # ---- Tier A-4: text (report BEFORE/AFTER per-class support) ----
    df["text"] = df[d["text_cols"]].fillna("").astype(str).agg(" ".join, axis=1) \
                                   .str.replace(r"\s+", " ", regex=True).str.strip()
    report["support_before_text_filter"] = support_table(df["label"])
    usable = df["text"].str.len() >= d["min_summary_chars"]
    report["events_without_usable_text"] = int((~usable).sum())
    df = df[usable].reset_index(drop=True)
    report["support_after_text_filter"] = support_table(df["label"])
    report["true_usable_N"] = len(df)

    # per-scheme corpus sizes (D4)
    for scheme, drops in d["label_schemes"].items():
        sub = df[~df["label"].isin(drops)]
        report[f"corpus_{scheme}"] = {"events": len(sub), "classes": int(sub["label"].nunique())}

    ensure_dir("data/processed")
    df.to_parquet(p["processed"], index=False)
    fp = hashlib.sha256(pd.util.hash_pandas_object(df["eventid"]).values.tobytes()).hexdigest()[:16]
    report["corpus_fingerprint"] = fp

    ensure_dir(p["preprocess_dir"])
    json.dump(report, open(f"{p['preprocess_dir']}/tier_a_report.json", "w"), indent=2)
    log(f"wrote {len(df):,} events, {df['label'].nunique()} classes -> {p['processed']}")
    log("class support (after text filter):\n"
        + "\n".join(f"  {k}: {v:,}" for k, v in report["support_after_text_filter"].items()))
    log(f"corpus fingerprint: {fp}")
    log(f"Tier-A report -> {p['preprocess_dir']}/tier_a_report.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml")
    main(load_config(ap.parse_args().config))
