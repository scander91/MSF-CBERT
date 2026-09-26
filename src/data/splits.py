"""Step 01 — Create split manifests. THE single source of truth for partitions.

Emits, for each seed s:      artifacts/splits/seed_{s}.csv   (eventid,split)
and if enabled:              artifacts/splits/chrono.csv     (eventid,split)
plus                         artifacts/splits/support_report.md

Rules encoded here:
  * stratified by label, 70/15/15 (controlled within-distribution setting)
  * chronological split sorted by (date, eventid); the boundary NEVER splits a day,
    so max(train.date) < min(val.date) <= max(val.date) < min(test.date)  (E2)
  * manifests contain ORIGINAL events only — augmentation happens later, train-only.
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.util import load_config, log, ensure_dir


def stratified(df, seed, ratios):
    tr_ids, rest = train_test_split(df["eventid"], test_size=1 - ratios[0],
                                    stratify=df["label"], random_state=seed)
    rest_df = df[df["eventid"].isin(rest)]
    rel = ratios[2] / (ratios[1] + ratios[2])
    va_ids, te_ids = train_test_split(rest_df["eventid"], test_size=rel,
                                      stratify=rest_df["label"], random_state=seed)
    m = pd.Series("train", index=df["eventid"])
    m.loc[va_ids] = "val"; m.loc[te_ids] = "test"
    return m.rename("split").rename_axis("eventid").reset_index()


def chronological(df, ratios):
    df = df.sort_values(["date", "eventid"]).reset_index(drop=True)
    n = len(df)
    c1, c2 = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    # push each boundary forward so a calendar day is never split across partitions
    while c1 < n and df.loc[c1, "date"] == df.loc[c1 - 1, "date"]: c1 += 1
    while c2 < n and df.loc[c2, "date"] == df.loc[c2 - 1, "date"]: c2 += 1
    df["split"] = "train"
    df.loc[c1:c2 - 1, "split"] = "val"
    df.loc[c2:, "split"] = "test"
    assert df[df.split == "train"].date.max() < df[df.split == "val"].date.min()
    assert df[df.split == "val"].date.max() < df[df.split == "test"].date.min()
    return df[["eventid", "split"]]


def support_table(df, manifest, name):
    j = df.merge(manifest, on="eventid")
    t = j.groupby(["label", "split"]).size().unstack(fill_value=0)
    t = t[["train", "val", "test"]]
    return f"\n### {name}\n\n{t.to_markdown()}\n"


def main(cfg):
    p, s = cfg["paths"], cfg["splits"]
    df = pd.read_parquet(p["processed"])[["eventid", "label", "date"]]
    ensure_dir(p["splits_dir"])
    report = ["# Split support report\n",
              "Per-class support in every partition. The chronological table shows\n"
              "whether rare classes survive a temporal cut.\n"]
    for seed in s["seeds"]:
        m = stratified(df, seed, s["ratios"])
        m.to_csv(f"{p['splits_dir']}/seed_{seed}.csv", index=False)
        log(f"seed {seed}: " + ", ".join(f"{k}={v:,}" for k, v in m.split.value_counts().items()))
    report.append(support_table(df, m, f"Stratified (seed {s['seeds'][-1]}, representative)"))
    if s["chrono"]:
        cm = chronological(df, s["ratios"])
        cm.to_csv(f"{p['splits_dir']}/chrono.csv", index=False)
        j = df.merge(cm, on="eventid")
        for part in ["train", "val", "test"]:
            sub = j[j.split == part]
            log(f"chrono {part}: {len(sub):,} events, {sub.date.min().date()} .. {sub.date.max().date()}")
        report.append(support_table(df, cm, "Chronological"))
    open(f"{p['splits_dir']}/support_report.md", "w").write("\n".join(report))
    log(f"wrote support report -> {p['splits_dir']}/support_report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml")
    main(load_config(ap.parse_args().config))
