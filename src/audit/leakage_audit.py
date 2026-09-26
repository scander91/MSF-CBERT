"""Step 05 — LEAKAGE AUDIT (hard gate). Run before EVERY training campaign:

    python -m src.audit.leakage_audit --split seed_42
    python -m src.audit.leakage_audit --split chrono

Exit code 0 = all checks PASS. Anything else = do not train, do not report.
These checks are the executable form of the protocol's guarantees:

  A1  split manifest: partitions disjoint, every corpus event assigned exactly once
  A2  chrono manifest only: max(train.date) < min(val.date) <= ... < min(test.date)
  A3  augmentation: every augmented row's source eventid is in TRAIN; no augmented rows
      exist in val/test; accounting JSON internally consistent (monotone counts)
  A4  graph artifact: every stored neighbor eventid is a TRAIN event of THIS split
      (fingerprint match) -> "graph built from training partition only" is now a fact,
      not a sentence
  A5  feature-selection artifact fingerprint == sha256(train ids of THIS split)
      -> "Cramér's V computed on the training partition only" is verifiable
  A6  label-string scan: report (not fail) the rate at which the input text contains the
      event's own label string — context for the attribute-leakage discussion
  A7  Tier-B preprocessing artifact (D8): fingerprint == sha256(train ids of THIS split),
      no protected column was dropped, and (chrono only) max_train_date <= the manifest's
      train boundary — i.e. no Tier-B statistic saw any event after the boundary
  A8  post-hoc (val-fit) artifacts: any run whose metrics.json carries a `posthoc_fit`
      block — calibration temperature, per-class log-prob biases, any second-stage fit —
      was fitted on VALIDATION, never test, and its fingerprint matches this split's val
      ids. Without this, invariant 3 ("never tune on test") is convention, not a check.
"""
import argparse, glob, json, os, sys
import pandas as pd
from src.util import load_config, log, read_manifest, train_fingerprint, partition_fingerprint

FAIL = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    log(f"[{status}] {name} {detail}")
    if not cond:
        FAIL.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--split", required=True)
    a = ap.parse_args(); cfg = load_config(a.config); p = cfg["paths"]

    df = pd.read_parquet(p["processed"])
    man = read_manifest(cfg, a.split)
    ids = {s: set(man.loc[man.split == s, "eventid"]) for s in ["train", "val", "test"]}

    # A1 disjoint + coverage
    check("A1.manifest_eventids_unique", man["eventid"].is_unique,
          f"({int(man['eventid'].duplicated().sum()):,} duplicate rows)")
    check("A1.corpus_eventids_unique", df["eventid"].is_unique,
          f"({int(df['eventid'].duplicated().sum()):,} duplicate rows)")
    check("A1.disjoint", not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"])
          and not (ids["val"] & ids["test"]))
    check("A1.coverage", set(man["eventid"]) == set(df["eventid"]),
          f"(manifest {len(man):,} vs corpus {len(df):,})")

    # A2 chronological ordering
    if a.split == "chrono":
        j = df.merge(man, on="eventid")
        tmax = j[j.split == "train"].date.max(); vmin = j[j.split == "val"].date.min()
        vmax = j[j.split == "val"].date.max(); smin = j[j.split == "test"].date.min()
        check("A2.chrono_order", tmax < vmin and vmax < smin,
              f"(train<= {tmax.date()} | val {vmin.date()}..{vmax.date()} | test>= {smin.date()})")

    # A3 augmentation containment + accounting
    aug_path = f"{p['aug_dir']}/train_aug_{a.split}.parquet"
    try:
        ta = pd.read_parquet(aug_path)
        aug = ta[ta["is_augmented"] == 1]
        check("A3.sources_in_train", set(aug["source_eventid"]) <= ids["train"],
              f"({len(aug):,} augmented rows)")
        check("A3.no_aug_ids_in_heldout",
              not (set(aug["eventid"]) & (ids["val"] | ids["test"])))
        acc = json.load(open(f"{p['aug_dir']}/accounting_{a.split}.json"))
        mono = (acc["generated"] >= acc["sim_filtered"] >= acc["human_filtered"]
                >= acc["train_source_filtered"] == acc["final_added"]
                and acc["final_train_size"] == acc["original_train"] + acc["final_added"])
        check("A3.accounting_consistent", mono, str(acc))
    except FileNotFoundError:
        log("[SKIP] A3 (no augmentation artifact for this split)")

    # A4 graph neighbors are train-only
    try:
        import pickle
        g = pickle.load(open(f"{p['graph_dir']}/graph_{a.split}.pkl", "rb"))
        gids = set(map(str, g["train_eventids"]))
        check("A4.graph_nodes_are_train_only",
              gids == set(map(str, ids["train"])), f"({len(gids):,} graph events)")
        check("A4.graph_fingerprint",
              g["train_fingerprint"] == train_fingerprint(ids["train"]))
    except FileNotFoundError:
        log("[SKIP] A4 (no graph artifact for this split)")

    # A5 feature-selection fingerprint
    try:
        fs = json.load(open(f"{p['features_dir']}/selected_{a.split}.json"))
        check("A5.cramers_v_train_only",
              fs["train_fingerprint"] == train_fingerprint(ids["train"]),
              f"(selected: {fs['selected']})")
    except FileNotFoundError:
        log("[SKIP] A5 (no feature-selection artifact for this split)")

    # A7 Tier-B preprocessing fingerprint + chrono boundary (D8)
    try:
        tb = json.load(open(f"{p['preprocess_dir']}/tierb_{a.split}.json"))
        check("A7.preprocess_fingerprint",
              tb["train_fingerprint"] == train_fingerprint(ids["train"]),
              f"(fit on n_train={tb['n_train']:,})")
        dropped = [x["dropped"] for kind in tb["dropped"].values() for x in kind]
        prot = set(cfg["preprocess"]["protected_cols"])
        check("A7.protected_columns_intact", not (set(dropped) & prot),
              f"(dropped: {dropped})")
        if a.split == "chrono":
            j = df.merge(man, on="eventid")
            boundary = j[j.split == "train"].date.max()
            check("A7.chrono_no_future_stats",
                  pd.Timestamp(tb["max_train_date"]) <= boundary,
                  f"(tierb max date {tb['max_train_date']} vs boundary {boundary.date()})")
    except FileNotFoundError:
        log("[SKIP] A7 (no Tier-B preprocessing artifact for this split)")

    # A8 post-hoc (val-fit) artifact provenance.
    #
    # Fails CLOSED. An artifact that looks post-hoc (it derives from another run, or
    # declares a fit) but carries no verifiable provenance is a FAIL, not a skip —
    # otherwise A8 would silently examine nothing and still print PASS, which is worse
    # than having no check at all.
    runs_root = os.path.abspath(p["runs_dir"])
    log(f"[INFO] A8 scanning {runs_root}")
    n_posthoc = 0
    for mpath in sorted(glob.glob(os.path.join(runs_root, "*", "metrics.json"))):
        run = os.path.basename(os.path.dirname(mpath))
        try:
            m = json.load(open(mpath))
        except (json.JSONDecodeError, OSError) as e:
            check(f"A8.{run}.metrics_readable", False, f"({e})")
            continue
        if not isinstance(m, dict):
            check(f"A8.{run}.metrics_is_object", False, "(metrics.json is not an object)")
            continue

        fit = m.get("posthoc_fit")
        # `base_run` marks a run derived from another run's predictions, i.e. post-hoc.
        looks_posthoc = ("base_run" in m) or (fit is not None)
        if not looks_posthoc:
            continue

        args_blk = m.get("args") if isinstance(m.get("args"), dict) else {}
        args_split = args_blk.get("split")
        fit_split = fit.get("split") if isinstance(fit, dict) else None
        if args_split and fit_split and args_split != fit_split:
            check(f"A8.{run}.split_consistent", False,
                  f"(args declares {args_split!r}, provenance declares {fit_split!r})")
            continue
        declared = args_split or fit_split
        if not declared:
            # Cannot tell which split it belongs to -> cannot verify it -> FAIL.
            check(f"A8.{run}.split_declared", False,
                  "(post-hoc artifact does not declare the split it was fitted on)")
            continue
        manifest_path = os.path.join(p["splits_dir"], f"{declared}.csv")
        if not os.path.isfile(manifest_path):
            check(f"A8.{run}.split_known", False,
                  f"(declared split {declared!r} has no manifest)")
            continue

        n_posthoc += 1
        if not isinstance(fit, dict):
            check(f"A8.{run}.has_provenance", False,
                  "(post-hoc artifact carries no posthoc_fit provenance block)")
            continue

        part = fit.get("partition")
        check(f"A8.{run}.fit_partition_is_val", part == "val",
              f"(fitted on {part!r}; fitting on test is a leak)")
        if part in ("train", "val", "test"):
            run_man = man if declared == a.split else read_manifest(cfg, declared)
            run_ids = {s: set(run_man.loc[run_man.split == s, "eventid"])
                       for s in ("train", "val", "test")}
            # The expected id set must respect the run's label scheme: an 8-class run's
            # preds_val.csv is a strict subset of the manifest's val partition, so
            # hashing the full partition would fail a legitimate run (D4).
            scheme = m.get("label_scheme", "with_unknown")
            check(f"A8.{run}.label_scheme_known",
                  scheme in cfg["data"]["label_schemes"], f"(scheme={scheme!r})")
            drops = cfg["data"]["label_schemes"].get(scheme, [])
            dropped = set(df.loc[df["label"].isin(drops), "eventid"].astype(str)) if drops else set()
            expected = {str(i) for i in run_ids[part]} - dropped
            check(f"A8.{run}.fit_fingerprint",
                  fit.get("fingerprint") == partition_fingerprint(expected),
                  f"(split={declared}, scheme={scheme}, expected n={len(expected)}, "
                  f"recorded n={fit.get('n_ids')}, method={fit.get('method')})")
            check(f"A8.{run}.fit_count", fit.get("n_ids") == len(expected),
                  f"(expected n={len(expected)}, recorded n={fit.get('n_ids')})")
    log(f"[INFO] A8 examined {n_posthoc} post-hoc artifact(s) across declared splits")

    # A6 label-string scan (report only)
    sample = df.sample(min(20000, len(df)), random_state=0)
    rate = (sample.apply(lambda r: str(r["label"]).lower() in str(r["text"]).lower(),
                         axis=1)).mean()
    log(f"[INFO] A6.label_string_in_summary_rate = {rate:.3%} (informational, not a failure)")

    if FAIL:
        log(f"AUDIT FAILED: {FAIL}"); sys.exit(1)
    log("AUDIT PASSED — clear to train.")


if __name__ == "__main__":
    main()
