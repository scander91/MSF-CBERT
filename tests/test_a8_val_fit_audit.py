"""A8 negative test — prove the val-fit audit check actually fails on a leak.

An audit check that has never failed is not evidence of anything. This plants
artifacts in a throwaway runs/ dir and asserts A8's verdict AND the audit's exit
code on each:

  1. valid         fitted on val, correct fingerprint                 -> PASS
  2. valid_8class  fitted on the label-scheme-filtered val subset     -> PASS
  3. test-fit       fitted on TEST (the leak this exists to catch)     -> FAIL
  4. forged         claims "val" but carries a wrong fingerprint       -> FAIL
  5. no_provenance  post-hoc artifact with no posthoc_fit block        -> FAIL
  6. no_split       provenance block that declares no split            -> FAIL
  7. unknown_split  provenance names a split with no manifest          -> FAIL
  8. split_mismatch args and provenance name different splits          -> FAIL
  9. wrong_count    fingerprint is right but n_ids is false            -> FAIL

Cases 5 and 6 are the fail-closed cases: before they were covered, A8 silently
examined zero artifacts on the real tree and still printed PASS.

A FAIL must also make the audit exit non-zero — that is what makes it a gate
rather than a log line.

Usage:  python tests/test_a8_val_fit_audit.py --split seed_42
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.util import load_config, read_manifest, partition_fingerprint  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plant(runs_dir, name, payload):
    d = os.path.join(runs_dir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2)


def artifact(split, scheme="with_unknown", fit=None, base_run="base"):
    p = {"run": "unit-test", "base_run": base_run, "labels": [],
         "label_scheme": scheme, "args": {"split": split}}
    if fit is not None:
        p["posthoc_fit"] = fit
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="seed_42")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    a = ap.parse_args()

    os.chdir(ROOT)
    cfg = load_config(a.config)
    man = read_manifest(cfg, a.split)
    ids = {s: set(man.loc[man.split == s, "eventid"]) for s in ("train", "val", "test")}

    # the 8-class val subset, exactly as train.py would produce it
    import pandas as pd
    df = pd.read_parquet(cfg["paths"]["processed"])[["eventid", "label"]]
    df["eventid"] = df["eventid"].astype(str)
    drops = cfg["data"]["label_schemes"]["without_unknown"]
    dropped = set(df.loc[df["label"].isin(drops), "eventid"])
    val8 = {str(i) for i in ids["val"]} - dropped

    def fp(s):
        return partition_fingerprint(sorted({str(i) for i in s}))

    cases = [
        ("a8_valid", "PASS", artifact(a.split, fit={
            "partition": "val", "fingerprint": fp(ids["val"]),
            "n_ids": len(ids["val"]), "method": "unit-test", "split": a.split})),
        ("a8_valid8", "PASS", artifact(a.split, scheme="without_unknown", fit={
            "partition": "val", "fingerprint": fp(val8),
            "n_ids": len(val8), "method": "unit-test", "split": a.split})),
        ("a8_testfit", "FAIL", artifact(a.split, fit={
            "partition": "test", "fingerprint": fp(ids["test"]),
            "n_ids": len(ids["test"]), "method": "unit-test", "split": a.split})),
        ("a8_forged", "FAIL", artifact(a.split, fit={
            "partition": "val", "fingerprint": "0" * 64,
            "n_ids": len(ids["val"]), "method": "unit-test", "split": a.split})),
        ("a8_noprov", "FAIL", artifact(a.split, fit=None)),
        ("a8_nosplit", "FAIL", {"run": "unit-test", "base_run": "base", "labels": [],
                                "label_scheme": "with_unknown",
                                "posthoc_fit": {"partition": "val",
                                                "fingerprint": fp(ids["val"]),
                                                "n_ids": len(ids["val"]),
                                                "method": "unit-test"}}),
        ("a8_unknownsplit", "FAIL", artifact("not_a_real_split", fit={
            "partition": "val", "fingerprint": fp(ids["val"]),
            "n_ids": len(ids["val"]), "method": "unit-test",
            "split": "not_a_real_split"})),
        ("a8_splitmismatch", "FAIL", artifact(a.split, fit={
            "partition": "val", "fingerprint": fp(ids["val"]),
            "n_ids": len(ids["val"]), "method": "unit-test", "split": "chrono"})),
        ("a8_wrongcount", "FAIL", artifact(a.split, fit={
            "partition": "val", "fingerprint": fp(ids["val"]),
            "n_ids": len(ids["val"]) + 1, "method": "unit-test", "split": a.split})),
    ]

    tmp = tempfile.mkdtemp(prefix="a8_audit_")
    failures = []
    try:
        runs_dir = os.path.join(tmp, "runs")
        os.makedirs(runs_dir)
        cfg_path = os.path.join(tmp, "config_a8.yaml")
        import yaml
        cfg2 = load_config(a.config)
        cfg2["paths"]["runs_dir"] = runs_dir
        with open(cfg_path, "w") as fh:
            yaml.safe_dump(cfg2, fh)

        for name, expect, payload in cases:
            for stale in os.listdir(runs_dir):
                shutil.rmtree(os.path.join(runs_dir, stale))
            plant(runs_dir, name, payload)
            r = subprocess.run([sys.executable, "-m", "src.audit.leakage_audit",
                                "--config", cfg_path, "--split", a.split],
                               capture_output=True, text=True, cwd=ROOT)
            a8 = [l for l in r.stdout.splitlines() if f"A8.{name}" in l]
            got = "FAIL" if any("[FAIL]" in l for l in a8) else "PASS"
            # A verdict must exist, must match, and a FAIL must gate the pipeline.
            problems = []
            if not a8:
                problems.append("A8 produced no verdict (check never ran)")
            if got != expect:
                problems.append(f"expected {expect}, got {got}")
            if expect == "FAIL":
                if r.returncode == 0:
                    problems.append("audit exited 0 despite a FAIL — the gate is dead")
                if f"A8.{name}" not in r.stdout.split("AUDIT FAILED:")[-1]:
                    problems.append("check name absent from the AUDIT FAILED list")
            elif r.returncode != 0:
                problems.append(f"audit exited {r.returncode} on an valid artifact")

            print(f"  [{'ok' if not problems else 'BROKEN'}] {name:12s} "
                  f"expected={expect} got={got} exit={r.returncode}")
            for l in a8:
                print(f"        {l.strip()}")
            for pr in problems:
                failures.append(f"{name}: {pr}")
                print(f"        !! {pr}")
            if not a8 and r.stderr.strip():
                print(f"        stderr: {r.stderr.strip().splitlines()[-1]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("\nA8 NEGATIVE TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nA8 negative test passed: valid val-fits (both label schemes) pass; "
          "test-fits, forged fingerprints, bad counts, missing provenance, and invalid "
          "split declarations all FAIL and exit non-zero.")


if __name__ == "__main__":
    main()
