import hashlib, multiprocessing, os, sys, time
import pandas as pd, yaml

# Python 3.14 changed the default multiprocessing start method on Linux from 'fork'
# to 'forkserver'. DataLoader worker processes then have to pickle their collate_fn,
# which is a closure here (train.py:make_collate) and cannot be pickled. Every run in
# the existing matrix was produced under 'fork' on 3.13, so restore it explicitly:
# this keeps worker semantics identical to the recorded runs rather than silently
# changing them. Safe here because the workers only touch CPU tensors.
if multiprocessing.get_start_method(allow_none=True) != "fork":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):  # pragma: no cover - platform without fork
        pass


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def read_manifest(cfg, split_name):
    """split_name: 'seed_42' or 'chrono' -> DataFrame(eventid, split)."""
    path = os.path.join(cfg["paths"]["splits_dir"], f"{split_name}.csv")
    if not os.path.exists(path):
        sys.exit(f"FATAL: missing split manifest {path} — run src.data.splits first")
    m = pd.read_csv(path)
    m["eventid"] = m["eventid"].astype(str)   # canonical id type
    assert set(m["split"].unique()) <= {"train", "val", "test"}
    return m


def train_fingerprint(train_ids):
    ids = sorted(str(i) for i in train_ids)
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


# Same hash, named for the general case: any artifact fitted on ANY partition records
# the fingerprint of the id set it saw. Train-fit artifacts are verified by A5/A7,
# val-fit (post-hoc calibration, per-class biases) by A8.
partition_fingerprint = train_fingerprint


def posthoc_fit_record(partition, ids, method):
    """Provenance block every post-hoc (val-fit) artifact must carry, so A8 can verify
    which partition it was fitted on. Fitting on test is a leak that A8 rejects.

    Ids are de-duplicated before hashing so this agrees with the audit side, which
    derives its expected fingerprint from a set. Without this the two sides would only
    match by luck (they differ the moment a prediction file contains a repeated id)."""
    ids = sorted({str(i) for i in ids})
    return {"partition": partition, "fingerprint": partition_fingerprint(ids),
            "n_ids": len(ids), "method": method}
