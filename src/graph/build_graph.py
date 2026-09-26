"""Step 04 + runtime — Heterogeneous event–entity graph, TRAIN-ONLY by construction (E1, E2).

Design:

  * Entity vocabularies and all event–entity edges are built from the TRAIN partition of
    one specific split manifest. Held-out events are NEVER nodes in the stored graph.
  * At inference, a held-out event is ATTACHED, not inserted: its non-label metadata is
    matched against the train-built entity vocabulary; it RECEIVES decay-weighted messages
    from train events sharing those entities and SENDS nothing. Entities absent from the
    train vocabulary fall back to a learned UNK vector in the model. No test–test edges
    exist anywhere.
  * `past_only=True` additionally masks train neighbors dated ON/AFTER the query event —
    the deployable, online setting. Report both modes.
  * Temporal decay: w = exp(-ln(2)/half_life * |t_query - t_neighbor|) days.

Artifact: artifacts/graph/graph_{split}.pkl with a train-ID fingerprint the audit verifies.
"""
import argparse, pickle
import numpy as np, pandas as pd
from src.util import load_config, log, ensure_dir, read_manifest, train_fingerprint


def build(cfg, split_name):
    p, g = cfg["paths"], cfg["graph"]
    df = pd.read_parquet(p["processed"])
    man = read_manifest(cfg, split_name)
    train_ids = man.loc[man.split == "train", "eventid"]
    tr = df[df["eventid"].isin(set(train_ids))].copy()
    tr = tr.sort_values("eventid").reset_index(drop=True)      # canonical train row order
    tr["ord"] = tr["date"].map(pd.Timestamp.toordinal)

    excl = set(g["exclude_entity_values"])
    index = {}                                                  # rel -> {entity: (idx[], ord[])}
    for rel, col in g["relations"].items():
        rel_index = {}
        vals = tr[col].fillna("").astype(str)
        for ent, grp in tr.groupby(vals):
            if ent in excl or ent == "":
                continue
            order = np.argsort(grp["ord"].values, kind="stable")
            rel_index[ent] = (grp.index.values[order].astype(np.int64),
                              grp["ord"].values[order].astype(np.int64))
        index[rel] = rel_index
        log(f"[{split_name}] relation '{rel}' ({col}): {len(rel_index):,} train entities")

    art = {"split": split_name,
           "train_eventids": tr["eventid"].tolist(),           # row i <-> this eventid
           "index": index,
           "half_life_days": g["half_life_days"],
           "max_neighbors": g["max_neighbors_per_entity"],
           "train_fingerprint": train_fingerprint(train_ids)}
    ensure_dir(p["graph_dir"])
    out = f"{p['graph_dir']}/graph_{split_name}.pkl"
    pickle.dump(art, open(out, "wb"))
    log(f"wrote {out} ({len(tr):,} train events)")


class GraphContext:
    """Runtime neighbor lookup. Used identically for train (self excluded) and held-out
    events (attach-only). The ONLY difference past_only makes is a strict date mask."""

    def __init__(self, artifact_path):
        a = pickle.load(open(artifact_path, "rb"))
        self.index = a["index"]
        self.lam = np.log(2.0) / a["half_life_days"]
        self.k = a["max_neighbors"]
        self.train_eventids = a["train_eventids"]
        self.fingerprint = a["train_fingerprint"]

    def neighbors(self, rel, entity, ref_ordinal, past_only=False, self_row=-1):
        """-> (train_row_indices, decay_weights) or (None, None) if entity unseen in train."""
        ent = self.index[rel].get(str(entity))
        if ent is None:
            return None, None
        idx, ords = ent
        if past_only:
            cut = np.searchsorted(ords, ref_ordinal, side="left")   # strictly before query
            idx, ords = idx[:cut], ords[:cut]
        if self_row >= 0:
            keep = idx != self_row
            idx, ords = idx[keep], ords[keep]
        if len(idx) == 0:
            return None, None
        if len(idx) > self.k:                                   # cap hubs: nearest in time
            dist = np.abs(ords - ref_ordinal)
            top = np.argpartition(dist, self.k)[: self.k]
            idx, ords = idx[top], ords[top]
        w = np.exp(-self.lam * np.abs(ords - ref_ordinal).astype(np.float64))
        s = w.sum()
        return idx, (w / s if s > 0 else w)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--split", required=True)
    a = ap.parse_args(); build(load_config(a.config), a.split)
