"""Executable proof of the graph's leakage properties.
Run after build_graph:   python tests/test_graph_semantics.py --split seed_42
T1 neighbors are train events only | T2 decay weights normalized
T3 past_only returns strictly-earlier train events | T5 self-exclusion
T6 unseen entity -> UNK path | T7 hub neighbor cap respected."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.graph.build_graph import GraphContext
from src.util import load_config, read_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="seed_42")
ap.add_argument("--config", default="config.yaml")
a = ap.parse_args()
cfg = load_config(a.config)

g = GraphContext(f"{cfg['paths']['graph_dir']}/graph_{a.split}.pkl")
df = pd.read_parquet(cfg["paths"]["processed"])
man = read_manifest(cfg, a.split)
train_ids = set(man[man.split == "train"]["eventid"])
dates = df.set_index("eventid").loc[list(map(str, g.train_eventids)), "date"] \
          .map(pd.Timestamp.toordinal).values

ev = df.merge(man[man.split == "test"], on="eventid").iloc[0]
ref = pd.Timestamp(ev["date"]).toordinal()
for rel, col in cfg["graph"]["relations"].items():
    idx, w = g.neighbors(rel, ev[col], ref)
    if idx is None:
        continue
    assert {str(g.train_eventids[i]) for i in idx} <= train_ids, "T1 failed: non-train neighbor"
    assert abs(w.sum() - 1) < 1e-9, "T2 failed: weights not normalized"
    pidx, _ = g.neighbors(rel, ev[col], ref, past_only=True)
    assert pidx is None or (dates[pidx] < ref).all(), "T3 failed: future train event in past_only"

tr = df.merge(man[man.split == "train"], on="eventid").iloc[0]
srow = list(map(str, g.train_eventids)).index(str(tr["eventid"]))
idx, _ = g.neighbors("weapon", tr["weaptype1_txt"],
                     pd.Timestamp(tr["date"]).toordinal(), self_row=srow)
assert idx is None or srow not in set(idx.tolist()), "T5 failed: self-loop present"

assert g.neighbors("weapon", "NOT_A_WEAPON", ref) == (None, None), "T6 failed"
g.k = 5
idx, _ = g.neighbors("weapon", ev["weaptype1_txt"], ref)
assert idx is None or len(idx) <= 5, "T7 failed: hub cap ignored"
print("ALL GRAPH SEMANTICS TESTS PASSED")
