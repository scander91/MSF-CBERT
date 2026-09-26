"""Generate config variants for Stage 6/6a levers (configs/*.yaml).

Each variant is the frozen base config with ONE lever changed (plus disjoint artifact
paths where the lever changes a train-fit artifact, so nothing ever overwrites the
matrix artifacts). Deterministic: safe to re-run.
"""
import copy, os, yaml

BASE = "config.yaml"
OUT = "configs"

def emit(name, mutate):
    cfg = yaml.safe_load(open(BASE))
    mutate(cfg)
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}.yaml"
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    print("wrote", path)

def set_lr(v):
    def m(c): c["train"]["lr"] = v
    return m

def set_ml(v):
    def m(c): c["model"]["max_len"] = v
    return m

def aug8(c):
    c["paths"]["paraphrases"] = "data/raw/paraphrases8.jsonl"
    c["paths"]["aug_dir"] = "artifacts/augmented8"
    c["augmentation"]["paraphrases_per_event"] = 8

def half_life(days):
    def m(c):
        c["graph"]["half_life_days"] = days
        c["paths"]["graph_dir"] = f"artifacts/graph_hl{days}"
    return m

def flag(key):
    def m(c): c["model"][key] = True
    return m

emit("lr1e5", set_lr(1.0e-5))
emit("lr3e5", set_lr(3.0e-5))
emit("ml384", set_ml(384))
emit("aug8", aug8)
emit("hl365", half_life(365))
emit("hl1825", half_life(1825))
emit("hl36500", half_life(36500))          # effectively no decay (100 years)
emit("auxg", flag("aux_graph_loss"))
emit("relattn", flag("rel_attention"))
emit("moprior", flag("mo_prior"))
