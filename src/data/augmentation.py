"""Step 02 — Filter paraphrases, merge into TRAIN ONLY, emit full accounting.

Input paraphrase file (jsonl), one object per paraphrase:
  {"source_eventid": ..., "text": ..., "label": ..., "sim": 0.91, "human_ok": 1}
# ADAPT: rename fields below if your generation logs differ.

Guarantees (enforced by assertion here AND re-checked by the audit):
  G1  every augmented row's source event is in the TRAIN partition of the given split
  G2  augmented rows carry is_augmented=1 and synthetic ids "aug_{source}_{k}"
  G3  val/test files contain ORIGINAL events only
Accounting emitted per split -> artifacts/augmented/accounting_{split}.json :
  original_corpus, original_train, generated, sim_filtered, human_filtered,
  train_source_filtered, final_added, final_train_size
"""
import argparse, json
import pandas as pd
from src.util import load_config, log, ensure_dir, read_manifest


def main(cfg, split_name):
    p, a = cfg["paths"], cfg["augmentation"]
    df = pd.read_parquet(p["processed"])
    man = read_manifest(cfg, split_name)
    train_ids = set(man.loc[man.split == "train", "eventid"])

    gen = pd.read_json(p["paraphrases"], lines=True)
    gen["source_eventid"] = gen["source_eventid"].astype(str)   # canonical id type
    generators = sorted(gen.get("generator", pd.Series(["unknown"])).astype(str).unique())
    acc = {"generator": generators,        # invariant 5: the generator is always named explicitly
           "original_corpus": int(len(df)),
           "original_train": int(len(train_ids)),
           "generated": int(len(gen))}

    gen = gen[gen["sim"] >= a["sim_threshold"]]
    acc["sim_filtered"] = int(len(gen))
    if a["require_human_ok"]:
        gen = gen[gen["human_ok"] == 1]
    acc["human_filtered"] = int(len(gen))
    gen = gen[gen["label"].isin(a["target_classes"])]

    # ---- G1: train-only sources. This is the leakage guarantee, not a preference. ----
    before = len(gen)
    gen = gen[gen["source_eventid"].isin(train_ids)].copy()
    acc["train_source_filtered"] = int(len(gen))
    if before != len(gen):
        log(f"NOTE: dropped {before - len(gen)} paraphrases whose source fell in val/test "
            f"for split '{split_name}' (correct behavior — sources are split-dependent)")

    gen["eventid"] = ["aug_" + str(s) + "_" + str(i) for i, s in enumerate(gen["source_eventid"])]
    gen["is_augmented"] = 1
    # inherit date + ALL structured metadata from the source event, so verbalization and
    # graph attachment treat a paraphrase exactly like its (train) source
    src_meta = df.drop(columns=["label", "text"]).rename(columns={"eventid": "source_eventid"})
    gen = gen[["eventid", "source_eventid", "label", "text", "is_augmented", "sim"]] \
        .merge(src_meta, on="source_eventid", how="left")

    orig_train = df[df["eventid"].isin(train_ids)].copy()
    orig_train["is_augmented"] = 0
    orig_train["source_eventid"] = orig_train["eventid"]
    train_aug = pd.concat([orig_train, gen], ignore_index=True)

    assert set(gen["source_eventid"]) <= train_ids, "G1 violated"
    assert not set(gen["eventid"]) & set(df["eventid"]), "G2 violated: id collision"
    acc["final_added"] = int(len(gen))
    acc["final_train_size"] = int(len(train_aug))

    ensure_dir(p["aug_dir"])
    out = f"{p['aug_dir']}/train_aug_{split_name}.parquet"
    train_aug.to_parquet(out, index=False)
    json.dump(acc, open(f"{p['aug_dir']}/accounting_{split_name}.json", "w"), indent=2)
    log(f"[{split_name}] accounting: {acc}")
    log(f"wrote {out}")

    # invariant 5: export a stratified sample of ACCEPTED paraphrases for the
    # AUTHOR's manual validation. The pipeline itself never claims a human-validation
    # statistic; human_ok stays 0 until the author reviews this file.
    n_rev = int(a.get("review_sample_size", 150))
    if len(gen):
        per_cls = max(1, n_rev // gen["label"].nunique())
        rev = (gen.sample(frac=1, random_state=0)
                  .groupby("label", group_keys=False).head(per_cls))
        src_txt = df.set_index("eventid")["text"]
        rev = rev.assign(source_text=rev["source_eventid"].map(src_txt))
        rev_out = f"{p['aug_dir']}/paraphrase_review_sample_{split_name}.csv"
        rev[["source_eventid", "label", "sim", "source_text", "text"]] \
            .rename(columns={"text": "paraphrase"}).to_csv(rev_out, index=False)
        log(f"review sample ({len(rev)} rows) for the author -> {rev_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", required=True, help="seed_42 | chrono | ...")
    args = ap.parse_args()
    main(load_config(args.config), args.split)
