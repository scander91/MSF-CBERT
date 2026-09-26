"""Baseline: XGBoost on structured attributes ONLY (no text, no graph, no augmentation).

Purpose: quantifies exactly how much signal the structured metadata carries by
itself; the gap to the full model is the evidence that verbalized attributes do not
trivialize the task.

All distribution-dependent preprocessing comes from the Tier-B artifact
(artifacts/preprocess/tierb_{split}.json): train-fit medians, train-selected feature
set after PCC/NMI/VIF redundancy elimination (D8) — verified by audit check A7.
The OrdinalEncoder is likewise fit on TRAIN only.
"""
import argparse, json, os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
from src.util import load_config, log, ensure_dir, read_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--split", required=True)
    ap.add_argument("--label-scheme", default="with_unknown",
                    choices=["with_unknown", "without_unknown"])
    a = ap.parse_args(); cfg = load_config(a.config)

    df = pd.read_parquet(cfg["paths"]["processed"])
    drops = cfg["data"]["label_schemes"][a.label_scheme]
    df = df[~df["label"].isin(drops)]                       # D4: uniform across partitions
    man = read_manifest(cfg, a.split)

    tb = json.load(open(f"{cfg['paths']['preprocess_dir']}/tierb_{a.split}.json"))
    cat_cols, num_cols = tb["kept_categorical"], tb["kept_numeric"]
    med = tb["numeric_medians"]

    labels = sorted(df["label"].unique()); label2id = {l: i for i, l in enumerate(labels)}
    df = df.copy(); df["y"] = df["label"].map(label2id)
    for c in num_cols:                                      # train-fit medians, frozen (A7)
        df[c] = df[c].fillna(med[c])

    parts = {s: df.merge(man[man.split == s], on="eventid") for s in ["train", "val", "test"]}
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xtr = pd.concat([pd.DataFrame(enc.fit_transform(parts["train"][cat_cols].astype(str)),
                                  index=parts["train"].index),
                     parts["train"][num_cols].reset_index(drop=True)], axis=1).values
    Xte = pd.concat([pd.DataFrame(enc.transform(parts["test"][cat_cols].astype(str)),
                                  index=parts["test"].index),
                     parts["test"][num_cols].reset_index(drop=True)], axis=1).values

    clf = XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.1,
                        subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                        eval_metric="mlogloss", random_state=42)
    clf.fit(Xtr, parts["train"]["y"])
    pred = clf.predict(Xte)
    mf1 = f1_score(parts["test"]["y"], pred, average="macro")

    name = f"{a.split}_xgb_structured" + ("_nounk" if a.label_scheme == "without_unknown" else "")
    run = os.path.join(cfg["paths"]["runs_dir"], name); ensure_dir(run)
    pd.DataFrame({"eventid": parts["test"]["eventid"], "true": parts["test"]["y"],
                  "pred": pred}).to_csv(f"{run}/preds_test.csv", index=False)
    json.dump({"run": name, "labels": labels, "label_scheme": a.label_scheme,
               "features": {"categorical": cat_cols, "numeric": num_cols},
               "test_macro_f1": float(mf1),
               "test_accuracy_secondary_imbalance_caveat":
                   float((parts["test"]["y"].values == pred).mean()),
               "report": classification_report(parts["test"]["y"], pred,
                                               target_names=labels, output_dict=True,
                                               zero_division=0)},
              open(f"{run}/metrics.json", "w"), indent=2)
    log(f"[{name}] test macro-F1 = {mf1:.4f}")


if __name__ == "__main__":
    main()
