#!/usr/bin/env bash
# Audited pipeline for ONE split, training the evaluated systems.
#   bash scripts/run_all.sh seed_42        (splits: seed_13 seed_21 seed_42 seed_87 seed_100
#                                                   seed_7 seed_33 seed_55 seed_76 seed_91)
#   bash scripts/run_all.sh chrono         (chronological split)
set -euo pipefail
SPLIT=${1:?usage: run_all.sh <seed_XX|chrono>}
PY="${PY:-python}"
MODEL_CFG=configs/structdense.yaml      # model configuration (dense numeric branch enabled)

# ---- data preparation (idempotent; load/splits skipped if artifacts exist) ----
[ -f data/processed/gtd.parquet ] || $PY -m src.data.load_gtd
[ -f "artifacts/splits/${SPLIT}.csv" ] || $PY -m src.data.splits --config "$MODEL_CFG"
$PY -m src.data.preprocess_tierb      --split "$SPLIT"     # training-fit preprocessing
$PY -m src.data.augmentation          --split "$SPLIT"
$PY -m src.features.select_features   --split "$SPLIT"
$PY -m src.graph.build_graph          --split "$SPLIT"

# ---- gate: nothing trains unless every leakage check passes ----
$PY -m src.audit.leakage_audit        --split "$SPLIT"
$PY tests/test_graph_semantics.py     --split "$SPLIT"
$PY tests/test_a8_val_fit_audit.py    --split "$SPLIT"

# ---- systems (checkpoint selection and early stopping use validation macro-F1) ----
$PY -m src.train.train --split "$SPLIT" --config "$MODEL_CFG" --no-graph --name "${SPLIT}_msfcbert"     # MSF-CBERT
$PY -m src.train.train --split "$SPLIT" --config "$MODEL_CFG"            --name "${SPLIT}_msfcbert_g"   # MSF-CBERT+G
$PY -m src.train.train --split "$SPLIT"                                  --name "${SPLIT}_text_graph"   # text+graph
$PY -m src.train.train --split "$SPLIT" --no-graph                       --name "${SPLIT}_text_cat"     # text+categoricals
$PY -m src.train.bertgcn_inductive --split "$SPLIT" --bert-run "runs/${SPLIT}_text_cat"                 # BertGCN-style

echo "Done: ${SPLIT}. Repeat for all ten seeds; each run writes runs/<name>/metrics.json."
