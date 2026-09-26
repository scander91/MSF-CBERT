# MSF-CBERT

Multi-source fusion for terrorism attack-type classification on the Global Terrorism Database
(GTD). The model combines a conflict-domain encoder (ConfliBERT) over the event narrative with
verbalised categorical attributes, a gated dense branch over numeric attributes, training-only
rare-class paraphrase augmentation and focal loss. An optional partition-safe inductive event
graph (`MSF-CBERT+G`) can be enabled.

## Repository layout

```
src/
  data/       GTD loading, row-local preprocessing, split manifests, train-fit preprocessing,
              paraphrase generation and augmentation
  features/   training-partition feature selection (bias-corrected Cramér's V), verbalisation
  graph/      inductive event graph (training events only; held-out events receive messages)
  models/     MSF-CBERT model
  train/      training loop, structured-only and BertGCN-style baselines
  eval/       evaluation utilities
  audit/      fail-closed leakage audit
scripts/      pipeline driver, inference, config generation, data-quality and figure scripts
configs/      structdense.yaml: model configuration
tests/        graph-semantics test and audit negative test
artifacts/    split manifests and fingerprinted preprocessing / feature metadata
data/         place the GTD export here (not redistributed)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The GTD is licensed by the National Consortium for the Study of Terrorism and Responses to
Terrorism (START), University of Maryland, and is not redistributed here. Obtain
`globalterrorismdb_0522dist.xlsx` from START and place it in `data/raw/`.

## Preprocessing

Row-local steps (before splitting):

- event day recovered from the event identifier (yyyymmdd prefix) when the day field is 0;
- GTD unknown codes (−9 / −99) in numeric fields converted to missing values;
- categorical gaps kept as an explicit `Unknown` value (no mode filling);
- records without a usable narrative removed.

Training-fit steps (per split, fitted on training rows only and fingerprinted with the SHA-256
hash of the training-ID set): median imputation of remaining numeric gaps, redundancy filtering
(Pearson correlation, normalised mutual information, variance inflation), Cramér's V feature
selection, class weights, vocabularies and standardisation.

`scripts/gtd_quality_enhancement.py` reports the data-quality operations, including city
recovery from coordinates (nearest known city in the same country within 0.1 km, with a
hold-out check). `scripts/gtd_imputation_analysis.py` profiles unknown values in sensitive
fields.

## Leakage audit

```bash
python -m src.audit.leakage_audit --split seed_42
```

The audit exits non-zero unless partitions are disjoint and complete, augmented rows are
training-only, graph nodes and edges come from training events only, and every train-fit or
post-hoc artefact carries the expected partition fingerprint.

## Running

```bash
bash scripts/run_all.sh seed_42        # audited preprocessing and artefacts for one split
bash scripts/run_all.sh chrono         # chronological split

# MSF-CBERT (dense branch, no graph pathway)
python -m src.train.train --split seed_42 --config configs/structdense.yaml --no-graph

# MSF-CBERT+G (with the inductive event graph)
python -m src.train.train --split seed_42 --config configs/structdense.yaml
```

Splits: `seed_13, seed_21, seed_42, seed_87, seed_100, seed_7, seed_33, seed_55, seed_76,
seed_91` (70/15/15 stratified) and `chrono`. Model selection uses validation macro-F1.

## License

Code: MIT (see `LICENSE`). The GTD is licensed separately by START and is not included.
