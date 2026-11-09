# MSF-CBERT

Multi-source fusion for terrorism attack-type classification on the Global Terrorism Database
(GTD). MSF-CBERT combines a ConfliBERT encoder over the event narrative with verbalised
categorical attributes and a gated dense branch over numeric attributes. An optional inductive
event graph (`MSF-CBERT+G`) can be enabled.

## Repository layout

```
src/            data loading and preprocessing, feature selection, graph, model, training, audit
scripts/        pipeline driver and data-quality scripts
configs/        model configuration (structdense.yaml)
experiments/    group-disjoint splits, fusion and feature-group variants, baselines, statistics
supplementary/  Supplementary_Material.pdf (Tables S1-S14, figures) and result CSV files
artifacts/      split manifests and training-fit metadata
tests/          graph and leakage-audit tests
data/           place the GTD export here (not redistributed)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The GTD is licensed by START (University of Maryland) and is not redistributed. Obtain
`globalterrorismdb_0522dist.xlsx` from START and place it in `data/raw/`.

## Running

```bash
bash scripts/run_all.sh seed_42                    # preprocessing, artefacts and leakage audit
python -m src.train.train --split seed_42 --config configs/structdense.yaml --no-graph   # MSF-CBERT
python -m src.train.train --split seed_42 --config configs/structdense.yaml              # MSF-CBERT+G
```

Stratified splits: `seed_13, seed_21, seed_42, seed_87, seed_100, seed_7, seed_33, seed_55,
seed_76, seed_91` (70/15/15); group-disjoint splits: `groupseed_*`; chronological split:
`chrono`. Model selection uses validation macro-F1. Additional experiments are described in
`experiments/README.md`.

## License

Code: MIT (see `LICENSE`). The GTD is licensed separately by START and is not included.
