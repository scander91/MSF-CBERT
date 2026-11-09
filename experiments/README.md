# Additional experiments

Scripts and configurations for the fusion, feature-group, group-disjoint and graph experiments
reported in the paper and in `supplementary/Supplementary_Material.pdf`. Run every command from the
repository root with `PYTHONPATH=.`.

## Group-disjoint splits

```bash
python experiments/build_groups.py            # label-free event groups -> artifacts/groups/
python experiments/make_group_splits.py       # ten group-disjoint 70/15/15 manifests -> artifacts/splits/groupseed_*.csv
bash experiments/prep_group_split.sh groupseed_42   # train-fit artefacts, leakage audit, group-disjointness check
```

A group is a connected component of events linked by the GTD `related` field, by an identical
summary (leading date stamp removed, whitespace-normalised, lower-cased), or by TF-IDF cosine
>= 0.90 between those texts; all pairs are compared exhaustively. No label is read. Groups at
cosine 0.95/0.85/0.80 and related+identical only are stored for sensitivity analysis. The
manifests use the same seeds and label stratification as the stratified splits
(StratifiedGroupKFold with 20 folds assigned 14/3/3). The group-disjoint manifests, train-fit
metadata and group reports are included under `artifacts/`.

## Systems

`make_arms.py` writes one configuration per system to `experiments/configs/` and the run list
`experiments/queue.tsv` (columns: priority, system, split, config, flags). Each run is

```bash
python -m src.train.train --config experiments/configs/<system>.yaml --split <split> --name <split>__<system> <flags>
```

| system | inputs / fusion |
|---|---|
| `gated` | MSF-CBERT: text + 5 verbalised categoricals + 12 numerics, gated fusion |
| `concat_direct` | standardised numerics concatenated to the text vector |
| `concat_dense` | dense numeric vector concatenated to the text vector |
| `fixed_avg` | gate replaced by a fixed 0.5/0.5 average |
| `mlp_matched` | MLP fusion head, 681 hidden units, within 102 trainable parameters of `gated` |
| `text_cat`, `text_num`, `text_only` | without numerics / without categoricals / narrative only |
| `no_outcome`, `no_success`, `no_extended`, `no_property`, `no_casualty`, `no_coded`, `vicinity_only` | numeric field groups removed |
| `no_weapon`, `no_perp` | one verbalised categorical removed |
| `restricted`, `restricted_first`, `text_first` | reduced-input sensitivity settings (narrative cut to its first sentence for `_first`) |
| `graph_nocat`, `graph_gated` | graph pathway without categoricals; MSF-CBERT+G |

The fusion alternatives are selected with `model.fusion_mode` in `src/models/msf_cbert.py`;
`model.verbalize_exclude`, `model.struct_dense_exclude` and `model.narrative: first_sentence`
are handled in `src/train/train.py`. `params.py` reports trainable parameters per fusion variant.
Structured-only baselines (no narrative): `structured_baselines.py --split <split> --model {xgb,xgb_cat,mlp} --features {published,parity,no_outcome,restricted}`
(settings chosen on validation macro-F1). The BertGCN-style baseline is `src/train/bertgcn_inductive.py`.

## Statistics, tables and figures

```bash
python experiments/analyze.py              # per-run metrics, summaries, corrected resampled t-tests, Holm
python experiments/secondary_analyses.py   # group exposure, group-balanced macro-F1, exposed/unexposed accuracy
python experiments/make_tables.py          # supplementary/tables/
python experiments/make_figures.py         # supplementary/figures/
```

`analyze.py` and `secondary_analyses.py` read `runs/<run>/metrics.json` and `preds_test.csv`;
their outputs are included under `supplementary/results/`, so the tables and figures can be
regenerated without the run directories except for the per-class tables, which read the run
files. Statistics:
paired Nadeau-Bengio corrected resampled t (two-sided, n_test/n_train = 0.15/0.70) with Holm
correction within four families (fusion, feature dependence, group-disjoint, graph); all other
comparisons are descriptive and unadjusted.
