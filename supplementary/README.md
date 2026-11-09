# Supplementary material

`Supplementary_Material.pdf` contains Supplementary Tables S1-S8 cited in the paper, additional
tables (S9-S14) and figures.

| Table | Content | CSV |
|---|---|---|
| S1 | Feature availability and target-leakage audit | `tables/csv/T3_feature_availability_audit.csv`, `tables/csv/T3_feature_availability_audit_full.csv` |
| S2 | Gated fusion vs. simpler fusion of the same inputs, parameter counts | `tables/csv/T2_parameter_matched_fusion.csv` |
| S3 | Feature-group exclusion and restricted input | `tables/csv/T4_feature_group_ablation.csv` |
| S4 | Original stratified vs. group-disjoint evaluation, group-balanced macro-F1 | `tables/csv/T5_original_vs_group_disjoint.csv` |
| S5 | Grouping thresholds and exposure | `tables/csv/T5c_group_exposure_sensitivity.csv` |
| S6 | Graph contribution | `tables/csv/T7_graph_contribution.csv` |
| S7 | Complete baseline comparison incl. structured-only models | `tables/csv/T1_baseline_comparison.csv` |
| S8 | All paired tests | `tables/csv/T6_statistical_tests.csv` |
| S9 | Per-class accuracy, test events with / without a group member in training | `tables/csv/T5d_perclass_exposed_unexposed.csv` |
| S10 | Per-class F1, original vs. group-disjoint | `tables/csv/T11_perclass_f1_S_vs_G.csv` |
| S11 | Chronological split | `tables/csv/T9_chronological.csv` |
| S12 | XGBoost on restricted inputs: accuracy variation across seeds | `tables/csv/T10_xgb_restricted_diagnosis.csv` |
| S13 | Group-disjoint partition sizes | `tables/csv/T5b_group_split_sizes.csv` |
| S14 | Paper values vs. recomputed values | `tables/csv/T8_existing_vs_new.csv` |

`results/processed/per_run_metrics.csv` lists every run (partition, system, seed, test macro-F1,
test accuracy, validation macro-F1); `results/statistical_tests/` holds all paired tests;
`results/secondary/` the group-exposure analyses. Figures are in `figures/` with their plotting
data in `figures/data/`. No GTD records, narratives, paraphrases or checkpoints are included.
