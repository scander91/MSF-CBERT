#!/usr/bin/env bash
# Train-fit artefacts for one group-disjoint split, then the leakage audit and the group-disjoint assertion.
set -euo pipefail
s=$1; C=experiments/configs/gated.yaml; PY=${PY:-python}
$PY -m src.data.preprocess_tierb --config $C --split $s
$PY -m src.data.augmentation --config $C --split $s
$PY -m src.features.select_features --config $C --split $s
$PY -m src.graph.build_graph --config $C --split $s
$PY -m src.audit.leakage_audit --config $C --split $s
$PY - "$s" <<'P'
import sys,pandas as pd
s=sys.argv[1]; m=pd.read_csv(f'artifacts/splits/{s}.csv',dtype={'eventid':str}); g=pd.read_parquet('artifacts/groups/groups.parquet')
j=m.merge(g,on='eventid'); assert len(j)==143586
for c in ['g_90','g_95','g_rel_exact']: assert j.groupby(c).split.nunique().max()==1,c
print('GROUP_DISJOINT_OK',s,{c:round(100*j[j.split=='test'][c].isin(set(j[j.split=='train'][c])).mean(),3) for c in ['g_85','g_80']})
P
