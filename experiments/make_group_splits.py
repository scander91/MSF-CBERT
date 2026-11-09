"""Group-disjoint 70/15/15 manifests over the same 143,586 events. Stratification uses the label as in the original
splits; groups do not. Twenty StratifiedGroupKFold folds are assigned 14/3/3 with the split seed."""
import argparse, numpy as np, pandas as pd, json, os
from sklearn.model_selection import StratifiedGroupKFold
ap=argparse.ArgumentParser(); ap.add_argument('--col',default='g_90'); ap.add_argument('--prefix',default='groupseed'); ap.add_argument('--seeds',default='13,21,42,87,100,7,33,55,76,91'); a=ap.parse_args()
d=pd.read_parquet('data/processed/gtd.parquet',columns=['eventid','label']).merge(pd.read_parquet('artifacts/groups/groups.parquet')[['eventid',a.col]],on='eventid')
out=[]
for s in map(int,a.seeds.split(',')):
    f=np.zeros(len(d),int)
    for k,(_,te) in enumerate(StratifiedGroupKFold(20,shuffle=True,random_state=s).split(d,d.label,d[a.col])): f[te]=k
    order=np.random.default_rng(s).permutation(20); part=np.empty(20,dtype=object); part[order[:14]]='train'; part[order[14:17]]='val'; part[order[17:]]='test'
    m=pd.DataFrame({'eventid':d.eventid,'split':part[f]}); m.to_csv(f'artifacts/splits/{a.prefix}_{s}.csv',index=False)
    j=d.assign(split=m.split); assert j.groupby(a.col).split.nunique().max()==1
    t=pd.crosstab(j.label,j.split); out.append({'seed':s,'sizes':j.split.value_counts().to_dict(),'groups':j.groupby('split')[a.col].nunique().to_dict(),'class_counts':t.to_dict(),
      'max_abs_class_share_diff_test_vs_corpus':float((t['test']/t['test'].sum()-j.label.value_counts(normalize=True)).abs().max())})
    print(s,out[-1]['sizes'],out[-1]['max_abs_class_share_diff_test_vs_corpus'],t['test'].to_dict())
json.dump(out,open(f'artifacts/groups/{a.prefix}_split_report_{a.seeds}.json','w'),indent=1)
