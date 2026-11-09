"""Structured-only baselines (no narrative) on one split, fitted on TRAIN; hyper-parameters chosen on VALIDATION; CPU.
Feature sets: published (Tier-B kept set with ordinal codes, xgb only, reproduces the earlier baseline); parity (the 5 verbalised categoricals + 12 numerics given to the neural model); no_outcome; restricted (parity with the neural restricted arm).
Models: xgb (earlier settings), xgb_cat (native categoricals, validation grid), mlp (train-fitted one-hot, validation grid). Prints no test score."""
import argparse, json, os, sys, time, itertools
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
ap=argparse.ArgumentParser(); ap.add_argument('--split',required=True); ap.add_argument('--model',required=True); ap.add_argument('--features',default='parity'); a=ap.parse_args()
OUT=['nkill','nwound','success','extended','property']
df=pd.read_parquet('data/processed/gtd.parquet'); man=pd.read_csv(f'artifacts/splits/{a.split}.csv',dtype={'eventid':str})
tb=json.load(open(f'artifacts/preprocess/tierb_{a.split}.json')); num,cat=list(tb['kept_numeric']),list(tb['kept_categorical']); med=tb['numeric_medians']
if a.features!='published': cat=json.load(open(f'artifacts/features/selected_{a.split}.json'))['selected']
if a.features=='no_outcome': num=[c for c in num if c not in OUT]
if a.features=='restricted': num=['vicinity']; cat=[c for c in cat if c in('region_txt','country_txt','targtype1_txt')]
labels=sorted(df.label.unique()); lid={l:i for i,l in enumerate(labels)}
df['y']=df.label.map(lid); df=df.merge(man,on='eventid')
for c in num: df[c]=df[c].astype(float).fillna(med[c])
P={s:df[df.split==s].reset_index(drop=True) for s in['train','val','test']}
t0=time.time(); sel={}
def XO():
    e=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1).fit(P['train'][cat].astype(str))
    return {s:np.c_[e.transform(P[s][cat].astype(str)),P[s][num].values] for s in P}
ys={s:P[s].y.values for s in P}
if a.model=='xgb':
    from xgboost import XGBClassifier
    X=XO(); clf=XGBClassifier(n_estimators=600,max_depth=8,learning_rate=0.1,subsample=0.9,colsample_bytree=0.9,tree_method='hist',eval_metric='mlogloss',random_state=42,n_jobs=12).fit(X['train'],ys['train'])
elif a.model=='xgb_cat':
    from xgboost import XGBClassifier
    vocab={c:pd.CategoricalDtype(sorted(P['train'][c].astype(str).unique())) for c in cat}
    X={k:pd.concat([pd.DataFrame({c:P[k][c].astype(str).astype(vocab[c]) for c in cat}),P[k][num].astype(float)],axis=1) for k in P}
    best=-1
    w=(len(ys['train'])/(len(labels)*np.bincount(ys['train'],minlength=len(labels))))[ys['train']]
    for lr,dp,bal in itertools.product([.1],[6,8],[False,True]):
        c=XGBClassifier(n_estimators=600,max_depth=dp,learning_rate=lr,subsample=.9,colsample_bytree=.9,tree_method='hist',enable_categorical=True,max_cat_to_onehot=16,eval_metric='mlogloss',random_state=42,n_jobs=12).fit(X['train'],ys['train'],sample_weight=w if bal else None)
        v=f1_score(ys['val'],c.predict(X['val']),average='macro')
        if v>best: best,clf,sel=v,c,{'lr':lr,'max_depth':dp,'class_balanced':bal,'val_macro_f1':v}
elif a.model=='mlp':
    tr=P['train']; keep={c:set(tr[c].astype(str).value_counts().head(200).index) for c in cat}
    cap=lambda d:pd.DataFrame({c:d[c].astype(str).where(d[c].astype(str).isin(keep[c]),'__other') for c in cat})
    oh=OneHotEncoder(handle_unknown='ignore').fit(cap(tr)); sc=StandardScaler().fit(tr[num].values)
    import scipy.sparse as sp
    X={s:sp.hstack([oh.transform(cap(P[s])),sp.csr_matrix(np.clip(sc.transform(P[s][num].values),-5,5))]).tocsr() for s in P}
    best=-1
    for h in [(256,),(512,256)]:
        c=MLPClassifier(hidden_layer_sizes=h,batch_size=256,max_iter=20,early_stopping=False,random_state=int(a.split.split('_')[-1]) if a.split[-1].isdigit() else 42).fit(X['train'],ys['train'])
        v=f1_score(ys['val'],c.predict(X['val']),average='macro')
        if v>best: best,clf,sel=v,c,{'hidden':h,'val_macro_f1':v}
pv,pt=clf.predict(X['val']),clf.predict(X['test'])
n=f"{a.split}__struct_{a.model}_{a.features}"; r=f'runs/{n}'; os.makedirs(r,exist_ok=True)
pd.DataFrame({'eventid':P['test'].eventid,'true':ys['test'],'pred':pt}).to_csv(f'{r}/preds_test.csv',index=False)
json.dump({'run':n,'labels':labels,'features':{'cat':cat,'num':num},'selected':sel,'val_macro_f1':float(f1_score(ys['val'],pv,average='macro')),'test_macro_f1':float(f1_score(ys['test'],pt,average='macro')),'test_accuracy_secondary_imbalance_caveat':float((ys['test']==pt).mean()),'wall_clock_sec':round(time.time()-t0,1),'host':os.uname().nodename,'report':classification_report(ys['test'],pt,target_names=labels,output_dict=True,zero_division=0)},open(f'{r}/metrics.tmp','w'),indent=1); os.replace(f'{r}/metrics.tmp',f'{r}/metrics.json')
