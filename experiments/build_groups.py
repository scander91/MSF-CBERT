"""Label-free event groups for group-disjoint partitioning (split construction, not a fitted model input).
An edge joins two events if (a) GTD `related` links them, (b) their whitespace-normalised summaries with the
leading date removed are identical, or (c) TF-IDF cosine of those texts >= threshold. Groups = connected components.
All pairs are compared exhaustively. No label is read."""
import argparse, json, re, multiprocessing as mp
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
ap=argparse.ArgumentParser(); ap.add_argument('--raw',default='data/raw/globalterrorismdb_0522dist.xlsx'); ap.add_argument('--out',default='artifacts/groups'); ap.add_argument('--procs',type=int,default=12)
a=ap.parse_args()
d=pd.read_parquet('data/processed/gtd.parquet',columns=['eventid','text'])
r=(pd.read_parquet(a.raw,columns=['eventid','related']) if a.raw.endswith('.parquet') else
   pd.read_excel(a.raw,usecols=['eventid','related'],dtype={'eventid':str,'related':str}))
r['eventid']=r.eventid.astype(str)
d=d.merge(r,on='eventid',how='left'); n=len(d); idx={e:i for i,e in enumerate(d.eventid)}
strip=d.text.str.replace(r'^\d{2}/\d{2}/\d{4}:\s*','',regex=True).str.replace(r'\s+',' ',regex=True).str.strip().str.lower()
rel=[]; outside=0
for i,s in enumerate(d.related):
    if pd.isna(s): continue
    for t in re.split(r'[,;\s]+',s.strip()):
        t=t.strip()
        if not t: continue
        if t in idx:
            if idx[t]!=i: rel.append((i,idx[t]))
        else: outside+=1
exact=[]; 
for _,g in pd.Series(np.arange(n)).groupby(strip.values):
    v=g.values
    exact+= [(v[0],x) for x in v[1:]]
X=TfidfVectorizer(sublinear_tf=True,dtype=np.float32,token_pattern=r'(?u)\b\w+\b').fit_transform(strip).tocsr()
T=0.80; Xt=X.T.tocsr()
def block(s):
    S=(X[s:s+200]@Xt).toarray()
    r,c=np.where(S>=T); k=(r+s)<c
    return np.c_[r[k]+s,c[k],S[r[k],c[k]]]
with mp.get_context('fork').Pool(a.procs) as pool: pairs=pool.map(block,range(0,n,200),chunksize=4)
P=np.vstack(pairs); print('exhaustive pairs>=.80',len(P),flush=True)
np.save(f'{a.out}/near_pairs.npy',P.astype(np.float32))
def comps(edges):
    if len(edges)==0: return np.arange(n)
    e=np.asarray(edges)[:,:2].astype(int)
    A=sp.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(n,n))
    return connected_components(A,directed=False)[1]
rep={'n':n,'related_edges':len(rel),'related_targets_outside_corpus':outside,'exact_edges':len(exact),
     'events_with_related':int(d.related.notna().sum())}
base=rel+exact; out=pd.DataFrame({'eventid':d.eventid})
def summ(c):
    s=pd.Series(c).value_counts(); return {'groups':int(len(s)),'singletons':int((s==1).sum()),'events_in_multi':int(s[s>1].sum()),'largest':int(s.iloc[0]),'top5':s.head(5).tolist(),'p99_size':float(s.quantile(.99))}
rep['related_only']=summ(comps(rel)); rep['exact_only']=summ(comps(exact)); rep['related+exact']=summ(comps(base)); out['g_rel_exact']=comps(base)
for t in [.95,.90,.85,.80]:
    e=[tuple(x) for x in P[P[:,2]>=t][:,:2].astype(int)]
    c=comps(base+e); rep[f'near_only_{t}']=summ(comps(e)); rep[f'all_{t}']=summ(c); out[f'g_{int(t*100)}']=c
out.to_parquet(f'{a.out}/groups.parquet'); json.dump(rep,open(f'{a.out}/groups_report.json','w'),indent=1); print(json.dumps(rep,indent=1))
