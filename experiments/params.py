"""Trainable parameters per fusion arm (full model and fusion-specific layers); no data needed beyond the model files."""
import yaml, json, torch, os
os.environ['HF_HUB_OFFLINE']='1'
from src.models.msf_cbert import MSFCBert
meta=lambda n,:{"num_dim":n,"cat_cols":[],"cat_vocab_sizes":{}}
res={}
for a,n in [('gated',12),('concat_direct',12),('concat_dense',12),('mlp_matched',12),('fixed_avg',12),('text_cat',None)]:
    c=yaml.safe_load(open(f'experiments/configs/{a}.yaml'))
    m=MSFCBert(c,c['graph']['relations'].keys(),9,False,struct_meta=meta(n) if n else None)
    tot=sum(p.numel() for p in m.parameters()); enc=sum(p.numel() for p in m.encoder.parameters())
    res[a]={'total':tot,'encoder':enc,'non_encoder':tot-enc,'text_proj':sum(p.numel() for p in m.text_head_proj.parameters()),'fusion_and_head':tot-enc-sum(p.numel() for p in m.text_head_proj.parameters())}
    print(a,res[a])
json.dump(res,open('supplementary/results/param_counts.json','w'),indent=1)
