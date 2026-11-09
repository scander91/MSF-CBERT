"""Writes experiments/configs/<arm>.yaml and experiments/queue.tsv (priority, arm, split, config, flags). One place defines every arm."""
import yaml, copy
base=yaml.safe_load(open('configs/structdense.yaml'))
A='artifacts'
base['paths'].update(splits_dir=f'{A}/splits',features_dir=f'{A}/features',aug_dir=f'{A}/augmented',preprocess_dir=f'{A}/preprocess',graph_dir=f'{A}/graph',runs_dir='runs')
OUT=['nkill','nwound','success','extended','property']
CODED=['crit1','crit2','crit3','multiple','int_any']
ALLNUM=['nkill','nwound','success','suicide','extended','multiple','crit1','crit2','crit3','vicinity','property','int_any']
RESTRICT_NUM=[c for c in ALLNUM if c!='vicinity']
RESTRICT_CAT=['gname','weaptype1_txt']
ARMS={ # arm: (model overrides, flags)
 'gated':({},'--no-graph'),
 'text_only':({'struct_dense':False},'--no-graph --no-structured'),
 'text_cat':({'struct_dense':False},'--no-graph'),
 'concat_direct':({'fusion_mode':'concat_direct'},'--no-graph'),
 'concat_dense':({'fusion_mode':'concat_dense'},'--no-graph'),
 'mlp_matched':({'fusion_mode':'mlp_matched','fusion_mlp_hidden':681},'--no-graph'),
 'fixed_avg':({'fusion_mode':'fixed_avg'},'--no-graph'),
 'no_outcome':({'struct_dense_exclude':OUT},'--no-graph'),
 'text_num':({},'--no-graph --no-structured'),
 'vicinity_only':({'struct_dense_exclude':RESTRICT_NUM},'--no-graph'),
 'restricted':({'struct_dense_exclude':RESTRICT_NUM,'verbalize_exclude':RESTRICT_CAT},'--no-graph'),
 'restricted_first':({'struct_dense_exclude':RESTRICT_NUM,'verbalize_exclude':RESTRICT_CAT,'narrative':'first_sentence'},'--no-graph'),
 'text_first':({'struct_dense':False,'narrative':'first_sentence'},'--no-graph --no-structured'),
 'no_weapon':({'verbalize_exclude':['weaptype1_txt']},'--no-graph'),
 'no_perp':({'verbalize_exclude':['gname']},'--no-graph'),
 'no_coded':({'struct_dense_exclude':CODED},'--no-graph'),
 'no_extended':({'struct_dense_exclude':['extended']},'--no-graph'),
 'no_property':({'struct_dense_exclude':['property']},'--no-graph'),
 'no_casualty':({'struct_dense_exclude':['nkill','nwound']},'--no-graph'),
 'no_success':({'struct_dense_exclude':['success']},'--no-graph'),
 'graph_nocat':({'struct_dense':False},'--no-structured'),
 'graph_gated':({},''),
}
for k,(o,_) in ARMS.items():
    c=copy.deepcopy(base); c['model'].update(o); yaml.safe_dump(c,open(f'experiments/configs/{k}.yaml','w'),sort_keys=False)
S=[42,13,21,87,100,7,33,55,76,91]
def rows(t,arms,pre):
    return [(t,a,f'{pre}_{s}') for s in S for a in arms]
Q=[]
Q+=rows(1,['gated','text_cat','text_only','concat_direct','mlp_matched','fixed_avg','no_outcome','vicinity_only','text_num','restricted','restricted_first'],'seed')
Q+=rows(1,['gated','text_cat','text_only','restricted'],'groupseed')
Q+=rows(2,['concat_direct','mlp_matched','no_outcome'],'groupseed')
Q+=rows(2,['concat_dense','text_first','no_weapon','no_perp','no_coded','no_extended','no_property'],'seed')
Q+=rows(2,['no_casualty','no_success'],'seed')
Q+=[(3,a,'chrono') for a in ['text_only','no_outcome','restricted','restricted_first']]
Q+=rows(3,['graph_nocat'],'seed')
Q+=rows(3,['graph_gated'],'groupseed')
with open('experiments/queue.tsv','w') as f:
    for t,a,sp in Q:
        f.write(f'{t}\t{a}\t{sp}\texperiments/configs/{a}.yaml\t{ARMS[a][1]}\n')
from collections import Counter; print(Counter(q[0] for q in Q),len(Q))
