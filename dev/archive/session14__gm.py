import json,re,glob,collections,sys
from metric import cat,AC,CONT
PD='new/pl/playlists (=genre)/'
def loadpl(f):
    t=open(PD+f,'rb').read().decode('utf-16').splitlines()[1:]
    return {(x.split('\t')[1]+' - '+x.split('\t')[2]).strip().lower() for x in t if x.count('\t')>2}
G={}
for g,fs in (('dubstep',['Dubstep.txt']),('riddim',['Riddim.txt']),('tearout',['Tearout.txt']),('140',['140.txt']),('dnb',['DnB.txt','Liquid Dnb&Jungle.txt']),('jungle',['Jungle.txt']),('2step',['2-Step.txt']),('ukg',['UKG (4x4 only).txt'])):
    G[g]=set().union(*[loadpl(f) for f in fs])
def genre(r):
    k=(r['artist']+' - '+r['title']).strip().lower()
    for g in ('tearout','riddim','jungle','dnb','2step','ukg','140','dubstep'):
        if k in G[g]: return g
    return 'unknown'
def load(p): return json.load(open(p))
U='/mnt/user-data/uploads/'
import os
SETS={k:load(U+f) for k,f in (('old961','eval_report.json'),('new698','eval_report_-_698_dubstep_songs.json'),('extra','eval_report__extra_subgenres_.json')) if os.path.exists(U+f)}
FOLDS=set(json.load(open('bundle/llm_beatgrid_bundle_v6/dev/session13/folds.json')))
if __name__=='__main__':
    for k,R in SETS.items():
        c=collections.Counter(); ps=sum(any('phase_select' in s for s in r['sections']) for r in R); fs=sum(any(s.get('fine_shift_ms') is not None for s in r['sections']) for r in R)
        print(k,len(R),'phase_select',ps,'fine_shift',fs,'overlap with old ids',len({r['id'] for r in R}&{r['id'] for r in SETS['old961']}) if k!='old961' else '')
        print(collections.Counter(genre(r) for r in R))
