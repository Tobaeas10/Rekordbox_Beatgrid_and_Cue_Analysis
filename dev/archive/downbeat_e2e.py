import sys,json,glob,pickle,numpy as np
sys.path.insert(0,'.')
import pipeline as P
C='../c/cache/'
rep=json.load(open(glob.glob('/mnt/user-data/attachments/*/fb9acb36-eval_report.json')[0]))
ok=0;rows=[]
for e in rep:
    i=e['id']; ev=pickle.load(open(C+i+'.ff2.pkl','rb')); z=np.load(C+i+'.feat.npz')
    ev['bass']=z['env'][1]; ev['bass_rate']=float(z['sr'])/float(z['hop'])
    raw=P.find_sections(P.block_scans(ev))
    fits=P.fix_rb_ratio(ev,raw,P.clean_sections(ev,raw,[P.fit_section(ev,x) for x in raw]),e['rekordbox_bpm'])
    secs=[P.refine_anchor(ev,P.fix_parity(ev,P.snare_override(ev,f),first=(k==0))) for k,f in enumerate(fits)]
    model=e['offset_model']['ms']
    built=P.build_grid(secs,model,ev['dur'],ev)
    fd=e['rb_grid']['first_downbeat_ms']
    j=int(np.argmin([abs(b[2]-fd) for b in built]))
    good=built[j][0]==1 and abs(built[j][2]-fd)<10
    ok+=good
    rows.append((e['artist'][:12],built[j][0],round(built[j][2]-fd),secs[0]['downbeat'] and (secs[0]['downbeat']['best'],secs[0]['downbeat']['margin'],secs[0]['downbeat']['applied']),P.assess(ev,secs,ev['dur'])[1][-1:] ))
    print(rows[-1],'OK' if good else 'WRONG')
print(ok,'/',len(rep))
