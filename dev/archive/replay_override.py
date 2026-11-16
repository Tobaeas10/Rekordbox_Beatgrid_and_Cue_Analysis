import sys,glob,pickle,os,json,numpy as np
sys.path.insert(0,'.')
import pipeline as P
from multiprocessing import Pool
def job(f):
    tid=os.path.basename(f).split('.')[0]
    try:
        ev=pickle.load(open(f,'rb'))
        blocks=P.block_scans(ev);raw=P.find_sections(blocks)
        fits=P.clean_sections(ev,raw,[P.fit_section(ev,x) for x in raw])
        out=[]
        for k,s in enumerate(fits):
            o=P.snare_override(ev,s)
            t,c=ev['t'],ev['c'];T=s['period_s']
            m=(c==0)&(t>=s['start_s'])&(t<=s['end_s'])
            r=((t[m]-s['anchor_s']+T/2)%T-T/2)*1000
            out.append(dict(k=k,ovr=o.get('snare_override_ms'),rms=s['rms_ms'],kon=round(float((np.abs(r)<=4).mean()),2) if m.sum() else None,nk=int(m.sum()),bpm=s['bpm'],a=s['anchor_s'],dur=s['end_s']-s['start_s']))
        return tid,out
    except Exception as e:
        return tid,repr(e)
if __name__=='__main__':
    fs=sorted(glob.glob('/home/claude/w/cache/cache/*.ff2.pkl'))
    done={json.loads(l)[0] for l in open('/tmp/replay_out.jsonl')}
    fs=[f for f in fs if os.path.basename(f).split('.')[0] not in done]
    with Pool(2) as p, open('/tmp/replay_out.jsonl','a') as fo:
        for r in p.imap_unordered(job,fs):
            fo.write(json.dumps(r)+'\n');fo.flush()
