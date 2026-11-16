import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import gzip,pickle,json,numpy as np,collections,sys,os
beat_features_path=sys.argv[1] if len(sys.argv)>1 else str(paths.CUE_TRAINING / 'beat_features_slim.pkl.gz')
cues_path=sys.argv[2] if len(sys.argv)>2 else str(paths.CUE_TRAINING / 'cues.jsonl')
beat_features=pickle.load(gzip.open(beat_features_path,'rb'))
band_features_path=sys.argv[3] if len(sys.argv)>3 else None   # optional band_features_slim.pkl.gz from dev/tools/extract_band_features.py
band_features=pickle.load(gzip.open(band_features_path,'rb')) if band_features_path else {}
cue_rows=[json.loads(l) for l in open(cues_path)]
dataset=[];skip_reasons=collections.Counter();alignment_errors=[]
for cue_row in cue_rows:
    track_beat_features=beat_features.get(cue_row['id'])
    if track_beat_features is None: skip_reasons['no_beat_features']+=1;continue
    if cue_row['grid']['multi_tempo'] or track_beat_features['info_n_sections']>1: skip_reasons['multi']+=1;continue
    hot_cues={cue['kind']:cue for cue in cue_row['cues'] if cue['kind']>0 and cue['bar'] is not None}
    if 5 not in hot_cues: skip_reasons['noD']+=1;continue
    if any(kind in hot_cues and hot_cues[kind]['bar']<hot_cues[5]['bar'] for kind in (6,7)): skip_reasons['sys2']+=1;continue
    period_seconds=float(track_beat_features['info_period_s']);lattice_origin_seconds=float(track_beat_features['info_lattice_t0_s']);beat_energy_table=track_beat_features['BE'].astype(np.float32)
    shift=round((cue_row['grid']['first_ms']/1000-lattice_origin_seconds)/period_seconds)   # BE index of stored beat 0
    stored_bar_zero_index=cue_row['grid']['first_bar_beat_idx']+shift       # BE index of stored bar 0
    # consistency: D cue beat index vs BE index
    first_drop_beat_index=(hot_cues[5]['in_ms']/1000-lattice_origin_seconds)/period_seconds; expected_beat_index=stored_bar_zero_index+hot_cues[5]['bar']*4
    alignment_errors.append(first_drop_beat_index-expected_beat_index)
    if abs(first_drop_beat_index-expected_beat_index)>0.6: skip_reasons['misaligned']+=1;continue
    band_energy_table=band_features.get(cue_row['id'])
    if band_energy_table is not None and len(band_energy_table)!=len(beat_energy_table): band_energy_table=None
    if stored_bar_zero_index<0:
        beat_energy_table=np.vstack([np.repeat(beat_energy_table[:1],-stored_bar_zero_index,0),beat_energy_table])
        if band_energy_table is not None: band_energy_table=np.concatenate([np.repeat(band_energy_table[:1],-stored_bar_zero_index,0),band_energy_table])
        stored_bar_zero_index=0
    beat_energy_table=beat_energy_table[stored_bar_zero_index:]
    if band_energy_table is not None: band_energy_table=band_energy_table[stored_bar_zero_index:]
    dataset.append(dict(BX=band_energy_table,id=cue_row['id'],name=cue_row['name'],T=period_seconds,bpm=60/period_seconds,phase=0,BE=beat_energy_table,hot={kind:cue['bar'] for kind,cue in hot_cues.items()},
        mem=sorted(cue['bar'] for cue in cue_row['cues'] if cue['kind']==0 and cue['bar'] is not None),pl=cue_row['playlists'],length=cue_row['length_s']))
print(len(dataset),skip_reasons,np.percentile(alignment_errors,[1,50,99]))
pickle.dump(dataset,open(str(paths.WORK / 'cue_dataset.pkl'),'wb'))
