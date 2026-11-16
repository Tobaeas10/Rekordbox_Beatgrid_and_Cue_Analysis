import _bootstrap  # noqa: F401  (puts the project root on sys.path)
import sys,pickle,numpy as np
from lib.beatgrid import grid_pipeline
events_path,anchor_seconds,period_seconds=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
events=pickle.load(open(events_path,'rb'))
event_times,event_classes,event_weights=events['t'],events['c'],events['w']
for event_class,event_name in((0,'kick'),(1,'snare')):
    selected=(event_classes==event_class)&(event_times>anchor_seconds)
    residuals_ms=((event_times[selected]-anchor_seconds+period_seconds/2)%period_seconds-period_seconds/2)*1000
    histogram,_=np.histogram(residuals_ms,bins=np.arange(-214,215,10))
    print(event_name,selected.sum(),'hist(10ms bins from -214):',histogram.tolist())
    # beat-position histogram (in beats mod 4) for those near lattice +-30ms
    beat_positions=((event_times[selected]-anchor_seconds)/period_seconds)%4; near_lattice_residuals_ms=((event_times[selected]-anchor_seconds+period_seconds/2)%period_seconds-period_seconds/2)*1000
    print(' by beat (|r|<30):',[int(((np.abs(near_lattice_residuals_ms)<30)&(np.floor(beat_positions+.5)%4==beat_index)).sum()) for beat_index in range(4)])
