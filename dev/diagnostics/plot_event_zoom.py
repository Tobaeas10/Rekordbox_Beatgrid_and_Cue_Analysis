import _bootstrap  # noqa: F401  (puts the project root on sys.path)
import sys,pickle,subprocess,numpy as np
from scipy.signal import butter,sosfiltfilt
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
audio_path,events_path,anchor_seconds,period_seconds,beat_in_bar,output_path=sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),sys.argv[6]
sample_rate=44100
samples=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-i',audio_path,'-ac','1','-ar',str(sample_rate),'-f','f32le','-'],capture_output=True).stdout,np.float32)
events=pickle.load(open(events_path,'rb')); event_times,event_classes=events['t'],events['c']
snare_filtered=np.abs(sosfiltfilt(butter(4,[2000,12000],'band',fs=sample_rate,output='sos'),samples))
window_samples=int(sample_rate*.001); envelope=np.convolve(snare_filtered,np.ones(window_samples)/window_samples,'same')
fig,axes=plt.subplots(4,1,figsize=(14,9))
for panel_index,bar_index in enumerate([30,40,50,60]):
    center_seconds=anchor_seconds+(bar_index*4+beat_in_bar)*period_seconds
    first_sample,last_sample=int((center_seconds-.15)*sample_rate),int((center_seconds+.4)*sample_rate)
    milliseconds=(np.arange(first_sample,last_sample)/sample_rate-center_seconds)*1000
    axes[panel_index].plot(milliseconds,envelope[first_sample:last_sample],lw=.6)
    near_center=(event_times>center_seconds-.15)&(event_times<center_seconds+.4)
    for event_time,event_class in zip(event_times[near_center],event_classes[near_center]): axes[panel_index].axvline((event_time-center_seconds)*1000,color='r' if event_class==1 else 'g',lw=1)
    axes[panel_index].axvline(0,color='k',lw=.5)
plt.tight_layout();plt.savefig(output_path,dpi=70)
