import _bootstrap  # noqa: F401  (puts the project root on sys.path)
import sys,subprocess,numpy as np
from scipy.signal import butter,sosfilt
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
audio_path,anchor_seconds,period_seconds,output_path=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),sys.argv[4]
start_seconds=float(sys.argv[5]) if len(sys.argv)>5 else 0; end_seconds=float(sys.argv[6]) if len(sys.argv)>6 else 1e9
sample_rate=44100
samples=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-i',audio_path,'-ac','1','-ar',str(sample_rate),'-f','f32le','-'],capture_output=True).stdout,np.float32)
bands={'sub 30-120':(30,120),'low 120-500':(120,500),'mid 500-3k':(500,3000),'hi 3k-12k':(3000,12000)}
bar_seconds=4*period_seconds
fig,axes=plt.subplots(len(bands),1,figsize=(14,9),sharex=True)
for band_index,(band_name,(low_hz,high_hz)) in enumerate(bands.items()):
    sos=butter(4,[low_hz,high_hz],'band',fs=sample_rate,output='sos'); filtered=np.abs(sosfilt(sos,samples))
    window_samples=int(sample_rate*0.002); envelope=np.convolve(filtered,np.ones(window_samples)/window_samples,'same')
    # fold over 2 bars
    two_bar_seconds=2*bar_seconds; fold_count=int((len(samples)/sample_rate-anchor_seconds)/two_bar_seconds)
    folds=[]
    for fold_index in range(fold_count):
        fold_start=anchor_seconds+fold_index*two_bar_seconds
        if fold_start<start_seconds or fold_start+two_bar_seconds>end_seconds: continue
        segment=envelope[int(fold_start*sample_rate):int((fold_start+two_bar_seconds)*sample_rate)]
        folds.append(segment[:int(two_bar_seconds*sample_rate)])
    min_length=int(min(len(fold) for fold in folds)); mean_fold=np.mean([fold[:min_length] for fold in folds],0)
    beat_positions=np.arange(min_length)/sample_rate/period_seconds
    axes[band_index].plot(beat_positions,mean_fold,lw=.7); axes[band_index].set_ylabel(band_name)
    for beat_index in range(0,9): axes[band_index].axvline(beat_index,color='r',lw=.5,alpha=.5)
    for beat_index in np.arange(0.5,8,1): axes[band_index].axvline(beat_index,color='gray',lw=.3,alpha=.5)
axes[-1].set_xlabel('beats from lattice anchor (red=beat, n=%d two-bar folds)'%len(folds))
plt.tight_layout(); plt.savefig(output_path,dpi=70)
