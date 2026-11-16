import _bootstrap  # noqa: F401  (puts the project root on sys.path)
import sys,glob,subprocess,numpy as np,time,json,pickle
from lib.beatgrid import grid_pipeline
from scipy.signal import butter,sosfilt
def load_mono_audio(audio_path,sample_rate=44100):
    return np.frombuffer(subprocess.run(['ffmpeg','-v','error','-i',audio_path,'-ac','1','-ar',str(sample_rate),'-f','f32le','-'],capture_output=True).stdout,np.float32),sample_rate
def fold_phase(samples,sample_rate,anchor_seconds,period_seconds,start_seconds,end_seconds):
    """folded positive-flux novelty per band; returns dict band->(peak offset ms from lattice, sharpness)"""
    peak_by_band={}
    for band_name,(low_hz,high_hz) in {'low':(40,300),'mid':(300,3000),'hi':(3000,12000)}.items():
        # 1 ms envelope via 44 samples (sr 44100 -> 44.1 ; use 44)
        filtered=np.abs(sosfilt(butter(4,[low_hz,high_hz],'band',fs=sample_rate,output='sos'),samples))
        bin_samples=44; bin_count=len(filtered)//bin_samples; envelope=filtered[:bin_count*bin_samples].reshape(bin_count,bin_samples).mean(1)  # ~1.0204 ms bins
        bin_seconds=bin_samples/sample_rate
        novelty=np.maximum(0,np.convolve(envelope,[0.25]*4,'same')-np.concatenate([np.zeros(6),np.convolve(envelope,[0.25]*4,'same')[:-6]]))
        bin_times=np.arange(bin_count)*bin_seconds
        in_section=(bin_times>=start_seconds)&(bin_times<=end_seconds)
        phase_offsets=((bin_times[in_section]-anchor_seconds+period_seconds/2)%period_seconds)-period_seconds/2
        histogram_edges=np.arange(-period_seconds/2,period_seconds/2+1e-9,0.002)
        histogram,_=np.histogram(phase_offsets,bins=histogram_edges,weights=novelty[in_section])
        histogram=np.convolve(histogram,np.ones(5)/5,'same')
        peak_index=int(np.argmax(histogram));
        peak_by_band[band_name]=(round(1000*(histogram_edges[peak_index]+0.001),1),round(float(histogram[peak_index]/ (np.median(histogram)+1e-12)),1))
    return peak_by_band
if __name__=='__main__':
    audio_path=sys.argv[1]
    samples,sample_rate=load_mono_audio(audio_path)
    events=grid_pipeline.get_events(audio_path,cache=False)
    blocks=grid_pipeline.block_scans(events);raw_sections=grid_pipeline.find_sections(blocks)
    fits=grid_pipeline.clean_sections(events,raw_sections,[grid_pipeline.fit_section(events,raw_section) for raw_section in raw_sections])
    for section in fits[:2]:
        print(audio_path.split('/')[-1][:40],section['start_s'],section['end_s'],section['bpm'],'anchor',round(section['anchor_s']*1000,1),fold_phase(samples,sample_rate,section['anchor_s'],section['period_s'],section['start_s'],section['end_s']))
