import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import sys, os, json, glob, math
import numpy as np
from lib.beatgrid import grid_pipeline

AUDIO = ('.mp3', '.m4a', '.wav')


def wrap(value, period):
    return (value + period / 2) % period - period / 2


def compare(folder):
    reference = json.load(open(os.path.join(folder, 'ground_truth.json'), encoding='utf-8'))
    audio_path = next(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(AUDIO))
    cache = os.path.join(str(paths.WORK / 'event_cache'), os.path.basename(audio_path) + '.events.pkl')
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    events = grid_pipeline.get_events(audio_path) if not os.path.exists(cache) else None
    if events is None:
        import pickle
        events = pickle.load(open(cache, 'rb'))
    else:
        import pickle
        pickle.dump(events, open(cache, 'wb'))
    blocks = grid_pipeline.block_scans(events)
    sections = [fitted for fitted in (grid_pipeline.fit_section(events, raw_section) for raw_section in grid_pipeline.find_sections(blocks)) if fitted]
    rekordbox_beat_ms = np.array([b[2] for b in reference['rb_grid']['beats']], float)
    ground_truth = reference['ground_truth']
    rows = []
    for section in sections:
        period_ms = 1000 * section['period_s']
        anchor_ms = 1000 * section['anchor_s']
        inside_section = (rekordbox_beat_ms >= 1000 * section['start_s']) & (rekordbox_beat_ms <= 1000 * section['end_s'])
        if inside_section.sum() < 8:
            rows.append(dict(sec=section, n=int(inside_section.sum()))); continue
        offset_mod_beat = wrap(rekordbox_beat_ms[inside_section] - anchor_ms, period_ms)           # RB beat minus my beat lattice (mod beat)
        offset_mod_half_beat = wrap(rekordbox_beat_ms[inside_section] - anchor_ms, period_ms / 2)       # mod half beat
        offset_ms = float(np.median(offset_mod_beat))
        residuals_ms = offset_mod_beat - offset_ms
        rows.append(dict(sec=section, n=int(inside_section.sum()), off=offset_ms, off_half=float(np.median(offset_mod_half_beat)),
                         p05=float(np.percentile(residuals_ms, 5)), p95=float(np.percentile(residuals_ms, 95)),
                         worst=float(np.abs(residuals_ms).max()), rms=float(np.sqrt(np.mean(residuals_ms ** 2)))))
    return dict(folder=os.path.basename(folder), audio=os.path.basename(audio_path), ext=audio_path[-4:], gt_bpm=ground_truth['bpm'] if ground_truth else None,
                rb_bpm=reference['rekordbox_bpm'], gt_rms=ground_truth['fit_rms_ms'] if ground_truth else None, rows=rows, dur=events['dur'], smpb=reference.get('itunsmpb'),
                sr=reference['sample_rate'])


if __name__ == '__main__':
    results = []
    for folder in sorted(glob.glob(sys.argv[1] + '/*/')):
        try:
            result = compare(folder.rstrip('/'))
        except Exception as e:
            print(folder, 'ERROR', repr(e), flush=True); continue
        results.append(result)
        print(f"{result['folder'][:44]:44s} {result['ext']} sr{result['sr']} RBbpm {result['rb_bpm']} gtbpm {result['gt_bpm'] and round(result['gt_bpm'],3)} dur {result['dur']:.0f}", flush=True)
        for row in result['rows']:
            section = row['sec']
            if 'off' not in row:
                print(f"     sec {section['start_s']:.0f}-{section['end_s']:.0f} bpm {section['bpm']} (only {row['n']} RB beats inside)", flush=True); continue
            print(f"     sec {section['start_s']:5.0f}-{section['end_s']:5.0f} bpm {section['bpm']:9.4f} | offset(RB-mine) {row['off']:+7.1f} ms (mod T/2 {row['off_half']:+6.1f}) | "
                  f"spread p5..p95 {row['p05']:+5.1f}..{row['p95']:+5.1f} worst {row['worst']:5.1f} rms {row['rms']:4.1f} (n={row['n']})", flush=True)
    json.dump(results, open(str(paths.WORK / 'compare_against_ground_truth.json'), 'w'), default=str, indent=1)
