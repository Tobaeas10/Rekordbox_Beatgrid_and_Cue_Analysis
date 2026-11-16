"""Run track_job.detect on audio files (temp cache) and print the final anchors with phase/override info."""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
import sys, tempfile, glob
from lib.beatgrid import track_job
cache_folder = tempfile.mkdtemp()
for track_index, audio_path in enumerate(sys.argv[1:]):
    events, sections = track_job.detect(cache_folder, audio_path, f'x{track_index}')
    section = sections[0]
    print(audio_path.split('/')[-1][:45], 'bpm', section['bpm'], 'anchor_ms', round(section['anchor_s'] * 1000, 1), 'phase_peaks', section.get('phase_peaks'),
          'phase_shift', section.get('phase_shift_ms'), 'override', section.get('snare_override_ms'), 'refine', (section.get('refine') or {}).get('shift_ms'), flush=True)
