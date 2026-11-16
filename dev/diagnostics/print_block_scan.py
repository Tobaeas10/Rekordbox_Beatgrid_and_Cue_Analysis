import _bootstrap  # noqa: F401  (puts the project root on sys.path)
import sys
import numpy as np
from lib.beatgrid import grid_pipeline

for audio_path in sys.argv[1:]:
    events = grid_pipeline.get_events(audio_path)
    blocks = grid_pipeline.block_scans(events)
    print(audio_path, 'dur %.1f' % events['dur'], 'kicks', events['nk'], 'snares', events['ns'])
    for block in blocks:
        scan_result = block['res']
        if scan_result:
            print('  %5.0f valid=%d bpm %8.2f z %6.1f frac %.2f n %3d %s' % (block['start'], block['valid'], scan_result['bpm'], scan_result['z'], scan_result['frac'], scan_result['n'], scan_result['template']))
        else:
            print('  %5.0f too few events' % block['start'])
