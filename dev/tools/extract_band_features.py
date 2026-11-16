"""Per-8th-note 10-band features for the whole library (training data for the cue models). Needs the slim beat-feature pickle and cues.jsonl."""
import _bootstrap
import argparse
import gzip
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from lib import paths
from lib.audio_features import compute_band_energies


def job(job_arguments):
    try:
        from lib.beatgrid import tempo_detection
        samples, sample_rate = tempo_detection.load_audio(job_arguments['src'], 44100)   # fixed rate: skips the ffprobe call
        if sample_rate < 24000: return job_arguments['id'], None, 'sr %d too low' % sample_rate
        return job_arguments['id'], compute_band_energies(samples, sample_rate, job_arguments['t0'], job_arguments['T'], job_arguments['nb']), None
    except Exception as e:
        return job_arguments['id'], None, repr(e)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--beat-features-slim-path', required=True); argument_parser.add_argument('--cues-path', required=True); argument_parser.add_argument('--output-path', default='bandx2_slim.pkl.gz')
    argument_parser.add_argument('--parallel-workers', type=int, default=4); argument_parser.add_argument('--track-limit', type=int, default=0); argument_parser.add_argument('--database-path'); argument_parser.add_argument('--encryption-key'); argument_parser.add_argument('--share-directory')
    arguments = argument_parser.parse_args()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    slim_features = pickle.load(gzip.open(arguments.beat_features_slim_path, 'rb'))
    track_ids = [json.loads(line)['id'] for line in open(arguments.cues_path, encoding='utf-8')]
    track_ids = [i for i in track_ids if i in slim_features]
    if arguments.track_limit: track_ids = track_ids[:arguments.track_limit]
    part = arguments.output_path + '.part'
    done = pickle.load(open(part, 'rb')) if os.path.exists(part) else {}
    todo = [i for i in track_ids if i not in done]
    print(len(track_ids), 'tracks,', len(done), 'already done', flush=True)
    from lib.rekordbox import ground_truth
    from lib.rekordbox import database as rekordbox_database
    from lib.rekordbox.database import DjmdContent
    from types import SimpleNamespace
    environment = ground_truth.load_env(paths.ENV_FILE)
    database_path, key = arguments.database_path or environment.get('REKORDBOX_DATABASE_PATH'), arguments.encryption_key or environment.get('REKORDBOX_ENCRYPTION_KEY', '')
    database = rekordbox_database.open_database(database_path, key)
    todo_set = set(todo)
    rows = {str(content_row.ID): content_row for content_row in database.query(DjmdContent).all() if not content_row.rb_local_deleted and str(content_row.ID) in todo_set}
    jobs = []
    for i in todo:
        content_row = rows.get(i)
        if content_row is None or not Path(content_row.FolderPath or '').is_file(): print('no file', i); continue
        feature_entry = slim_features[i]
        jobs.append(dict(id=i, src=str(content_row.FolderPath), t0=float(feature_entry['info_lattice_t0_s']), T=float(feature_entry['info_period_s']), nb=len(feature_entry['BE'])))
    database.close()
    start_time = time.time(); failures = {}
    with ProcessPoolExecutor(max_workers=max(1, arguments.parallel_workers)) as executor:
        futures = [executor.submit(job, job_arguments) for job_arguments in jobs]
        for completed_count, finished_future in enumerate(as_completed(futures), 1):
            i, band_energies, error_message = finished_future.result()
            if band_energies is None: failures[i] = error_message; print('FAIL', i, error_message, flush=True)
            else: done[i] = band_energies
            if completed_count % 25 == 0:
                print('%d/%d %.0fs' % (completed_count, len(jobs), time.time() - start_time), flush=True); pickle.dump(done, open(part, 'wb'))
    pickle.dump(done, open(part, 'wb'))
    with gzip.open(arguments.output_path, 'wb', 4) as output_file: pickle.dump(done, output_file)
    print('saved', arguments.output_path, len(done), 'tracks;', len(failures), 'failed', flush=True)


if __name__ == '__main__':
    main()


if __name__ == "__main__":
    main()
