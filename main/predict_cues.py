#!/usr/bin/env python3
"""predict_cues.py - predict hot cues A-H and memory cues for tracks and write them into master.db (same selection options as predict_beatgrids.py).
Default selection: tracks added in the last 8 hours. WRITES by default (--dry-run = predict and report only).
Only tracks that predict_beatgrids.py analysed (data/state/beatgrid_analysis_state.json) are used: the grid stored in the track's ANLZ .DAT (PQTZ) is the scaffold, bar 0 = its first downbeat.

  python predict_cues.py                          last 8 h
  python predict_cues.py --since-hours 24
  python predict_cues.py --from 2026-11-01 --to 2026-11-09
  python predict_cues.py --playlist Dubstep       whole playlist
  python predict_cues.py --names "Strawman,Poison"
  python predict_cues.py --force                  redo tracks done before: deletes the cues THIS script wrote earlier (data/state/cue_analysis_state.json), rewrites them
  python predict_cues.py --dry-run

Safety: tracks that have ANY cue not written by this script are skipped (nothing of yours is ever overwritten or deleted); tracks with several tempo
sections / no constant grid (never) and, only with --skip-likely-wrong, tracks whose grid predict_beatgrids flagged 'Likely wrong'.
New rows are clones of an existing DjmdCue of the same Kind (library colour convention). master.db is backed up first. Works with Rekordbox
open like predict_beatgrids.py (restart Rekordbox afterwards to see the cues). Hot cue Kind: A,B,C = 1,2,3; D..H = 5..9; memory = 0."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import shutil
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import _bootstrap
from lib import adaptive_workers, paths
OUT = paths.LOGS / 'cues'; BACKUP = paths.BACKUPS / 'cues'; KEEP_BACKUPS = 5
STATE = paths.STATE_DIR / 'cue_analysis_state.json'; BEATGRID_STATE = paths.STATE_DIR / 'beatgrid_analysis_state.json'
HOTKIND = {'A': 1, 'B': 2, 'C': 3, 'D': 5, 'E': 6, 'F': 7, 'G': 8, 'H': 9}

def pjob(job_arguments):
    """worker: features + prediction for one track -> (id, pred dict | dict(error=...))"""
    try:
        from lib.cues import cue_predictor
        track, error_message = cue_predictor.grid_features(job_arguments['src'], job_arguments['beats'], job_arguments['pl'])
        if track is None: return job_arguments['id'], dict(error=error_message)
        prediction = cue_predictor.predict(track)
        return job_arguments['id'], (dict(error='no prediction') if prediction is None else {field_name: (None if field_value is None else float(field_value)) for field_name, field_value in prediction.items()})
    except Exception as e:
        return job_arguments['id'], dict(error=repr(e)[:120])

def add_selection_args(argument_parser):
    """identical selection options to predict_beatgrids.py"""
    argument_parser.add_argument('--since-hours', type=float, default=8)
    argument_parser.add_argument('--from', dest='date_from'); argument_parser.add_argument('--to', dest='date_to')
    argument_parser.add_argument('--playlist', action='append'); argument_parser.add_argument('--names')

def plan_cues(prediction, times, first_downbeat_index):
    """pred (bars) -> list of dict(kind,label,bar,ms) sorted by time. Hot cue letters + memory cues."""
    from lib.cues import cue_predictor
    hot_bars, memory_bars = cue_predictor.cues_for(prediction)
    planned_cues = [dict(kind=HOTKIND[letter], label=letter, bar=float(bar), ms=int(round(cue_predictor.bar_ms(times, first_downbeat_index, bar)))) for letter, bar in sorted(hot_bars.items())]
    planned_cues += [dict(kind=0, label='memory', bar=float(bar), ms=int(round(cue_predictor.bar_ms(times, first_downbeat_index, bar)))) for bar in memory_bars]
    last_ms = times[-1]
    return sorted([cue for cue in planned_cues if 0 <= cue['ms'] <= last_ms], key=lambda cue: (cue['ms'], cue['kind']))

def clone_cue(database, masterdb_models, template, content, cue, new_id, now):
    """New DjmdCue = copy of template's column values with identity/time fields replaced."""
    column_values = {column.name: getattr(template, column.name) for column in masterdb_models.DjmdCue.__table__.columns}
    cue_ms = cue['ms']
    column_values.update(ID=str(new_id), ContentID=str(content.ID), ContentUUID=content.UUID, UUID=str(uuid.uuid4()),
                InMsec=cue_ms, InFrame=int(round(cue_ms * 150 / 1000.0)), InMpegFrame=0, InMpegAbs=0,
                OutMsec=-1, OutFrame=0, OutMpegFrame=0, OutMpegAbs=0, Kind=cue['kind'], ActiveLoop=0,
                Comment='', InPointSeekInfo='', OutPointSeekInfo='', rb_local_deleted=0, created_at=now, updated_at=now)
    return masterdb_models.DjmdCue(**column_values)

def write_track(database, masterdb_models, rekordbox_database, content, cues, templates, now):
    rows = []
    for cue in cues:
        template = templates.get(cue['kind'])
        if template is None: raise RuntimeError('no template cue for kind %d' % cue['kind'])
        row = clone_cue(database, masterdb_models, template, content, cue, database.generate_unused_id(masterdb_models.DjmdCue), now)
        if hasattr(rekordbox_database, 'frame_of'): row.InFrame = rekordbox_database.frame_of(cue['ms'])
        database.add(row); rows.append(row)
    return rows

def find_templates(database, masterdb_models):
    """one template cue per Kind, taken from the library's MOST COMMON colour for that Kind (so A/B get the user's red/blue, not whatever old
    uncoloured cue happened to come first), preferring plain cues (OutMsec -1)."""
    from collections import defaultdict
    rows_by_kind = defaultdict(list)
    for cue_row in database.query(masterdb_models.DjmdCue).filter(masterdb_models.DjmdCue.rb_local_deleted == 0).all(): rows_by_kind[int(cue_row.Kind or 0)].append(cue_row)
    templates_by_kind = {}
    for kind, kind_rows in rows_by_kind.items():
        plain_rows = [cue_row for cue_row in kind_rows if cue_row.OutMsec in (-1, None)] or kind_rows
        colour_counts = Counter((getattr(cue_row, 'Color', None), getattr(cue_row, 'ColorTableIndex', None)) for cue_row in plain_rows)
        # if kind != 0: print(f'  kind {kind}: colour (Color, ColorTableIndex) usage in library: {colour_counts.most_common(4)}')
        modal_colour = colour_counts.most_common(1)[0][0]
        templates_by_kind[kind] = next(cue_row for cue_row in plain_rows if (getattr(cue_row, 'Color', None), getattr(cue_row, 'ColorTableIndex', None)) == modal_colour)
    any_cue = next(iter(templates_by_kind.values()), None)
    return templates_by_kind, any_cue

def recolor_written_cues(database, masterdb_models, state, templates, database_path):
    """repair: give the hot cues THIS script wrote earlier (state file) the library's colour for their Kind. Nothing else is touched."""
    own_ids = {cue_id for entry in state.values() for cue_id in entry.get('cue_ids', [])}; changed = 0
    for cue_row in database.query(masterdb_models.DjmdCue).filter(masterdb_models.DjmdCue.rb_local_deleted == 0).all():
        kind = int(cue_row.Kind or 0); template = templates.get(kind)
        if str(cue_row.ID) not in own_ids or kind == 0 or template is None: continue
        for column in ('Color', 'ColorTableIndex'):
            if hasattr(cue_row, column) and getattr(cue_row, column) != getattr(template, column, None): setattr(cue_row, column, getattr(template, column, None)); changed += 1
    return changed

def rotate_backups():
    for backup_file in sorted(BACKUP.glob('master_*.db'))[:-KEEP_BACKUPS]: backup_file.unlink()

def delete_own(database, masterdb_models, track_id, cue_ids):
    deleted_count = 0
    for cue_row in database.query(masterdb_models.DjmdCue).filter(masterdb_models.DjmdCue.ContentID == track_id).all():
        if str(cue_row.ID) in cue_ids: database.session.delete(cue_row); deleted_count += 1
    return deleted_count

def main(argv=None):
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selection_args(argument_parser)
    argument_parser.add_argument('--force', action='store_true', help='redo tracks done before (replaces the cues this script wrote earlier)')
    argument_parser.add_argument('--dry-run', action='store_true', help='predict and report, write nothing')
    argument_parser.add_argument('--minimum-confidence', type=float, default=0.0, help='do not write tracks whose D or e1 probability is below this (still reported)')
    argument_parser.add_argument('--skip-likely-wrong', action='store_true', help="do not write cues for tracks predict_beatgrids flagged 'Likely wrong' (default: write for every track)")
    argument_parser.add_argument('--allow-unanalysed', action='store_true', help='also use tracks that predict_beatgrids.py did not analyse (grid from Rekordbox itself; unreliable)')
    argument_parser.add_argument('--recolor', action='store_true', help="only repair: give the cues this script wrote earlier the library's colour per Kind, then exit")
    adaptive_workers.add_argument(argument_parser)
    argument_parser.add_argument('--database-path'); argument_parser.add_argument('--encryption-key'); argument_parser.add_argument('--share-directory')
    arguments, _ = argument_parser.parse_known_args(argv)
    controller = adaptive_workers.WorkerController(arguments.force_number_of_jobs)   # starts measuring the load right away
    # predict_beatgrids-only flags are ignored so the same command line can be given to both scripts
    from lib.rekordbox import ground_truth
    from lib.rekordbox import database as rekordbox_database
    from lib.cues import cue_export
    from lib.cues import cue_predictor
    from lib.rekordbox import track_selection
    from lib.rekordbox.database import DjmdContent
    environment = ground_truth.load_env(paths.ENV_FILE)
    database_path, key = arguments.database_path or environment.get('REKORDBOX_DATABASE_PATH'), arguments.encryption_key or environment.get('REKORDBOX_ENCRYPTION_KEY', '')
    if not database_path: sys.exit('REKORDBOX_DATABASE_PATH missing (.env or --database-path)')
    database = rekordbox_database.open_database(database_path, key); share = rekordbox_database.share_directory_of(database, SimpleNamespace(share_directory=arguments.share_directory, database_path=database_path))
    state = json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}
    if arguments.recolor:
        from pyrekordbox.masterdb import models as masterdb_models
        templates, _ = find_templates(database, masterdb_models)
        BACKUP.mkdir(exist_ok=True); shutil.copy2(database_path, BACKUP / f'master_{datetime.now():%Y%m%d_%H%M%S}.db'); rotate_backups()
        changed = recolor_written_cues(database, masterdb_models, state, templates, database_path)
        if rekordbox_database.rekordbox_running():
            import pyrekordbox.masterdb.database as masterdb_database
            masterdb_database.get_rekordbox_pid = lambda: 0
        import logging; logging.getLogger('pyrekordbox').setLevel(logging.ERROR)
        database.commit(); print(f'recolor: {changed} colour field(s) updated. Restart Rekordbox.'); return
    rows = track_selection.select(database, arguments, ground_truth, DjmdContent)
    beatgrid_state = json.loads(BEATGRID_STATE.read_text(encoding='utf-8')) if BEATGRID_STATE.exists() else {}
    playlists_by_content_id = cue_export.playlist_map(database, ground_truth, rekordbox_database); existing_cue_ids = {}
    for cue_row in database.query(rekordbox_database.DjmdCue).filter(rekordbox_database.DjmdCue.rb_local_deleted == 0).all(): existing_cue_ids.setdefault(str(cue_row.ContentID), []).append(str(cue_row.ID))
    skipped = Counter(); jobs = []; meta = {}
    for content_row in rows:
        track_id = str(content_row.ID); name = f'{ground_truth.artist_of(content_row)} - {content_row.Title}'
        own_cue_ids = set(state.get(track_id, {}).get('cue_ids', []))
        if track_id in state and not arguments.force: skipped['already done'] += 1; continue
        if set(existing_cue_ids.get(track_id, [])) - own_cue_ids: skipped['has cues of its own'] += 1; continue
        beatgrid_entry = beatgrid_state.get(track_id)
        if beatgrid_entry is None and not arguments.allow_unanalysed: skipped['not analysed by predict_beatgrids'] += 1; continue
        if beatgrid_entry is not None and (beatgrid_entry.get('sections') or 1) > 1: skipped['several tempo sections'] += 1; continue
        if beatgrid_entry is not None and beatgrid_entry.get('review_level') == 'wrong' and arguments.skip_likely_wrong: skipped["grid flagged 'Likely wrong'"] += 1; continue
        try:
            files = rekordbox_database.anlz_files(share, content_row)
            if 'DAT' not in files: skipped['no DAT'] += 1; continue
            beats = ground_truth.read_pqtz(files['DAT'].read_bytes())
            times, first_downbeat_index, multi = cue_predictor.grid_info(beats) if len(beats) >= 4 else (None, None, False)
            if first_downbeat_index is None or multi or not Path(content_row.FolderPath or '').is_file(): skipped['no/non-constant grid or audio missing'] += 1; continue
        except Exception as e: skipped['error ' + repr(e)[:50]] += 1; continue
        jobs.append(dict(id=track_id, src=str(content_row.FolderPath), beats=beats, pl=sorted(playlists_by_content_id.get(content_row.ID, []))))
        meta[track_id] = (content_row, name, times, first_downbeat_index, own_cue_ids)
    print(f'{track_selection.describe(arguments)}: {len(rows)} selected, {len(jobs)} to predict, skipped {dict(skipped)}' + ('   [DRY RUN]' if arguments.dry_run else ''), flush=True)
    if not jobs: controller.stop(); return
    OUT.mkdir(exist_ok=True); plans = []; start_time = time.time()
    position = 0; waiting_jobs = list(jobs); running = {}
    with ProcessPoolExecutor(max_workers=controller.maximum_workers) as executor:
        while waiting_jobs or running:
            controller.update(len(running))
            while waiting_jobs and len(running) < controller.allowed_workers(): running[executor.submit(pjob, waiting_jobs.pop(0))] = True
            finished, _ = wait(list(running), timeout=0.5, return_when=FIRST_COMPLETED) if running else (set(), time.sleep(0.1))
            for finished_future in finished:
                del running[finished_future]; position += 1
                track_id, prediction = finished_future.result()
                content_row, name, times, first_downbeat_index, own_cue_ids = meta[track_id]
                if 'error' in prediction: skipped['predict: ' + prediction['error'][:40]] += 1; print(f'[{position}/{len(jobs)}] {name}: FAILED {prediction["error"]}', flush=True); continue
                cues = plan_cues(prediction, times, first_downbeat_index)
                plans.append(dict(id=track_id, name=name, pred=prediction, cues=cues))
                print(f'[{position}/{len(jobs)}] {name}: D={prediction["D"]:.0f} e1={prediction["e1"]:.0f} H={prediction["H"]} e2={prediction["e2"]} fk1={prediction["fk1"]} ({len(cues)} cues, conf D {prediction["conf_D"]:.2f} e1 {prediction["conf_e1"]:.2f}, {controller.live_workers()} worker(s))', flush=True)
    controller.stop()
    with open(OUT / 'phrase_cues.jsonl', 'w', encoding='utf-8') as output_file:
        for plan in plans: output_file.write(json.dumps(plan, ensure_ascii=False) + '\n')
    if arguments.dry_run: print(f'Dry run: nothing written. {len(plans)} plan(s) in {OUT / "phrase_cues.jsonl"}'); return
    if not plans: return
    from pyrekordbox.masterdb import models as masterdb_models
    templates, any_cue = find_templates(database, masterdb_models)
    if any_cue is None: sys.exit('no existing cue in the library to clone from; create one hot cue and one memory cue manually in Rekordbox once')
    for kind in set(cue['kind'] for plan in plans for cue in plan['cues']):
        if kind not in templates: templates[kind] = any_cue; print(f'note: no existing cue of kind {kind}, cloning another kind (colour may differ)')
    BACKUP.mkdir(exist_ok=True); shutil.copy2(database_path, BACKUP / f'master_{datetime.now():%Y%m%d_%H%M%S}.db'); rotate_backups()
    now = datetime.now(timezone.utc); written_count = replaced_count = 0; wrote = {}
    for plan in plans:
        if plan['pred']['conf_D'] < arguments.minimum_confidence or plan['pred']['conf_e1'] < arguments.minimum_confidence: print('skip (low confidence)', plan['name']); continue
        content_row, name, times, first_downbeat_index, own_cue_ids = meta[plan['id']]
        try:
            replaced_count += delete_own(database, masterdb_models, plan['id'], own_cue_ids)
            newrows = write_track(database, masterdb_models, rekordbox_database, content_row, plan['cues'], templates, now); written_count += 1
            wrote[plan['id']] = (plan, newrows)
        except Exception as e: database.session.rollback(); print('FAILED', plan['name'], repr(e)); break
    if rekordbox_database.rekordbox_running():
        import pyrekordbox.masterdb.database as masterdb_database
        masterdb_database.get_rekordbox_pid = lambda: 0  # pyrekordbox refuses commits while Rekordbox runs (same as predict_beatgrids.py)
    import logging; logging.getLogger('pyrekordbox').setLevel(logging.ERROR)
    database.commit()
    for track_id, (plan, newrows) in wrote.items():
        state[track_id] = dict(name=plan['name'], done=datetime.now().isoformat(timespec='seconds'), cue_ids=[str(new_row.ID) for new_row in newrows], n_cues=len(newrows),
                          D=plan['pred']['D'], e1=plan['pred']['e1'], H=plan['pred']['H'], e2=plan['pred']['e2'], conf_D=plan['pred']['conf_D'], conf_e1=plan['pred']['conf_e1'])
    STATE.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'master.db: cues written for {written_count} track(s) ({replaced_count} earlier cue(s) of this script replaced); backup in {BACKUP}, {time.time() - start_time:.0f}s')
    if written_count: print('IMPORTANT: restart Rekordbox to see the new cues.')

if __name__ == '__main__': main()