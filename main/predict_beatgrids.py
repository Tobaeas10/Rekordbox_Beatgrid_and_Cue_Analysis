"""Beat grid analysis for new Rekordbox tracks: detects the grid and writes it into each track's ANLZ .DAT (PQTZ).
Default selection: tracks added in the last 8 hours. Tracks done before (data/state/beatgrid_analysis_state.json) are skipped unless --force.
Waits for Rekordbox's own analysis files of a track (needed for the mp3 offset), then overwrites the .DAT.

  python predict_beatgrids.py                          last 8 h
  python predict_beatgrids.py --since-hours 24
  python predict_beatgrids.py --from 2026-11-01 --to 2026-11-09
  python predict_beatgrids.py --playlist Dubstep       whole playlist
  python predict_beatgrids.py --names "Strawman,Poison" file/title contains
  python predict_beatgrids.py --force --dry-run        redo everything, write nothing
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import _bootstrap
from lib import adaptive_workers, paths
from lib.rekordbox.track_selection import added_at, describe, select
STATE = paths.STATE_DIR / "beatgrid_analysis_state.json"
BACKUP = paths.BACKUPS / "beatgrid"  # safe to delete
PENDING = paths.STATE_DIR / "pending_bpm.json"
PENDING_REVIEW = paths.STATE_DIR / "pending_review.json"
KEEP_BACKUPS = 5  # newest master.db backups kept
CACHE = paths.CACHE / "beatgrid"


def job(job_arguments):
    from lib.beatgrid import grid_writer
    try:
        result = grid_writer.compute_grid(job_arguments["src"], job_arguments["ext"], job_arguments["cache"], job_arguments["id"], job_arguments.get("rb_bpm"))
        result["secs"] = [dict(start_s=section["start_s"], end_s=section["end_s"], bpm=section["bpm"]) for section in result["secs"]]
        result.pop("built_extra", None)
        return job_arguments["id"], result
    except Exception as e:
        return job_arguments["id"], dict(error=repr(e))


def stable(files):
    sizes = {analysis_kind: (analysis_file.stat().st_size, analysis_file.stat().st_mtime) for analysis_kind, analysis_file in files.items() if analysis_kind in ("DAT", "EXT")}
    return sizes if len(sizes) == 2 else None


def snap_rows(database, rekordbox_database, share, rows, margin):
    """Move cue/loop times of the given tracks onto the nearest beat of their (new) grid within margin ms (logic of snap_cues_to_grid.py)."""
    from lib.rekordbox import cue_snapping
    snapped_count = 0
    for content_row in rows:
        files = rekordbox_database.anlz_files(share, content_row)
        if "DAT" not in files:
            continue
        beats = cue_snapping.beats_from_dat(files["DAT"].read_bytes())
        if not beats:
            continue
        content_id = str(content_row.ID)
        for cue_row in database.query(cue_snapping.DjmdCue).filter(cue_snapping.DjmdCue.ContentID == content_id).all():
            for millisecond_field, frame_field in (("InMsec", "InFrame"), ("OutMsec", "OutFrame")):
                value = getattr(cue_row, millisecond_field)
                if value is not None and value >= 0:
                    snapped_value = cue_snapping.snap(value, beats, margin)
                    if snapped_value != value:
                        setattr(cue_row, millisecond_field, snapped_value)
                        setattr(cue_row, frame_field, rekordbox_database.frame_of(snapped_value))
                        snapped_count += 1
        for content_cue_row in database.query(cue_snapping.ContentCue).filter(cue_snapping.ContentCue.ContentID == content_id).all():
            try:
                data = json.loads(content_cue_row.Cues)
            except Exception:
                continue
            changed = False
            for cue_entry in data:
                for millisecond_field, frame_field in (("InMsec", "InFrame"), ("OutMsec", "OutFrame")):
                    value = cue_entry.get(millisecond_field)
                    if value is not None and value >= 0:
                        snapped_value = cue_snapping.snap(value, beats, margin)
                        if snapped_value != value:
                            cue_entry[millisecond_field], cue_entry[frame_field], changed = snapped_value, rekordbox_database.frame_of(snapped_value), True
            if changed:
                content_cue_row.Cues = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        for extension, analysis_path in files.items():
            if extension == "2EX":
                continue
            patched, changes = cue_snapping.snap_anlz(analysis_path.read_bytes(), beats, margin)
            if changes:
                analysis_path.write_bytes(patched)
    return snapped_count


def rotate_backups():
    old_backups = sorted(BACKUP.glob("master_*.db"))
    for backup_file in old_backups[:-KEEP_BACKUPS]:
        backup_file.unlink(missing_ok=True)


def add_review_playlist(database, flagged_ids):
    """ids = {track_id: "wrong"|"uncertain"} (a plain list counts as uncertain). Each track goes into the playlist of its level
    (RV.PLAYLISTS, created at the root when missing); tracks already in it are skipped. A track that was Uncertain earlier and is Likely Wrong
    now is added to the Likely Wrong playlist too. Returns the number of tracks added."""
    from lib.beatgrid import review_flags
    from pyrekordbox.masterdb import models as masterdb_models
    if not isinstance(flagged_ids, dict):
        flagged_ids = {flagged_track_id: "uncertain" for flagged_track_id in flagged_ids}
    added_count = 0
    for level, name in review_flags.PLAYLISTS.items():
        wanted_track_ids = [flagged_track_id for flagged_track_id, flagged_level in flagged_ids.items() if flagged_level == level]
        if not wanted_track_ids:
            continue
        playlist = database.query(masterdb_models.DjmdPlaylist).filter(masterdb_models.DjmdPlaylist.Name == name, masterdb_models.DjmdPlaylist.Attribute == 0).first()
        if playlist is None:
            playlist = database.create_playlist(name)
        existing_content_ids = {str(playlist_entry.ContentID) for playlist_entry in database.query(masterdb_models.DjmdSongPlaylist).filter(masterdb_models.DjmdSongPlaylist.PlaylistID == playlist.ID).all()}
        for track_id in wanted_track_ids:
            if str(track_id) in existing_content_ids:
                continue
            content_row = database.query(masterdb_models.DjmdContent).filter(masterdb_models.DjmdContent.ID == str(track_id)).first()
            if content_row is not None:
                database.add_to_playlist(playlist, content_row)
                added_count += 1
    return added_count


def _merge_review(*parts):
    """Merge review id collections (dict id->level, or old plain lists = uncertain); wrong beats uncertain."""
    merged = {}
    for part in parts:
        for track_id, review_level in (part.items() if isinstance(part, dict) else [(plain_track_id, "uncertain") for plain_track_id in part]):
            if merged.get(track_id) != "wrong":
                merged[track_id] = review_level
    return merged


def apply_db(database, database_path, rekordbox_database, DjmdContent, updates, share, rows_to_snap, margin, defer, flagged=None):
    flagged = flagged or {}
    """DjmdContent.BPM (bpm x 100) of the tracks written plus cue snapping, one commit. defer = keep the bpm updates for a later run (Rekordbox closed)."""
    if defer:
        PENDING.write_text(json.dumps(updates), encoding="utf-8")
        if flagged:
            pending_flagged = json.loads(PENDING_REVIEW.read_text()) if PENDING_REVIEW.exists() else {}
            PENDING_REVIEW.write_text(json.dumps(_merge_review(pending_flagged, flagged)), encoding="utf-8")
        print(f"{len(updates)} bpm update(s) kept in {PENDING.name}, applied by the next run")
        return
    PENDING.write_text(json.dumps(updates), encoding="utf-8")
    BACKUP.mkdir(exist_ok=True)
    shutil.copy2(database_path, BACKUP / f"master_{datetime.now():%Y%m%d_%H%M%S}.db")
    rotate_backups()
    updated_count = 0
    for track_id, bpm in updates.items():
        content_row = database.query(DjmdContent).filter(DjmdContent.ID == track_id).first()
        if content_row is not None and content_row.BPM != bpm:
            content_row.BPM = bpm
            updated_count += 1
    snapped_count = snap_rows(database, rekordbox_database, share, rows_to_snap, margin) if rows_to_snap else 0
    flagged_ids = _merge_review(flagged, json.loads(PENDING_REVIEW.read_text()) if PENDING_REVIEW.exists() else {})
    if rekordbox_database.rekordbox_running():
        import pyrekordbox.masterdb.database as masterdb_database
        masterdb_database.get_rekordbox_pid = lambda: 0  # pyrekordbox refuses commits while Rekordbox runs
    import logging
    logging.getLogger("pyrekordbox").setLevel(logging.ERROR)  # commit() warns about every playlist missing in masterPlaylists6.xml
    database.commit()
    PENDING.unlink(missing_ok=True)
    print(f"master.db: bpm updated on {updated_count} track(s), {snapped_count} cue value(s) snapped")
    if flagged_ids:
        try:
            added_count = add_review_playlist(database, flagged_ids)
            database.commit()
            PENDING_REVIEW.unlink(missing_ok=True)
            print(f"review playlist: {added_count} track(s) added ({len(flagged_ids) - added_count} already in it)")
        except Exception as e:
            database.session.rollback()
            PENDING_REVIEW.write_text(json.dumps(flagged_ids), encoding="utf-8")
            print(f"review playlist FAILED ({e!r}); the track ids are kept in {PENDING_REVIEW.name} and retried next run")


def main(argv=None, with_cues=False, description=None):
    """with_cues=True is used by predict_beatgrids_and_cues.py: the cues of every song are predicted and written in the same run and the same table."""
    argument_parser = argparse.ArgumentParser(description=description or __doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--since-hours", type=float, default=8)
    argument_parser.add_argument("--from", dest="date_from")
    argument_parser.add_argument("--to", dest="date_to")
    argument_parser.add_argument("--playlist", action="append")
    argument_parser.add_argument("--names")
    argument_parser.add_argument("--force", action="store_true", help="reanalyse tracks already done")
    argument_parser.add_argument("--dry-run", action="store_true", help="analyse and report, write nothing")
    argument_parser.add_argument("--no-cue-snapping", action="store_true", help="do not snap cues to the new grid")
    argument_parser.add_argument("--only-snap-cues", action="store_true", help="only snap the cues of the selected tracks to their current grid (no analysis)")
    argument_parser.add_argument("--cue-snap-margin-ms", type=int, help="cue snap margin in ms (default CUE_SNAP_MARGIN_MS from .env, else 10)")
    argument_parser.add_argument("--defer-database-write", action="store_true", help="do not write master.db now, keep the bpm updates for the next run")
    adaptive_workers.add_argument(argument_parser)
    argument_parser.add_argument("--plain", action="store_true", help="plain log lines instead of the live table")
    argument_parser.add_argument("--wait-minutes", type=float, default=30, help="max minutes to wait per track for Rekordbox's own analysis files")
    argument_parser.add_argument("--database-path")
    argument_parser.add_argument("--encryption-key")
    argument_parser.add_argument("--share-directory")
    argument_parser.add_argument("--force-advanced", action="store_true", help="run the advanced timing model (bar-folded raw audio, advanced.py) on EVERY track: slowest, most accurate. Default: only on tracks the basic pass is unsure about")
    argument_parser.add_argument("--basic-only", action="store_true", help="never run the advanced timing model (fastest)")
    if with_cues:
        argument_parser.add_argument("--no-cues", action="store_true", help="grid only, no cues")
        argument_parser.add_argument("--minimum-confidence", type=float, default=0.0, help="no cues for tracks whose D or e1 probability is below this")
        argument_parser.add_argument("--skip-likely-wrong", action="store_true", help="no cues for tracks whose grid was flagged 'Likely wrong' (default: cues for everything)")
    arguments = argument_parser.parse_args(argv)
    controller = adaptive_workers.WorkerController(arguments.force_number_of_jobs)   # starts measuring the load right away
    if arguments.force_advanced:
        os.environ["BEATGRID_ANALYSIS_MODE"] = "advanced"
    elif arguments.basic_only:
        os.environ["BEATGRID_ANALYSIS_MODE"] = "basic"

    from lib.rekordbox import ground_truth
    from lib.rekordbox import database as rekordbox_database
    from lib.rekordbox.database import DjmdContent
    from lib.rekordbox import anlz_files as anlz_grid

    environment = ground_truth.load_env(paths.ENV_FILE)
    database_path, key = arguments.database_path or environment.get("REKORDBOX_DATABASE_PATH"), arguments.encryption_key or environment.get("REKORDBOX_ENCRYPTION_KEY", "")
    if not database_path:
        sys.exit("REKORDBOX_DATABASE_PATH missing (.env or --database-path)")
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    database = rekordbox_database.open_database(database_path, key)
    share = rekordbox_database.share_directory_of(database, SimpleNamespace(share_directory=arguments.share_directory, database_path=database_path))
    rows = select(database, arguments, ground_truth, DjmdContent)
    margin = arguments.cue_snap_margin_ms if arguments.cue_snap_margin_ms is not None else int(environment.get("CUE_SNAP_MARGIN_MS", "10"))
    if arguments.only_snap_cues:
        if arguments.dry_run:
            sys.exit("--only-snap-cues has no dry run")
        apply_db(database, database_path, rekordbox_database, DjmdContent, {}, share, rows, margin, False)
        return
    todo = [content_row for content_row in rows if arguments.force or str(content_row.ID) not in state]
    if not todo:
        print(f"{len(rows)} tracks selected, all already done (use --force to redo)")
        if PENDING.exists() and not arguments.dry_run:
            apply_db(database, database_path, rekordbox_database, DjmdContent, json.loads(PENDING.read_text()), share, [], margin, False)
        return
    from lib import progress_table
    CACHE.mkdir(exist_ok=True)
    cues_enabled = with_cues and not arguments.no_cues and not arguments.defer_database_write
    if with_cues and arguments.defer_database_write and not arguments.no_cues:
        print("note: --defer-database-write given, so no cues are written (they need the database write)", flush=True)
    cue_waiting, cue_running, cue_meta, plans = [], {}, {}, []
    todo_by_id = {str(content_row.ID): content_row for content_row in todo}
    if cues_enabled:
        import predict_cues as cue_module
        from lib.cues import cue_export, cue_predictor
        cue_state = json.loads(cue_module.STATE.read_text(encoding="utf-8")) if cue_module.STATE.exists() else {}
        playlists_by_content_id = cue_export.playlist_map(database, ground_truth, rekordbox_database)
        existing_cue_ids = {}
        for cue_row in database.query(rekordbox_database.DjmdCue).filter(rekordbox_database.DjmdCue.rb_local_deleted == 0).all():
            existing_cue_ids.setdefault(str(cue_row.ContentID), []).append(str(cue_row.ID))
    database_prepared = False
    cue_templates = {}
    mode = {"advanced": "force-advanced (advanced model on every track)", "basic": "basic only"}.get(os.environ.get("BEATGRID_ANALYSIS_MODE", "auto"), "auto (advanced model only where the basic pass is unsure)")
    header = [f" Analyzing: {describe(arguments)}" + (" (beat grid + cues)" if cues_enabled else "") + ("   [DRY RUN, nothing is written]" if arguments.dry_run else ""),
              lambda: f" {len(todo)} song(s) to analyze ({len(rows) - len(todo)} already done, skipped)   |   {controller.description()}   |   mode: {mode}"]
    start_time = time.time()
    rows_by_track_id = {}
    for i, content_row in enumerate(todo, 1):
        audio_file_exists = Path(content_row.FolderPath or "").is_file()
        rows_by_track_id[str(content_row.ID)] = dict(n=i, file=Path(str(content_row.FolderPath or content_row.Title)).name, status="waiting" if audio_file_exists else "skipped", t_start=0, secs=0,
                                flag="", note="" if audio_file_exists else "audio file missing", bpm=None, nsec=None, **({"cues": ""} if cues_enabled else {}))
    table_rows = list(rows_by_track_id.values())
    live = None if (arguments.plain or not sys.stdout.isatty() or not sys.stdin.isatty()) else progress_table.Live(table_rows, header, controller.live_workers, start_time)
    if live is None:
        print("\n".join(line() if callable(line) else line for line in header), flush=True)
    pending = {str(content_row.ID): content_row for content_row in todo if rows_by_track_id[str(content_row.ID)]["status"] == "waiting"}
    seen, running, last_refresh, report = {}, {}, {}, []
    bpm_updates = dict(json.loads(PENDING.read_text())) if PENDING.exists() else {}
    review_ids = {}
    last_pass = 0.0

    def fail(track_id, status, note):
        row = rows_by_track_id[track_id]
        row.update(status=status, note=note)
        if live is None:
            print(f"{status.upper()} {row['file']}: {note}", flush=True)
        report.append(status)

    def cue_check(track_id, content_row, result, new_data):
        """(reason why no cues, None) or (None, cue job). The grid is the one just built."""
        if len(result["secs"]) > 1:
            return "several tempo sections", None
        if result["review_level"] == "wrong" and arguments.skip_likely_wrong:
            return "grid flagged Likely wrong", None
        own_cue_ids = set(cue_state.get(track_id, {}).get("cue_ids", []))
        if set(existing_cue_ids.get(track_id, [])) - own_cue_ids:
            return "has cues of its own", None
        if track_id in cue_state and not arguments.force:
            return "cues already done", None
        beats = ground_truth.read_pqtz(new_data)
        times, first_downbeat_index, multi = cue_predictor.grid_info(beats) if len(beats) >= 4 else (None, None, False)
        if first_downbeat_index is None or multi:
            return "no constant grid", None
        cue_meta[track_id] = (times, first_downbeat_index, own_cue_ids, f"{ground_truth.artist_of(content_row)} - {content_row.Title}")
        return None, dict(id=track_id, src=str(content_row.FolderPath), beats=beats, pl=sorted(playlists_by_content_id.get(content_row.ID, [])))

    def prepare_database():
        """once, before the first write of the run: backup of master.db, templates for new cues, commit allowed while Rekordbox runs"""
        nonlocal database_prepared
        from pyrekordbox.masterdb import models as masterdb_models
        if database_prepared:
            return masterdb_models
        cue_templates.update(cue_module.find_templates(database, masterdb_models)[0])
        cue_templates["any"] = next(iter(cue_templates.values()), None)
        BACKUP.mkdir(exist_ok=True)
        shutil.copy2(database_path, BACKUP / f"master_{datetime.now():%Y%m%d_%H%M%S}.db")
        rotate_backups()
        if rekordbox_database.rekordbox_running():
            import pyrekordbox.masterdb.database as masterdb_database
            masterdb_database.get_rekordbox_pid = lambda: 0
        import logging
        logging.getLogger("pyrekordbox").setLevel(logging.ERROR)
        database_prepared = True
        return masterdb_models

    def finish_song(track_id, plan, cue_note=""):
        """The song is done when its grid is written, its cues are written and master.db is committed (bpm, cue snapping, cues, review playlist: one commit per song)."""
        row, content_row = rows_by_track_id[track_id], todo_by_id[track_id]
        row["cues"] = len(plan["cues"]) if plan else "-"
        if cue_note:
            row["note"] = "; ".join(part for part in (row.get("note", ""), cue_note) if part)
        if not arguments.dry_run:
            bpm = bpm_updates.pop(track_id, None)
            try:
                masterdb_models = prepare_database()    # no-op after the first call
                if bpm is not None:
                    content_row.BPM = bpm
                own_cue_ids = cue_meta[track_id][2] if track_id in cue_meta else set()
                if plan:
                    if cue_templates.get("any") is None:
                        raise RuntimeError("no existing cue in the library to clone from; create one hot cue and one memory cue manually in Rekordbox once")
                    for kind in {cue["kind"] for cue in plan["cues"]}:
                        cue_templates.setdefault(kind, cue_templates["any"])
                    cue_module.delete_own(database, masterdb_models, track_id, own_cue_ids)
                if not arguments.no_cue_snapping:
                    snap_rows(database, rekordbox_database, share, [content_row], margin)
                new_rows = cue_module.write_track(database, masterdb_models, rekordbox_database, content_row, plan["cues"], cue_templates, datetime.now(timezone.utc)) if plan else []
                if track_id in review_ids:
                    add_review_playlist(database, {track_id: review_ids[track_id]})
                database.commit()
                if plan:
                    cue_state[track_id] = dict(name=plan["name"], done=datetime.now().isoformat(timespec="seconds"), cue_ids=[str(new_row.ID) for new_row in new_rows], n_cues=len(new_rows),
                                               D=plan["pred"]["D"], e1=plan["pred"]["e1"], H=plan["pred"]["H"], e2=plan["pred"]["e2"], conf_D=plan["pred"]["conf_D"], conf_e1=plan["pred"]["conf_e1"])
                    cue_module.STATE.write_text(json.dumps(cue_state, indent=1, ensure_ascii=False), encoding="utf-8")
                PENDING.write_text(json.dumps(bpm_updates), encoding="utf-8")
            except Exception as e:
                database.session.rollback()
                if bpm is not None:
                    bpm_updates[track_id] = bpm
                row.update(cues="FAIL", note="; ".join(part for part in (row.get("note", ""), f"database write failed: {e!r}"[:120]) if part))
        row.update(status="done", secs=time.time() - row["t_start"])
        if live is None:
            print(f"{'dry ' if arguments.dry_run else ''}done {row['file']}: {row['bpm'] or 0:.3f} bpm, {row['nsec']} section(s), cues {row['cues']}, {row['secs']:.0f}s" + (f"  {row['note']}" if row["note"] else ""), flush=True)

    interrupted = False
    try:
        if live:
            live.start()
        with ProcessPoolExecutor(max_workers=controller.maximum_workers) as executor:
            while pending or running or cue_waiting or cue_running:
                now = time.time()
                controller.update(len(running) + len(cue_running), waiting=sum(1 for track_id in pending if rows_by_track_id[track_id]["status"] == "waiting"))   # "waiting" = Rekordbox has not finished its own analysis yet
                allowed_workers = controller.allowed_workers()
                while cue_waiting and len(running) + len(cue_running) < allowed_workers:     # cues of finished grids first: songs get finished instead of piling up
                    cue_job = cue_waiting.pop(0)
                    cue_running[executor.submit(cue_module.pjob, cue_job)] = cue_job["id"]
                if now - last_pass >= 1.0:
                    last_pass = now
                    for track_id, content_row in list(pending.items()):
                        if len(running) + len(cue_running) >= allowed_workers:
                            if rows_by_track_id[track_id]["status"] == "waiting" and seen.get(track_id, (None,))[0] is not None:
                                rows_by_track_id[track_id]["status"] = "queued"
                            continue
                        if now - last_refresh.get(track_id, 0) > 8:
                            last_refresh[track_id] = now
                            try:
                                database.session.refresh(content_row)
                            except Exception:
                                pass
                        files = rekordbox_database.anlz_files(share, content_row) if content_row.AnalysisDataPath else {}
                        file_stability = stable(files) if files else None
                        if file_stability and seen.get(track_id, (None,))[0] == file_stability and now - seen[track_id][1] >= 3:
                            job_arguments = dict(id=track_id, src=str(content_row.FolderPath), ext=str(files["EXT"]), cache=str(CACHE), rb_bpm=(content_row.BPM or 0) / 100.0)
                            running[executor.submit(job, job_arguments)] = (track_id, content_row, files)
                            rows_by_track_id[track_id].update(status="running", t_start=now)
                            del pending[track_id]
                        else:
                            if not (seen.get(track_id) and seen[track_id][0] == file_stability):
                                seen[track_id] = (file_stability, now)
                            if now - start_time > arguments.wait_minutes * 60:
                                fail(track_id, "timeout", "Rekordbox analysis files never appeared")
                                del pending[track_id]
                for finished_future in [finished_future for finished_future in running if finished_future.done()]:
                    track_id, content_row, files = running.pop(finished_future)
                    row = rows_by_track_id[track_id]
                    row["secs"] = time.time() - row["t_start"]
                    _, result = finished_future.result()
                    if result.get("error") or not result.get("built"):
                        fail(track_id, "failed", result.get("error", "no grid found"))
                        continue
                    built = result["built"]
                    analysis_file = files["DAT"]
                    data = analysis_file.read_bytes()
                    if not anlz_grid.roundtrip_ok(data, built):
                        fail(track_id, "failed", "ANLZ rebuild check failed")
                        continue
                    new_data = anlz_grid.set_pqtz(data, built)
                    if not arguments.dry_run:
                        BACKUP.mkdir(exist_ok=True)
                        backup_path = BACKUP / f"{track_id}_ANLZ0000.DAT"
                        if not backup_path.exists():
                            shutil.copy2(analysis_file, backup_path)
                        bpm_updates[track_id] = int(round(result["secs"][0]["bpm"] * 100))
                        analysis_file.write_bytes(new_data)
                        state[track_id] = dict(name=row["file"], done=datetime.now().isoformat(timespec="seconds"), bpm=result["secs"][0]["bpm"],
                                          sections=len(result["secs"]), level=result["level"], review=result["review"], review_level=result["review_level"], review_why=result["review_why"])
                        if result["review"]:
                            review_ids[track_id] = result["review_level"]
                        STATE.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
                    review_level_name = {"wrong": "Likely wrong", "uncertain": "Uncertain"}.get(result["review_level"], result["review_level"])
                    row.update(status="running" if cues_enabled else "done", bpm=result["secs"][0]["bpm"], nsec=len(result["secs"]),
                               flag="REVIEW" if result["review"] else "", note=(f"{review_level_name}: " + "; ".join(result["review_why"])) if result["review"] else "")
                    report.append("review" if result["review"] else "ok")
                    if cues_enabled:
                        reason, cue_job = cue_check(track_id, todo_by_id[track_id], result, new_data)
                        if cue_job:
                            row["stage"] = "cues"
                            cue_waiting.append(cue_job)
                        else:
                            finish_song(track_id, None, f"cues skipped: {reason}")
                    elif live is None:
                        print(f"{'dry ' if arguments.dry_run else ''}done {row['file']}: {row['bpm']:.3f} bpm, {row['nsec']} section(s), {row['secs']:.0f}s" + (f"  REVIEW: {row['note']}" if row["flag"] else ""), flush=True)
                for finished_cue_future in [finished_cue_future for finished_cue_future in cue_running if finished_cue_future.done()]:
                    track_id = cue_running.pop(finished_cue_future)
                    _, prediction = finished_cue_future.result()
                    times, first_downbeat_index, _, name = cue_meta[track_id]
                    if "error" in prediction:
                        finish_song(track_id, None, f"cues FAILED: {prediction['error'][:60]}")
                        rows_by_track_id[track_id]["cues"] = "FAIL"
                    elif prediction["conf_D"] < arguments.minimum_confidence or prediction["conf_e1"] < arguments.minimum_confidence:
                        finish_song(track_id, None, "cues skipped: low confidence")
                    else:
                        plan = dict(id=track_id, name=name, pred=prediction, cues=cue_module.plan_cues(prediction, times, first_downbeat_index))
                        plans.append(plan)
                        finish_song(track_id, plan)
                if live:
                    live.render()
                time.sleep(0.1 if live else 0.5)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        controller.stop()
        if live:
            live.stop()
    if interrupted:
        PENDING.write_text(json.dumps(bpm_updates), encoding="utf-8")
        print("\nStopped by user. Finished tracks keep their new grids; their bpm updates are applied by the next run (pending_bpm.json).")
    if live:
        print("\n".join(line() if callable(line) else line for line in header))
        print(progress_table.final_table(table_rows))
    if plans:
        cue_module.OUT.mkdir(exist_ok=True)
        with open(cue_module.OUT / "phrase_cues.jsonl", "w", encoding="utf-8") as plan_file:
            for plan in plans:
                plan_file.write(json.dumps(plan, ensure_ascii=False) + "\n")
    if bpm_updates and not arguments.dry_run and not interrupted:
        apply_db(database, database_path, rekordbox_database, DjmdContent, bpm_updates, share, [] if arguments.no_cue_snapping else [content_row for content_row in todo if str(content_row.ID) in bpm_updates], margin, arguments.defer_database_write, review_ids)
    done = [table_row for table_row in table_rows if table_row["status"] == "done"]
    review_count, failed_count = sum(1 for table_row in table_rows if table_row["flag"]), sum(1 for table_row in table_rows if table_row["status"] in ("failed", "timeout"))
    elapsed_seconds = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"{'Interrupted' if interrupted else 'All scheduled songs finished'}: {len(done)}/{len(table_rows)} analyzed, {failed_count} failed, {len(table_rows) - len(done) - failed_count - sum(1 for table_row in table_rows if table_row['status'] == 'skipped')} not run, "
          f"{sum(1 for table_row in table_rows if table_row['status'] == 'skipped')} skipped, {review_count} flagged for review.")
    if done:
        print(f"Average analysis time: {sum(table_row['secs'] for table_row in done) / len(done):.1f} s per song ({controller.description()}), total wall time {elapsed_seconds // 60:.0f}:{elapsed_seconds % 60:02.0f} min.")
    if review_count and not arguments.dry_run:
        from lib.beatgrid import review_flags
        print(f"Flagged songs were added to the playlists: {', '.join(review_flags.PLAYLISTS.values())}. Remove a song from the playlist once you checked it.")
    if done and not arguments.dry_run:
        print("IMPORTANT: restart Rekordbox for the new BPM display to take effect (grids show after reloading a track).")
    if arguments.dry_run:
        print("Dry run: nothing was written.")


if __name__ == "__main__":
    main()
