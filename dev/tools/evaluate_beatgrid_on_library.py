#!/usr/bin/env python3
"""
evaluate_beatgrid_on_library.py - run the beat grid detector on tracks from the live Rekordbox library and compare the result
with Rekordbox's own grid and with the grid implied by the track's cues. No audio is copied anywhere;
the output is a small report (eval_out/eval_report.txt and eval_out/eval_report.json).

Per track: file header facts (container, sample rate, start_time, encoder, iTunSMPB), detected sections
(bpm, anchor), offset RB-minus-detector mod beat and mod half beat, residual spread, and whether Rekordbox's
grid agrees with the cue grid. Needs numpy, scipy, librosa, ffmpeg/ffprobe on PATH.
Settings come from .env (REKORDBOX_DATABASE_PATH, REKORDBOX_ENCRYPTION_KEY). Rekordbox should be closed.

  python evaluate_beatgrid_on_library.py --playlist Dubstep --playlist DnB --per-playlist 8
  python evaluate_beatgrid_on_library.py --all-tracks --track-limit 40
  python evaluate_beatgrid_on_library.py --track-ids 123,456
  python evaluate_beatgrid_on_library.py --playlist Dubstep --per-playlist 50 --batch-offset 1   (next 50 after the first batch; use the same --seed)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import _bootstrap
from lib import paths
for argument_index, argument_text in enumerate(sys.argv):
    for flag_name, environment_variable in (("--onset-fraction", "BEATGRID_ONSET_FRACTION"), ("--refine-fraction", "BEATGRID_REFINE_FRACTION")):
        if argument_text.startswith(flag_name):
            os.environ[environment_variable] = argument_text.split("=", 1)[1] if "=" in argument_text else sys.argv[argument_index + 1]
from lib.beatgrid import grid_pipeline
from lib.beatgrid import track_job

OUT = paths.DEBUG / "eval_out"


def pick_rows(database, arguments, ground_truth, DjmdContent):
    rows = []
    if arguments.track_ids:
        wanted_ids = set(arguments.track_ids.split(","))
        rows = [content_row for content_row in database.query(DjmdContent).all() if str(content_row.ID) in wanted_ids and not content_row.rb_local_deleted]
    elif arguments.all_tracks:
        rows = [content_row for content_row in database.query(DjmdContent).all() if not content_row.rb_local_deleted]
        random.Random(arguments.seed).shuffle(rows)
        if arguments.track_limit:
            rows = rows[arguments.batch_offset * arguments.track_limit:(arguments.batch_offset + 1) * arguments.track_limit]
    else:
        for name in arguments.playlist or []:
            found, near = ground_truth.playlist_rows(database, name)
            if not found:
                print(f"playlist '{name}': none found" + (f" (similar: {near})" if near else ""))
            random.Random(arguments.seed).shuffle(found)
            batch_start = arguments.batch_offset * arguments.per_playlist
            chunk = found[batch_start:batch_start + arguments.per_playlist]
            print(f"playlist '{name}': {len(found)} tracks, offset {arguments.batch_offset} -> {len(chunk)} selected")
            rows += chunk
    seen, unique_rows = set(), []
    for content_row in rows:
        if content_row.ID not in seen:
            seen.add(content_row.ID)
            unique_rows.append(content_row)
    return unique_rows


def report_lines(name, suffix, track_report, format_one_decimal):
    lines = [f"{name[:60]} | {suffix} sr {track_report['sample_rate']} | RB bpm {track_report['rekordbox_bpm']} | "
             f"hdr start {track_report['header'].get('start_time')} smpb {(track_report['header'].get('itunsmpb') or '')[:30]} enc {str(track_report['header'].get('encoder'))[:40]}"]
    ground_truth_grid, comparison = track_report["ground_truth"], track_report["rb_vs_ground_truth"]
    lines.append("    cues vs RB grid: " + (f"cue grid {ground_truth_grid['bpm']:.3f} bpm, RB off median {format_one_decimal(comparison['median'])} worst {format_one_decimal(comparison['worst'])} ms"
                                           if ground_truth_grid and comparison else "no cue grid"))
    for section in track_report["sections"]:
        parity = section.get("parity") or {}
        parity_text = f"parity on/off {parity.get('on_kick', 0) + parity.get('on_snare', 0):.0f}/{parity.get('off_kick', 0) + parity.get('off_snare', 0):.0f}" + (" FLIPPED" if parity.get("flipped") else "") + (f" [{parity['method']}]" if parity.get("method") == "near-zero tie-break" else "")
        if "offset_mod_beat_ms" in section:
            flag = "  <-- HALF-BEAT" if abs(section["offset_mod_beat_ms"] - section["offset_mod_half_beat_ms"]) > 20 else ""
            lines.append(f"    sec {section['start_s']:6.1f}-{section['end_s']:6.1f} bpm {section['bpm']:9.4f} | RB-mine mod beat {section['offset_mod_beat_ms']:+7.1f} "
                         f"(model {track_report['offset_model']['ms']:+.1f} [{track_report['offset_model']['status']}], err {section['err_vs_model_ms']:+.1f}) | spread "
                         f"{section['spread_p05']:+.1f}..{section['spread_p95']:+.1f} worst {section['spread_worst']:.1f} rms {section['spread_rms']:.1f} | {parity_text} | refine {(section.get('refine') or {}).get('shift_ms', 0):+.1f}" + (f" PHASE {section['phase_shift_ms']:+.1f}" if section.get('phase_shift_ms') else "") + (f" OVERRIDE {section['snare_override_ms']:+.1f}" if section.get('snare_override_ms') else "") + flag)
        else:
            lines.append(f"    sec {section['start_s']:6.1f}-{section['end_s']:6.1f} bpm {section['bpm']:9.4f} | only {section['n_rb_beats']} RB beats inside | {parity_text}")
    dry_run_result = track_report.get("dry_run")
    if dry_run_result and dry_run_result.get("ok"):
        lines.append(f"    dry run: built {dry_run_result['n_built']} / stored {dry_run_result['n_stored']} beats, diff median {dry_run_result['diff_median_ms']:+.1f} p95 {dry_run_result['diff_p95_abs_ms']:.1f} max {dry_run_result['diff_max_abs_ms']:.1f} ms, "
                     f"beat-in-bar shift {dry_run_result['numbering_shift']} (match {dry_run_result['numbering_match_frac']:.2f}), first beat {dry_run_result['first_built_ms']} vs {dry_run_result['first_stored_ms']} ms"
                     + ("  <-- NUMBERING" if dry_run_result['numbering_shift'] else "")
                     + ("" if dry_run_result.get('anlz_rebuild_ok', True) else "  <-- ANLZ REBUILD FAILED " + str(dry_run_result.get('anlz_error', ''))))
    elif dry_run_result:
        lines.append("    dry run: " + str(dry_run_result.get("reason")))
    assessment = track_report.get("assess")
    if assessment and assessment["level"] != "ok":
        lines.append(f"    review [{assessment['level']}]: " + "; ".join(assessment["reasons"]))
    if track_report.get("error"):
        lines.append("    ERROR " + track_report["error"])
    return lines


def write_reports(metas, results, format_one_decimal):
    report_rows, lines = [], []
    for job_index in sorted(results):
        name, suffix, audio_name, track_report = metas[job_index]
        track_report = dict(track_report, **results[job_index], audio_name=audio_name)
        report_rows.append(track_report)
        lines += report_lines(name, suffix, track_report, format_one_decimal)
    (OUT / "eval_report.json").write_text(json.dumps(report_rows, indent=1, ensure_ascii=False), encoding="utf-8")
    head = f"# bundle {grid_pipeline.VERSION} | onset_fraction {grid_pipeline.ONSET_FRACTION} refine_fraction {grid_pipeline.REFINE_FRACTION} | args: {' '.join(sys.argv[1:])}\n"
    (OUT / "eval_report.txt").write_text(head + "\n".join(lines) + "\n", encoding="utf-8")


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--database-path")
    argument_parser.add_argument("--encryption-key")
    argument_parser.add_argument("--share-directory")
    argument_parser.add_argument("--playlist", action="append")
    argument_parser.add_argument("--per-playlist", type=int, default=8)
    argument_parser.add_argument("--all-tracks", action="store_true")
    argument_parser.add_argument("--track-limit", type=int, default=0)
    argument_parser.add_argument("--batch-offset", type=int, default=0, help="batch index: --batch-offset 1 takes the next --per-playlist (or --track-limit) tracks after the first batch, same seed = same shuffle")
    argument_parser.add_argument("--track-ids", default="")
    argument_parser.add_argument("--parallel-workers", type=int, default=8, help="worker processes (1 = sequential)")
    argument_parser.add_argument("--band-envelopes", action="store_true", help="also cache 100 Hz band envelopes (eval_out/cache/<id>.band_envelopes.npz) for downbeat research")
    argument_parser.add_argument("--beat-features", action="store_true", help="also store raw-audio research features (eval_out/beat_features/<id>.npz, see lib/audio_features.py)")
    argument_parser.add_argument("--onset-sweep", default="", help="comma separated onset threshold fractions (default detector uses 0.25); prints residual modes per track instead of the normal analysis")
    argument_parser.add_argument("--onset-fraction", type=float, default=0.25, help="onset threshold fraction used by the detectors for the whole analysis (own cache files when != 0.25)")
    argument_parser.add_argument("--refine-fraction", type=float, default=0.1, help="onset fraction of the snare onsets used only for the final anchor refinement (0 = use the normal events; own cache files when != 0.1)")
    argument_parser.add_argument("--seed", type=int, default=1)
    argument_parser.add_argument("--bpm-snap-tolerance", type=float, default=0.1)
    argument_parser.add_argument("--force-advanced", action="store_true", help="run the advanced timing model (bar-folded raw audio, advanced_fine_adjust.py) on EVERY track: slowest, most accurate. Default: only on tracks the basic pass is unsure about")
    argument_parser.add_argument("--basic-only", action="store_true", help="never run the advanced timing model (fastest)")
    arguments = argument_parser.parse_args()
    if arguments.force_advanced:
        os.environ["BEATGRID_ANALYSIS_MODE"] = "advanced"
    elif arguments.basic_only:
        os.environ["BEATGRID_ANALYSIS_MODE"] = "basic"

    from lib.rekordbox import ground_truth as ground_truth
    from lib.rekordbox import database as rekordbox_database
    from lib.rekordbox.database import DjmdContent

    environment = ground_truth.load_env(paths.ENV_FILE)
    database_path, key = arguments.database_path or environment.get("REKORDBOX_DATABASE_PATH"), arguments.encryption_key or environment.get("REKORDBOX_ENCRYPTION_KEY", "")
    if not database_path:
        sys.exit("REKORDBOX_DATABASE_PATH missing (.env or --database-path)")
    if not (arguments.track_ids or arguments.all_tracks or arguments.playlist):
        sys.exit("give --playlist NAME, --all-tracks or --track-ids")

    database = rekordbox_database.open_database(database_path, key)
    share = rekordbox_database.share_directory_of(database, SimpleNamespace(share_directory=arguments.share_directory, database_path=database_path))
    rows = pick_rows(database, arguments, ground_truth, DjmdContent)
    print(f"{len(rows)} tracks selected", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "cache").mkdir(exist_ok=True)

    metas, jobs = {}, []
    for job_index, content_row in enumerate(rows):
        info = ground_truth.collect(database, share, content_row)
        name = f"{ground_truth.artist_of(content_row)} - {content_row.Title}"
        if info is None:
            print(f"skipped (no file or no grid): {name}")
            continue
        track_report = ground_truth.describe(info, arguments.bpm_snap_tolerance)
        track_report.pop("cue_vs_rb_grid", None)
        track_report["rb_grid"].pop("beats", None)
        beats = [tuple(beat) for beat in ground_truth.read_pqtz(info["files"]["DAT"].read_bytes())]
        extended_file = info["files"].get("EXT")
        metas[job_index] = (name, info["src"].suffix, info["src"].name, track_report)
        jobs.append((job_index, dict(id=content_row.ID, src=str(info["src"]), ext=str(extended_file) if extended_file else None, beats=beats, cache=str(OUT / "cache"), band_envelopes=arguments.band_envelopes, beat_features=arguments.beat_features, dat=str(info["files"]["DAT"]), rb_bpm=track_report.get("rekordbox_bpm"))))
    database.close()

    if arguments.onset_sweep:
        fractions = [float(fraction_text) for fraction_text in arguments.onset_sweep.split(",")]
        for job_index, job_arguments in jobs:
            job_arguments["fracs"] = fractions
        lines = ["residual mode of detector vs Rekordbox grid in ms (+ = detector later), per onset fraction: snare/kick (n)"]
        with ProcessPoolExecutor(max_workers=max(1, min(arguments.parallel_workers, len(jobs)))) as executor:
            futures = {executor.submit(track_job.sweep_track, job_arguments): job_index for job_index, job_arguments in jobs}
            sweep_results = {}
            for i, future in enumerate(as_completed(futures), 1):
                sweep_results[futures[future]] = future.result()
                print(f"[{i}/{len(jobs)}] {metas[futures[future]][0][:60]}", flush=True)
        for job_index in sorted(sweep_results):
            sweep_result = sweep_results[job_index]
            if sweep_result.get("error"):
                lines.append(f"{metas[job_index][0][:40]}: ERROR {sweep_result['error']}")
                continue
            cells = [f"{fraction}: {values_by_kind['snare'][0]:+.1f}/{values_by_kind['kick'][0]:+.1f} ({values_by_kind['snare'][1]}/{values_by_kind['kick'][1]})" for fraction, values_by_kind in sweep_result["fracs"].items()]
            lines.append(f"{metas[job_index][0][:40]:40s} " + " | ".join(cells))
        (OUT / "onset_sweep.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return

    results = {}
    total = len(jobs)
    jobs.sort(key=lambda indexed_job: -os.path.getsize(indexed_job[1]["src"]))
    start_time = time.time()
    if arguments.parallel_workers > 1 and total > 1:
        print(f"analysing {total} tracks on {min(arguments.parallel_workers, total)} processes", flush=True)
        with ProcessPoolExecutor(max_workers=min(arguments.parallel_workers, total)) as executor:
            futures = {executor.submit(track_job.run_track, job_arguments): job_index for job_index, job_arguments in jobs}
            for i, future in enumerate(as_completed(futures), 1):
                job_index = futures[future]
                try:
                    results[job_index] = future.result()
                except Exception as e:
                    results[job_index] = dict(error=repr(e), sections=[])
                print(f"[{i}/{total}] {metas[job_index][0][:60]} ({results[job_index].get('seconds', 0):.0f}s)", flush=True)
                write_reports(metas, results, ground_truth.format_one_decimal)
    else:
        for i, (job_index, job_arguments) in enumerate(jobs, 1):
            results[job_index] = track_job.run_track(job_arguments)
            print(f"[{i}/{total}] {metas[job_index][0][:60]} ({results[job_index].get('seconds', 0):.0f}s)", flush=True)
            write_reports(metas, results, ground_truth.format_one_decimal)

    write_reports(metas, results, ground_truth.format_one_decimal)
    print(f"\nreport: {OUT / 'eval_report.txt'}  and  {OUT / 'eval_report.json'}   (total {time.time() - start_time:.0f}s)")


if __name__ == "__main__":
    main()
