"""Detect the grid of one track and write it into its ANLZ .DAT (PQTZ). Dry run unless --write; --write backs up the .DAT first.
usage: python grid_writer.py AUDIO ANLZ0000.DAT [--extended-analysis-path ANLZ0000.EXT] [--write] [--cache DIR]
   or: python grid_writer.py FOLDER [--write]   (folder from export_ground_truth_examples: audio + ANLZ0000.DAT/.EXT inside; --write only touches that copy)"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

if __name__ == "__main__":
    import _bootstrap  # only when this file is run directly; makes `import lib...` work

from lib.rekordbox import anlz_files
from lib.beatgrid import grid_pipeline
from lib.beatgrid import track_job


def compute_grid(audio, extended_analysis_path, cache, track_id, rekordbox_bpm=None):
    """Detect and build the grid for one track. Returns dict(built, secs, level, why, model_ms, status, dur)."""
    Path(cache).mkdir(parents=True, exist_ok=True)
    container_header = grid_pipeline.probe_header(audio)
    try:
        waveform_offset = track_job.measure_timeline_offset(cache, track_id, audio, extended_analysis_path)
    except Exception:
        waveform_offset = None
    model_ms, status = grid_pipeline.container_offset_ms(container_header, waveform_offset)
    events, sections = track_job.detect(cache, audio, track_id, rekordbox_bpm)
    built = track_job.build_grid_advanced(events, sections, model_ms, audio)
    assessment_level, assessment_notes = grid_pipeline.assess(events, sections, events["dur"]) if sections else ("check", ["no grid"])
    from lib.beatgrid import review_flags
    review_flag_level, review_flag_reasons = review_flags.review_level(sections, assessment_level, assessment_notes, events["dur"])
    needs_review = review_flag_level is not None
    return dict(built=built, secs=sections, level=assessment_level, why=assessment_notes, model_ms=model_ms, status=status, dur=events["dur"], review=needs_review, review_level=review_flag_level, review_why=review_flag_reasons)


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("audio")
    argument_parser.add_argument("anlz_dat_path", nargs="?")
    argument_parser.add_argument("--extended-analysis-path")
    argument_parser.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "beatgrid_cache"))
    argument_parser.add_argument("--write", action="store_true")
    argument_parser.add_argument("--force-advanced", action="store_true", help="run the advanced timing model (bar-folded raw audio, advanced.py) on EVERY track: slowest, most accurate. Default: only on tracks the basic pass is unsure about")
    argument_parser.add_argument("--basic-only", action="store_true", help="never run the advanced timing model (fastest)")
    arguments = argument_parser.parse_args()
    if arguments.force_advanced:
        os.environ["BEATGRID_ANALYSIS_MODE"] = "advanced"
    elif arguments.basic_only:
        os.environ["BEATGRID_ANALYSIS_MODE"] = "basic"
    if Path(arguments.audio).is_dir():
        folder = Path(arguments.audio)
        arguments.audio = str(next(candidate for candidate in sorted(folder.iterdir()) if candidate.suffix.lower() in (".mp3", ".m4a", ".wav")))
        arguments.anlz_dat_path = str(folder / "ANLZ0000.DAT")
        arguments.extended_analysis_path = str(folder / "ANLZ0000.EXT") if (folder / "ANLZ0000.EXT").exists() else None
    if not arguments.anlz_dat_path:
        sys.exit("need the .DAT path or a folder")
    Path(arguments.cache).mkdir(parents=True, exist_ok=True)
    track_id = Path(arguments.audio).stem[:40]
    container_header = grid_pipeline.probe_header(arguments.audio)
    try:
        waveform_offset = track_job.measure_timeline_offset(arguments.cache, track_id, arguments.audio, arguments.extended_analysis_path)
    except Exception as error:
        print("waveform offset failed:", repr(error))
        waveform_offset = None
    model_ms, status = grid_pipeline.container_offset_ms(container_header, waveform_offset)
    events, sections = track_job.detect(arguments.cache, arguments.audio, track_id)
    built = grid_pipeline.build_grid(sections, model_ms, events["dur"], events)
    if not built:
        sys.exit("no grid found, nothing written")
    assessment_level, assessment_notes = grid_pipeline.assess(events, sections, events["dur"])
    anlz_bytes = Path(arguments.anlz_dat_path).read_bytes()
    stored_beats = anlz_files.read_pqtz(anlz_bytes)
    print(f"offset model {model_ms:+.1f} ms ({status}); review level {assessment_level} {assessment_notes}")
    for section in sections:
        print(f"  section {section['start_s']:.1f}-{section['end_s']:.1f} s  {section['bpm']:.3f} bpm")
    print(f"built {len(built)} beats (first {built[0]}), stored {len(stored_beats)} beats (first {stored_beats[0] if stored_beats else None})")
    print("compare:", {key: (round(value, 2) if isinstance(value, float) else value) for key, value in grid_pipeline.dry_compare(built, stored_beats).items()})
    if not anlz_files.roundtrip_ok(anlz_bytes, built):
        sys.exit("ANLZ rebuild check failed, nothing written")
    if not arguments.write:
        print("dry run, nothing written (use --write)")
        return
    backup_path = Path(arguments.anlz_dat_path).with_suffix(".DAT.bak")
    if not backup_path.exists():
        shutil.copy2(arguments.anlz_dat_path, backup_path)
    Path(arguments.anlz_dat_path).write_bytes(anlz_files.set_pqtz(anlz_bytes, built))
    print("written; backup:", backup_path)


if __name__ == "__main__":
    main()
