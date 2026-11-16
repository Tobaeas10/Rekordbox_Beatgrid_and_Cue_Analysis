#!/usr/bin/env python3
"""
ground_truth.py (was fetch_examples.py; CLI now dev/tools/export_ground_truth_examples.py) - copy example tracks plus beat grid ground truth out of the Rekordbox database.

For the main example track and a random sample from named playlists it writes one folder per
track: the audio file, ground_truth.md (readable), ground_truth.json (all beats and cues), and
optionally the raw ANLZ files. Settings come from .env (REKORDBOX_DATABASE_PATH, REKORDBOX_ENCRYPTION_KEY). Rekordbox should be closed.

Ground truth logic: hot cues sit on the first beat of a bar, memory cues on beats. A constant
grid is fitted through them (bpm snapped to a whole number when within --bpm-snap-tolerance), then
Rekordbox's own PQTZ grid is compared against that fit.
"""
import argparse
import json
import math
import random
import re
import shutil
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from lib.paths import load_env  # noqa: F401  (re-exported)
from lib.rekordbox import anlz_files
from lib.rekordbox import database as rekordbox_database
from lib.rekordbox.database import (DjmdContent, DjmdCue, DjmdPlaylist, DjmdSongPlaylist, artist_of,  # noqa: F401
                                    find_by_path, find_by_text, norm_path, playlist_rows)



EXTRA_TRACKS = ["Tabarnak - Strawman", "Automhate - Smooth Operator"]  # tempo change tracks, always exported with raw ANLZ files
MAIN_TRACK = r"C:\Users\Tobae\Downloads\_rekordbox\Collection\Subfiltronik - Blockz (Shockz Remix).m4a"
OUT_DIR = r"C:\Users\Tobae\Downloads\llm files\example songs + ground truth"
PLAYLISTS = ["Dubstep", "DnB", "2-Step", "UKG (4x4 only)"]


def safe_name(name, limit=80):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name[:limit] or "track"


# ---------------------------------------------------------------------------------------
# ANLZ access
# ---------------------------------------------------------------------------------------
def read_pqtz(data):
    """List of (beat_in_bar, tempo_x100, time_ms) from the PQTZ tag of a DAT file."""
    for tag_type, offset, header_length, tag_length in anlz_files.iter_tags(data):
        if tag_type == "PQTZ":
            entry_count = struct.unpack_from(">I", data, offset + 20)[0]
            return [struct.unpack_from(">HHI", data, offset + 24 + 8 * i) for i in range(entry_count)]
    return []


def itunsmpb(path):
    try:
        probe_data = rekordbox_database.ffprobe(path)
    except Exception:
        return None
    tags = {tag_name.lower(): tag_value for tag_name, tag_value in (probe_data.get("format", {}).get("tags") or {}).items()}
    try:
        tags.update({tag_name.lower(): tag_value for tag_name, tag_value in (rekordbox_database.audio_stream(probe_data).get("tags") or {}).items()})
    except Exception:
        pass
    if "itunsmpb" in tags:
        try:
            return rekordbox_database.parse_smpb(tags["itunsmpb"])
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------------------
# ground truth maths (pure python)
# ---------------------------------------------------------------------------------------
def linfit(lattice_indices, times):
    count = len(lattice_indices)
    mean_index, mean_time = sum(lattice_indices) / count, sum(times) / count
    index_variance_sum = sum((index - mean_index) ** 2 for index in lattice_indices)
    if index_variance_sum == 0:
        return None
    slope = sum((index - mean_index) * (observed_time - mean_time) for index, observed_time in zip(lattice_indices, times)) / index_variance_sum
    return mean_time - slope * mean_index, slope


def fit_lattice(times, period_hint):
    """Fit t = t0 + k*P through cue times. Returns dict or None."""
    sorted_times = sorted(set(times))
    if len(sorted_times) < 2:
        return None
    lattice_indices = [round((cue_time - sorted_times[0]) / period_hint) for cue_time in sorted_times]
    if len(set(lattice_indices)) < 2:
        return None
    fit_result = linfit(lattice_indices, sorted_times)
    if fit_result is None:
        return None
    anchor_ms, period = fit_result
    residuals = [cue_time - (anchor_ms + lattice_index * period) for cue_time, lattice_index in zip(sorted_times, lattice_indices)]
    return dict(t0=anchor_ms, period=period, k=lattice_indices, n=len(sorted_times), rms=math.sqrt(sum(residual * residual for residual in residuals) / len(residuals)),
                max_abs=max(abs(residual) for residual in residuals))


def snap_fit(times, lattice_indices, period):
    """Refit only the phase for a fixed period."""
    anchor_ms = sum(cue_time - lattice_index * period for cue_time, lattice_index in zip(times, lattice_indices)) / len(times)
    residuals = [cue_time - (anchor_ms + lattice_index * period) for cue_time, lattice_index in zip(times, lattice_indices)]
    return anchor_ms, math.sqrt(sum(residual * residual for residual in residuals) / len(residuals)), max(abs(residual) for residual in residuals)


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))]


def ground_truth(hot_cue_times, memory_cue_times, bpm_hint, snap_tolerance):
    """Constant grid implied by the cues. hot cues are bar starts, memory cues are beats."""
    bpm_hint = bpm_hint or 140.0
    fit_times, bars = (hot_cue_times, True) if len(set(hot_cue_times)) >= 2 else (hot_cue_times + memory_cue_times, False)
    unit = 4 if bars else 1
    hint = unit * 60000.0 / bpm_hint
    fit = fit_lattice(fit_times, hint)
    if fit is None:
        return None
    bpm_free = unit * 60000.0 / fit["period"]
    bpm_round = float(round(bpm_free))
    snapped = abs(bpm_free - bpm_round) <= snap_tolerance
    result = dict(source="hot cues (bar starts)" if bars else "all cues (beats)", n_cues=fit["n"],
               bpm_free=bpm_free, bpm_whole=bpm_round, snapped=snapped,
               free_rms_ms=fit["rms"], free_max_ms=fit["max_abs"])
    bpm = bpm_round if snapped else bpm_free
    period = unit * 60000.0 / bpm
    sorted_times = sorted(set(fit_times))
    lattice_indices = [round((cue_time - sorted_times[0]) / period) for cue_time in sorted_times]
    anchor_ms, rms_ms, max_ms = snap_fit(sorted_times, lattice_indices, period)
    beat = 60000.0 / bpm
    result.update(bpm=bpm, beat_ms=beat, bar_ms=4 * beat, t0_ms=anchor_ms, fit_rms_ms=rms_ms, fit_max_ms=max_ms)
    first = anchor_ms - math.floor(anchor_ms / (4 * beat)) * 4 * beat if bars else anchor_ms - math.floor(anchor_ms / beat) * beat
    result["first_downbeat_ms" if bars else "first_beat_phase_ms"] = first
    return result


def compare_grid(beats, ground_truth_grid):
    """Offsets of Rekordbox beats from the cue-implied beat lattice (ms, signed)."""
    if not beats or not ground_truth_grid:
        return None
    beat_milliseconds, anchor_ms = ground_truth_grid["beat_ms"], ground_truth_grid["t0_ms"]
    offsets = [((beat[2] - anchor_ms + beat_milliseconds / 2) % beat_milliseconds) - beat_milliseconds / 2 for beat in beats]
    count = len(offsets)
    return dict(median=percentile(offsets, 0.5), p05=percentile(offsets, 0.05), p95=percentile(offsets, 0.95),
                first=offsets[0], middle=offsets[count // 2], last=offsets[-1], worst=max(offsets, key=abs))


# ---------------------------------------------------------------------------------------
# database queries
# ---------------------------------------------------------------------------------------


def collect(database, share, row):
    """Everything known about one content row, or None when it has no usable grid/file."""
    source_path = Path(row.FolderPath or "")
    files = rekordbox_database.anlz_files(share, row)
    if not source_path.is_file() or "DAT" not in files:
        return None
    beats = read_pqtz(files["DAT"].read_bytes())
    if not beats:
        return None
    cues = []
    for cue_row in database.query(DjmdCue).filter(DjmdCue.ContentID == str(row.ID)).all():
        if cue_row.rb_local_deleted or cue_row.InMsec is None or cue_row.InMsec < 0:
            continue
        loop_end_ms = cue_row.OutMsec if cue_row.OutMsec not in rekordbox_database.NONE_LOOP and cue_row.OutMsec >= 0 else None
        cues.append(dict(kind=cue_row.Kind, label=("memory" if cue_row.Kind == 0 else "hot " + chr(64 + cue_row.Kind) if 1 <= cue_row.Kind <= 26 else str(cue_row.Kind)),
                         in_ms=cue_row.InMsec, out_ms=loop_end_ms, comment=cue_row.Comment or ""))
    cues.sort(key=lambda cue: cue["in_ms"])
    return dict(row=row, src=source_path, files=files, beats=beats, cues=cues)


def describe(info, snap_tolerance):
    row, beats, cues = info["row"], info["beats"], info["cues"]
    hot_cue_times = [cue["in_ms"] for cue in cues if cue["kind"] and cue["kind"] > 0]
    memory_cue_times = [cue["in_ms"] for cue in cues if cue["kind"] == 0]
    database_bpm = (row.BPM or 0) / 100.0
    tempos = sorted({beat[1] for beat in beats})
    first_downbeat_ms = next((beat[2] for beat in beats if beat[0] == 1), None)
    ground_truth_grid = ground_truth(hot_cue_times, memory_cue_times, database_bpm or beats[0][1] / 100.0, snap_tolerance)
    grid_comparison = compare_grid(beats, ground_truth_grid)
    times = [beat[2] for beat in beats]
    cue_versus_beat_offsets = []
    if times:
        import bisect
        for cue in cues:
            i = bisect.bisect_left(times, cue["in_ms"])
            candidate_indices = [j for j in (i - 1, i) if 0 <= j < len(times)]
            j = min(candidate_indices, key=lambda j: abs(times[j] - cue["in_ms"]))
            cue_versus_beat_offsets.append(dict(label=cue["label"], in_ms=cue["in_ms"], nearest_rb_beat_ms=times[j],
                                  beat_in_bar=beats[j][0], offset_ms=cue["in_ms"] - times[j]))
    return dict(
        id=str(row.ID), path=str(info["src"]), artist=artist_of(row), title=row.Title, file_type=row.FileType,
        length_s=row.Length, sample_rate=row.SampleRate, bit_rate=row.BitRate, rekordbox_bpm=database_bpm,
        anlz={anlz_name: str(anlz_path) for anlz_name, anlz_path in info["files"].items()},
        rb_grid=dict(n_beats=len(beats), tempos_bpm=[tempo / 100.0 for tempo in tempos], first_beat_ms=beats[0][2],
                     first_downbeat_ms=first_downbeat_ms, last_beat_ms=beats[-1][2], beats=[list(beat) for beat in beats]),
        cues=cues, cue_vs_rb_grid=cue_versus_beat_offsets, ground_truth=ground_truth_grid, rb_vs_ground_truth=grid_comparison,
        itunsmpb=itunsmpb(info["src"]) if info["src"].suffix.lower() in (".m4a", ".mp4", ".aac") else None)


def format_one_decimal(value):
    return "n/a" if value is None else f"{value:.1f}"


def write_md(track_data, path, source_description):
    lines = [f"# {track_data['artist']} - {track_data['title']}", "", f"Source: {source_description}", ""]
    lines += [f"- file: `{track_data['path']}`", f"- Rekordbox row ID: {track_data['id']}, file type code {track_data['file_type']}, "
          f"{track_data['sample_rate']} Hz, {track_data['bit_rate']} kbps, {track_data['length_s']} s",
          f"- Rekordbox BPM (database): {track_data['rekordbox_bpm']}"]
    for anlz_name, anlz_path in track_data["anlz"].items():
        lines.append(f"- ANLZ {anlz_name}: `{anlz_path}`")
    if track_data["itunsmpb"]:
        lines.append(f"- iTunSMPB priming {track_data['itunsmpb']['priming']} samples (Rekordbox ignores it, see handoff notes)")
    grid = track_data["rb_grid"]
    lines += ["", "## Rekordbox's own grid (PQTZ in .DAT)", "",
          f"- beats: {grid['n_beats']}, tempo values (bpm): {grid['tempos_bpm']}",
          f"- first beat {grid['first_beat_ms']} ms, first downbeat {grid['first_downbeat_ms']} ms, last beat {grid['last_beat_ms']} ms",
          f"- first 8 beats (beat-in-bar, ms): {[(beat[0], beat[2]) for beat in grid['beats'][:8]]}"]
    ground_truth_grid = track_data["ground_truth"]
    lines += ["", "## Ground truth implied by cues", ""]
    if not ground_truth_grid:
        lines.append("Not enough cues to fit a grid.")
    else:
        lines += [f"- from: {ground_truth_grid['source']}, {ground_truth_grid['n_cues']} distinct cue times",
              f"- free-fit bpm {ground_truth_grid['bpm_free']:.4f} (rms {ground_truth_grid['free_rms_ms']:.2f} ms, max {ground_truth_grid['free_max_ms']:.2f} ms); "
              f"whole-number bpm {ground_truth_grid['bpm_whole']:.0f}, snapped: {ground_truth_grid['snapped']}",
              f"- **used grid: {ground_truth_grid['bpm']:.4f} bpm, beat {ground_truth_grid['beat_ms']:.3f} ms, anchor {ground_truth_grid['t0_ms']:.2f} ms** "
              f"(fit rms {ground_truth_grid['fit_rms_ms']:.2f} ms, max {ground_truth_grid['fit_max_ms']:.2f} ms; cue times are integer ms, so about 0.3 ms rms is the floor)"]
        comparison = track_data["rb_vs_ground_truth"]
        if comparison:
            lines += [f"- Rekordbox beats minus this grid (ms): median {format_one_decimal(comparison['median'])}, 5-95% {format_one_decimal(comparison['p05'])}..{format_one_decimal(comparison['p95'])}, "
                  f"first {format_one_decimal(comparison['first'])}, middle {format_one_decimal(comparison['middle'])}, last {format_one_decimal(comparison['last'])}, worst {format_one_decimal(comparison['worst'])}"]
    lines += ["", "## Cues", "", "| type | time ms | loop end ms | comment | nearest RB beat ms | beat in bar | cue minus RB beat |", "|---|---|---|---|---|---|---|"]
    offsets_by_cue = {(matching_row["label"], matching_row["in_ms"]): matching_row for matching_row in track_data["cue_vs_rb_grid"]}
    for cue in track_data["cues"]:
        matching_row = offsets_by_cue.get((cue["label"], cue["in_ms"]), {})
        lines.append(f"| {cue['label']} | {cue['in_ms']} | {cue['out_ms'] if cue['out_ms'] is not None else ''} | {cue['comment']} | "
                 f"{matching_row.get('nearest_rb_beat_ms', '')} | {matching_row.get('beat_in_bar', '')} | {matching_row.get('offset_ms', '')} |")
    if not track_data["cues"]:
        lines.append("| (none) | | | | | | |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export(info, track_data, folder, source_description, copy_raw_files):
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(info["src"], folder / info["src"].name)
    (folder / "ground_truth.json").write_text(json.dumps(track_data, indent=1, ensure_ascii=False), encoding="utf-8")
    write_md(track_data, folder / "ground_truth.md", source_description)
    if copy_raw_files:
        for extension, file_path in info["files"].items():
            shutil.copy2(file_path, folder / file_path.name)


