#!/usr/bin/env python3
"""
cue_export.py - export hot cues, memory cues and the stored beat grid of every track as ground truth for the
cue predictor (see docs/handoff_cues.md). Read-only: nothing is written to Rekordbox. Rekordbox should be closed.
Settings come from .env (REKORDBOX_DATABASE_PATH, REKORDBOX_ENCRYPTION_KEY) like evaluate_beatgrid_on_library.py. Library module of dev/tools/export_cues_for_training.py.

  python dev/tools/export_cues_for_training.py                       all tracks that have at least one hot or memory cue
  python dev/tools/export_cues_for_training.py --playlist Dubstep    only tracks of a playlist (repeatable)
  python dev/tools/export_cues_for_training.py --track-ids 123,456

Output (export_out/): cues.jsonl (one JSON object per track), cues_summary.txt (counts and sanity checks).
Per cue: kind (raw DB value, 0 memory, 1,2,3,5,6,7,8,9 = hot A..H, 4 unused) and label (letter), in_ms, out_ms (loop end or null), beat (fractional beat index counted
from the first PQTZ beat, interpolated on the stored grid), bar (beat relative to the first beat with beat-in-bar 1,
divided by 4), snap_ms (distance to the nearest grid beat), comment, color fields when present.
Join key to the beat-features cache is the track id (beat_features/<id>.npz).
"""
import argparse
import bisect
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


CUE_EXTRA_FIELDS = ("Color", "ColorTableIndex", "ActiveLoop", "BeatLoopSize", "CueSize")


def cue_label(kind):
    """Rekordbox DB Kind -> hot cue letter. Kind 0 = memory cue; the DB skips 4 (A,B,C = 1,2,3; D..H = 5..9; verified on the library)."""
    if kind == 0:
        return "memory"
    if 1 <= kind <= 3:
        return chr(64 + kind)
    if kind >= 5:
        return chr(63 + kind)
    return f"kind{kind}"


def beat_position(times, milliseconds):
    """Fractional beat index of time ms on the grid given as a sorted list of beat times in ms.
    Extrapolates with the first/last beat interval outside the grid."""
    beat_count = len(times)
    if beat_count < 2:
        return None
    if milliseconds <= times[0]:
        return (milliseconds - times[0]) / (times[1] - times[0])
    if milliseconds >= times[-1]:
        return (beat_count - 1) + (milliseconds - times[-1]) / (times[-1] - times[-2])
    beat_index = bisect.bisect_right(times, milliseconds) - 1
    return beat_index + (milliseconds - times[beat_index]) / (times[beat_index + 1] - times[beat_index])


def grid_summary(beats):
    """beats = [(beat_in_bar, tempo_x100, time_ms), ...] from the PQTZ tag."""
    times = [beat[2] for beat in beats]
    tempos = sorted({beat[1] for beat in beats})
    first_bar = next((beat_index for beat_index, beat in enumerate(beats) if beat[0] == 1), None)
    return times, tempos, first_bar


def export_track(database, share, content, ground_truth, rekordbox_database, playlists_of):
    files = rekordbox_database.anlz_files(share, content)
    if "DAT" not in files:
        return None, "no DAT file"
    beats = ground_truth.read_pqtz(files["DAT"].read_bytes())
    if len(beats) < 4:
        return None, "no grid"
    times, tempos, first_bar = grid_summary(beats)
    if first_bar is None:
        return None, "no beat 1 in grid"
    cues = []
    for cue_row in database.query(rekordbox_database.DjmdCue).filter(rekordbox_database.DjmdCue.ContentID == str(content.ID)).all():
        if cue_row.rb_local_deleted or cue_row.InMsec is None or cue_row.InMsec < 0:
            continue
        kind = int(cue_row.Kind or 0)
        loop_end = cue_row.OutMsec if cue_row.OutMsec not in rekordbox_database.NONE_LOOP and cue_row.OutMsec >= 0 else None
        beat = beat_position(times, cue_row.InMsec)
        insert_index = bisect.bisect_left(times, cue_row.InMsec)
        near = min(times[max(insert_index - 1, 0):insert_index + 1], key=lambda beat_time: abs(beat_time - cue_row.InMsec))
        cue = dict(kind=kind, label=cue_label(kind),
                   in_ms=int(cue_row.InMsec), out_ms=None if loop_end is None else int(loop_end),
                   beat=None if beat is None else round(beat, 4),
                   bar=None if beat is None else round((beat - first_bar) / 4.0, 4),
                   snap_ms=int(cue_row.InMsec - near), comment=cue_row.Comment or "")
        for field_name in CUE_EXTRA_FIELDS:
            field_value = getattr(cue_row, field_name, None)
            if field_value is not None:
                cue[field_name] = field_value
        cues.append(cue)
    if not cues:
        return None, "no cues"
    cues.sort(key=lambda exported_cue: exported_cue["in_ms"])
    row = dict(id=str(content.ID), name=f"{ground_truth.artist_of(content)} - {content.Title}", file=Path(content.FolderPath or "").name,
               length_s=getattr(content, "Length", None), rb_bpm=(content.BPM or 0) / 100.0,
               playlists=sorted(playlists_of.get(content.ID, [])),
               grid=dict(n_beats=len(times), tempos_x100=tempos, first_ms=times[0], last_ms=times[-1],
                         first_beat_in_bar=beats[0][0], first_bar_beat_idx=first_bar,
                         multi_tempo=len(tempos) > 1),
               cues=cues)
    if len(tempos) > 1:
        row["grid"]["beat_times_ms"] = times
    return row, None


def playlist_map(database, ground_truth, rekordbox_database):
    """content id -> playlist names (all playlists the track is in)."""
    names = {playlist.ID: (playlist.Name or "") for playlist in database.query(ground_truth.DjmdPlaylist).all() if not playlist.rb_local_deleted}
    playlists_by_content = {}
    for song_playlist in database.query(ground_truth.DjmdSongPlaylist).all():
        if song_playlist.rb_local_deleted or song_playlist.PlaylistID not in names:
            continue
        playlists_by_content.setdefault(song_playlist.ContentID, set()).add(names[song_playlist.PlaylistID])
    return playlists_by_content


