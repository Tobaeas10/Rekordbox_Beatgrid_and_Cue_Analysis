"""Snap cues/loop ends to the beat grid (logic of snap_cues_to_grid.py). Pure functions on beat times and raw ANLZ bytes."""
import bisect
import struct

from lib.rekordbox import anlz_files
from lib.rekordbox.database import ContentCue, DjmdContent, DjmdCue  # noqa: F401  (re-exported for predict_beatgrids)


def beats_from_dat(data):
    for tag_type, offset, header_length, tag_length in anlz_files.iter_tags(data):
        if tag_type == "PQTZ":
            entry_count = struct.unpack_from(">I", data, offset + 20)[0]
            return sorted(struct.unpack_from(">I", data, offset + 24 + 8 * entry_index + 4)[0] for entry_index in range(entry_count))
    return []


def snap(time_ms, beats, margin):
    insert_index = bisect.bisect_left(beats, time_ms)
    neighbour_beats = ([beats[insert_index]] if insert_index < len(beats) else []) + ([beats[insert_index - 1]] if insert_index > 0 else [])
    if not neighbour_beats:
        return time_ms
    best = min(neighbour_beats, key=lambda beat: (abs(beat - time_ms), beat))
    return best if abs(best - time_ms) <= margin else time_ms


def snap_anlz(data, beats, margin):
    """Patch cue/loop times in raw ANLZ bytes. Returns (bytes, [(old, new)])."""
    patched_data = bytearray(data)
    changes = []
    for tag_type, offset, header_length, tag_length in anlz_files.iter_tags(data):
        if tag_type not in ("PCOB", "PCO2"):
            continue
        for time_field_offset, loop_field_offset in anlz_files.cue_entries(patched_data, tag_type, offset, header_length):
            time_ms = struct.unpack_from(">I", patched_data, time_field_offset)[0]
            loop_ms = struct.unpack_from(">I", patched_data, loop_field_offset)[0]
            if loop_ms != 0xFFFFFFFF and loop_ms >= time_ms:
                snapped_loop_ms = snap(loop_ms, beats, margin)
                if snapped_loop_ms != loop_ms:
                    struct.pack_into(">I", patched_data, loop_field_offset, snapped_loop_ms)
                    changes.append((loop_ms, snapped_loop_ms))
            snapped_time_ms = snap(time_ms, beats, margin)
            if snapped_time_ms != time_ms:
                struct.pack_into(">I", patched_data, time_field_offset, snapped_time_ms)
                changes.append((time_ms, snapped_time_ms))
    return bytes(patched_data), changes
