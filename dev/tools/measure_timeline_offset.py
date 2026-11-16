#!/usr/bin/env python3
"""
measure_timeline_offset.py - measure the time offset between Rekordbox's own timeline and the ffmpeg-decoded audio by
aligning the detail waveform Rekordbox stored in the track's ANLZ .EXT file (PWV3 or PWV5, 150 entries per
second, computed by Rekordbox from its own decode) with the peak envelope of the decoded audio.

offset = Rekordbox time minus decoded time (same sign convention as evaluate_beatgrid_on_library.py). This needs no rule about
container headers: it measures what Rekordbox actually did with each file.

  python dev/tools/measure_timeline_offset.py                 # every track in eval_out/eval_report.json (or ./eval_report.json)
  python dev/tools/measure_timeline_offset.py REPORT.json
Writes timeline_offset_report.txt next to the script. Needs numpy, scipy, ffmpeg/ffprobe; Rekordbox may stay open.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

import _bootstrap
from lib.beatgrid import tempo_detection

from lib.beatgrid import grid_pipeline

iter_tags = grid_pipeline.iter_tags
read_detail_waveform = grid_pipeline.read_detail_waveform
measure = grid_pipeline.measure_waveform_offset


def main():
    report_path = Path(sys.argv[1]) if len(sys.argv) > 1 else next(
        (Path(candidate_path) for candidate_path in ("eval_out/eval_report.json", "eval_report.json") if Path(candidate_path).exists()), None)
    if report_path is None or not report_path.exists():
        sys.exit("no eval_report.json found; pass its path")
    lines = []
    for track_record in json.loads(report_path.read_text(encoding="utf-8")):
        name = f"{track_record.get('artist', '')} - {track_record.get('title', '')}"[:48]
        extended_file = (track_record.get("anlz") or {}).get("EXT")
        try:
            data = Path(extended_file).read_bytes()
            waveform = read_detail_waveform(data)
            if waveform is None:
                raise RuntimeError("no PWV3/PWV5 tag in EXT")
            kind, waveform_values = waveform
            samples, sample_rate = tempo_detection.load_audio(track_record["path"])
            offset_ms, correlation, runner_up_correlation = measure(waveform_values, samples, sample_rate)
            sections = [section for section in track_record.get("sections", []) if "offset_mod_half_beat_ms" in section]
            beat = f"{sections[-1]['offset_mod_half_beat_ms']:+7.1f}" if sections else "   n/a"
            line = (f"{name:48s} {Path(track_record['path']).suffix:5s} | waveform offset {offset_ms:+7.1f} ms (corr {correlation:.2f}, runner-up {runner_up_correlation:.2f}, {kind}) "
                    f"| beat-based offset (mod T/2) {beat} ms | in units of 25.057 ms: {offset_ms / 25.057:+.2f}")
        except Exception as e:
            line = f"{name:48s} ERROR {e!r}"
        print(line, flush=True)
        lines.append(line)
        Path("timeline_offset_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
