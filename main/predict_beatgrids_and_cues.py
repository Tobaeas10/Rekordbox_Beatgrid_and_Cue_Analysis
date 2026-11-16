#!/usr/bin/env python3
"""predict_beatgrids_and_cues.py - beat grid AND cues for the same songs in ONE run with ONE table.
A song is 'done' when its grid is written, its cues are predicted and written, and master.db is committed (one commit per song). Writes (use --dry-run to test).
Same song selection as predict_beatgrids.py: default = added in the last 8 hours; --since-hours, --from/--to, --playlist, --names, --force (redo everything, replaces the cues this tool wrote earlier), ...
The number of parallel workers adapts to the free CPU and RAM (start with 1); --force-number-of-jobs N fixes it. Grid and cue jobs share the same workers.
Cue options: --minimum-confidence P, --skip-likely-wrong (no cues for tracks whose grid was flagged 'Likely wrong'; default: cues for everything), --no-cues (grid only).
Songs with several tempo sections, or with cues of your own in Rekordbox, get a grid but no cues (the table says why); nothing of yours is ever overwritten.
For the grid only use predict_beatgrids.py, for cues only (tracks whose grid was analysed earlier) predict_cues.py."""
import _bootstrap  # noqa: F401
import predict_beatgrids

if __name__ == "__main__":
    predict_beatgrids.main(with_cues=True, description=__doc__)
