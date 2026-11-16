"""Exports example tracks (audio + stored grid + cues) with ground truth for development. See ground_truth.py for the helpers."""
import _bootstrap
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from lib import paths
from lib.paths import load_env
from lib.rekordbox import database as rekordbox_database
from lib.rekordbox.database import DjmdContent, DjmdCue, DjmdPlaylist, DjmdSongPlaylist, artist_of, find_by_path, find_by_text, norm_path, playlist_rows
from lib.rekordbox.ground_truth import (EXTRA_TRACKS, MAIN_TRACK, OUT_DIR, PLAYLISTS, safe_name, read_pqtz, itunsmpb, linfit, fit_lattice, snap_fit, percentile, ground_truth, compare_grid, collect, describe, format_one_decimal, write_md, export)


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--database-path")
    argument_parser.add_argument("--encryption-key")
    argument_parser.add_argument("--share-directory")
    argument_parser.add_argument("--output-folder", default=OUT_DIR)
    argument_parser.add_argument("--main-track", default=MAIN_TRACK)
    argument_parser.add_argument("--playlist", action="append", help="playlist name (repeatable), default: the four genre playlists")
    argument_parser.add_argument("--per-playlist", type=int, default=3)
    argument_parser.add_argument("--seed", type=int, default=1)
    argument_parser.add_argument("--bpm-snap-tolerance", type=float, default=0.1)
    argument_parser.add_argument("--raw-files-for-all-tracks", action="store_true", help="copy raw ANLZ files for every track (default: main track only)")
    argument_parser.add_argument("--extra-track-search-text", action="append", help="extra track by file name or title text (repeatable), exported with raw ANLZ files; default: the tempo change tracks in EXTRA_TRACKS")
    argument_parser.add_argument("--only-extras", action="store_true", help="export only the --extra-track-search-text / EXTRA_TRACKS tracks, no main track and no playlist sample")
    argument_parser.add_argument("--list-only", action="store_true", help="only print what would be exported")
    arguments = argument_parser.parse_args()
    environment = load_env(paths.ENV_FILE)
    database_path, key = arguments.database_path or environment.get("REKORDBOX_DATABASE_PATH"), arguments.encryption_key or environment.get("REKORDBOX_ENCRYPTION_KEY", "")
    if not database_path:
        sys.exit("REKORDBOX_DATABASE_PATH missing (.env or --database-path)")
    if rekordbox_database.rekordbox_running():
        print("warning: Rekordbox is running, the database may be locked or stale")

    database = rekordbox_database.open_database(database_path, key)
    share = rekordbox_database.share_directory_of(database, SimpleNamespace(share_directory=arguments.share_directory, database_path=database_path))
    output_folder = Path(arguments.output_folder)
    random_generator = random.Random(arguments.seed)
    jobs, index = [], []

    main_row = None if arguments.only_extras else find_by_path(database, arguments.main_track)
    if arguments.only_extras:
        pass
    elif main_row is None:
        print(f"main track not found in database: {arguments.main_track}")
    else:
        info = collect(database, share, main_row)
        if info is None:
            print("main track found, but audio file or ANLZ grid is missing")
        else:
            jobs.append(("00_main", info, "main example (easy dubstep)", True))

    taken = {str(main_row.ID)} if main_row is not None else set()
    for track_number, text in enumerate(arguments.extra_track_search_text or EXTRA_TRACKS, 1):
        rows = find_by_text(database, text)
        if not rows:
            print(f"extra track not found: {text}")
            continue
        if len(rows) > 1:
            print(f"extra track '{text}': {len(rows)} matches, using the first; ids " + ", ".join(str(content_row.ID) for content_row in rows))
        info = collect(database, share, rows[0])
        if info is None:
            print(f"extra track '{text}' found (id {rows[0].ID}), but audio file or ANLZ grid is missing")
            continue
        print(f"extra track '{text}': id {rows[0].ID}")
        taken.add(str(rows[0].ID))
        jobs.append((f"extra_{track_number:02d}", info, "extra (tempo change track)", True))
    for name in ([] if arguments.only_extras else (arguments.playlist or PLAYLISTS)):
        rows, near = playlist_rows(database, name)
        if not rows:
            print(f"playlist '{name}': no tracks found" + (f" (similar names: {near})" if near else ""))
            continue
        infos = []
        for content_row in rows:
            if str(content_row.ID) in taken:
                continue
            track_info = collect(database, share, content_row)
            if track_info:
                infos.append(track_info)
        with_cues = [track_info for track_info in infos if any(cue["kind"] and cue["kind"] > 0 for cue in track_info["cues"])]
        other_tracks = [track_info for track_info in infos if track_info not in with_cues]
        random_generator.shuffle(with_cues)
        random_generator.shuffle(other_tracks)
        picked_tracks = (with_cues + other_tracks)[: arguments.per_playlist]
        print(f"playlist '{name}': {len(rows)} tracks, {len(infos)} usable, {len(with_cues)} with hot cues, picked {len(picked_tracks)}")
        for track_number, track_info in enumerate(picked_tracks, 1):
            taken.add(str(track_info["row"].ID))
            jobs.append((f"{safe_name(name)}_{track_number:02d}", track_info, f"playlist '{name}'", arguments.raw_files_for_all_tracks))

    summary = []
    for label, info, source_description, copy_raw_files in jobs:
        track_data = describe(info, arguments.bpm_snap_tolerance)
        folder = output_folder / f"{label} - {safe_name(track_data['artist'] + ' - ' + track_data['title'], 60)}"
        ground_truth_grid, comparison = track_data["ground_truth"], track_data["rb_vs_ground_truth"]
        line = (f"{label}: {track_data['artist']} - {track_data['title']} | RB bpm {track_data['rekordbox_bpm']} | "
                + (f"cue grid {ground_truth_grid['bpm']:.3f} bpm | RB grid off by median {format_one_decimal(comparison['median'])} ms, worst {format_one_decimal(comparison['worst'])} ms"
                   if ground_truth_grid and comparison else "no cue-based grid"))
        print(line)
        summary.append(line)
        if not arguments.list_only:
            export(info, track_data, folder, source_description, copy_raw_files)
    if not arguments.list_only and jobs:
        output_folder.mkdir(parents=True, exist_ok=True)
        (output_folder / "index.txt").write_text(f"# exported {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "\n".join(summary) + "\n", encoding="utf-8")
        print(f"\nwritten to {output_folder}")
    database.close()


if __name__ == "__main__":
    main()
