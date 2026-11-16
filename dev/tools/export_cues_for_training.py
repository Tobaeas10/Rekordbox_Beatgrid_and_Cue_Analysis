"""Exports every track's hot/memory cues relative to its stored grid (cues.jsonl, training labels for the cue models)."""
import _bootstrap
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from lib import paths
from lib.cues.cue_export import (CUE_EXTRA_FIELDS, cue_label, beat_position, grid_summary, export_track, playlist_map)

OUT = paths.DEBUG / "export_out"


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--database-path")
    argument_parser.add_argument("--encryption-key")
    argument_parser.add_argument("--share-directory")
    argument_parser.add_argument("--playlist", action="append")
    argument_parser.add_argument("--track-ids", default="")
    arguments = argument_parser.parse_args()

    from lib.rekordbox import ground_truth
    from lib.rekordbox import database as rekordbox_database

    environment = ground_truth.load_env(paths.ENV_FILE)
    database_path, key = arguments.database_path or environment.get("REKORDBOX_DATABASE_PATH"), arguments.encryption_key or environment.get("REKORDBOX_ENCRYPTION_KEY", "")
    if not database_path:
        sys.exit("REKORDBOX_DATABASE_PATH missing (.env or --database-path)")
    database = rekordbox_database.open_database(database_path, key)
    share = rekordbox_database.share_directory_of(database, SimpleNamespace(share_directory=arguments.share_directory, database_path=database_path))

    if arguments.track_ids:
        wanted_ids = set(arguments.track_ids.split(","))
        rows = [content_row for content_row in database.query(rekordbox_database.DjmdContent).all() if str(content_row.ID) in wanted_ids and not content_row.rb_local_deleted]
    elif arguments.playlist:
        rows, seen = [], set()
        for name in arguments.playlist:
            found, near = ground_truth.playlist_rows(database, name)
            if not found:
                print(f"playlist '{name}': none found" + (f" (similar: {near})" if near else ""))
            for content_row in found:
                if content_row.ID not in seen:
                    seen.add(content_row.ID)
                    rows.append(content_row)
    else:
        rows = [content_row for content_row in database.query(rekordbox_database.DjmdContent).all() if not content_row.rb_local_deleted]
    print(f"{len(rows)} tracks to check", flush=True)

    playlists_by_content_id = playlist_map(database, ground_truth, rekordbox_database)
    OUT.mkdir(exist_ok=True)
    skipped, kinds_per_track, snap_distances = Counter(), Counter(), []
    exported_count = 0
    with open(OUT / "cues.jsonl", "w", encoding="utf-8") as output_file:
        for i, content_row in enumerate(rows, 1):
            try:
                row, skip_reason = export_track(database, share, content_row, ground_truth, rekordbox_database, playlists_by_content_id)
            except Exception as e:
                row, skip_reason = None, f"error {e!r}"
            if row is None:
                skipped[skip_reason] += 1
                continue
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            exported_count += 1
            kinds_per_track[tuple(sorted({cue["label"] for cue in row["cues"]}))] += 1
            snap_distances += [abs(cue["snap_ms"]) for cue in row["cues"]]
            if i % 200 == 0:
                print(f"{i}/{len(rows)}", flush=True)
    database.close()

    snap_distances.sort()
    snap_percentile = (lambda fraction: snap_distances[int(fraction * (len(snap_distances) - 1))]) if snap_distances else (lambda fraction: 0)
    lines = [f"tracks checked {len(rows)}, exported {exported_count}, skipped {dict(skipped)}",
             f"cue distance to nearest grid beat in ms (all cues): median {snap_percentile(.5)} p95 {snap_percentile(.95)} p99 {snap_percentile(.99)} max {snap_percentile(1)}",
             "label sets per track (top 25):"]
    lines += [f"  {count:5d}  {','.join(label_set)}" for label_set, count in kinds_per_track.most_common(25)]
    (OUT / "cues_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {OUT / 'cues.jsonl'} and {OUT / 'cues_summary.txt'}")


if __name__ == "__main__":
    main()
