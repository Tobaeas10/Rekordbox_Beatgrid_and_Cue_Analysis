"""Track selection shared by the beatgrid and cue scripts (added-since, id/range/playlist/name filters)."""
from datetime import datetime, timedelta

def added_at(row):
    for column_name in ("created_at", "DateCreated", "StockDate"):
        value = getattr(row, column_name, None)
        if isinstance(value, datetime):
            return value.replace(tzinfo=None)
        if isinstance(value, str) and value[:10].count("-") == 2:
            try:
                return datetime.fromisoformat(value[:19].replace("T", " "))
            except ValueError:
                return datetime.fromisoformat(value[:10])
    return None


def select(database, arguments, ground_truth, DjmdContent):
    rows = [row for row in database.query(DjmdContent).all() if not row.rb_local_deleted]
    if arguments.playlist:
        playlist_track_ids = set()
        for playlist_name in arguments.playlist:
            playlist_track_ids |= {str(row.ID) for row in ground_truth.playlist_rows(database, playlist_name)[0]}
        return [row for row in rows if str(row.ID) in playlist_track_ids]
    if arguments.names:
        wanted_names = [name_part.strip().lower() for name_part in arguments.names.split(",") if name_part.strip()]
        return [row for row in rows if any(wanted_name in (str(row.FolderPath) + " " + str(row.Title)).lower() for wanted_name in wanted_names)]
    range_start = datetime.fromisoformat(arguments.date_from) if arguments.date_from else datetime.now() - timedelta(hours=arguments.since_hours)
    range_end = datetime.fromisoformat(arguments.date_to) + (timedelta(days=1) if arguments.date_to and len(arguments.date_to) <= 10 else timedelta(0)) if arguments.date_to else None
    selected_rows = []
    for row in rows:
        added_time = added_at(row)
        date_only = isinstance(getattr(row, "created_at", None), datetime) is False
        if added_time is None:
            continue
        if date_only and not arguments.date_from:
            added_time = added_time + timedelta(days=1)  # no time of day available: whole day counts
        if added_time >= range_start and (range_end is None or added_time < range_end):
            selected_rows.append(row)
    return selected_rows


def describe(arguments):
    if arguments.playlist:
        return "playlist " + ", ".join(arguments.playlist)
    if arguments.names:
        return f"songs matching {arguments.names}"
    if arguments.date_from:
        return f"songs added {arguments.date_from} to {arguments.date_to or 'now'}"
    return f"songs added in the last {arguments.since_hours:g} hours"
