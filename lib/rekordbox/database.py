"""Rekordbox master.db access and track lookup helpers (copied unchanged from the m4a converter project's
the m4a converter and from the ground-truth exporter; the m4a project keeps its own copy)."""
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

from lib import ffmpeg_tools

# ---- pyrekordbox imports (0.5 layout first, older layout as fallback) ------------------
try:
    from pyrekordbox.masterdb import MasterDatabase as RekordboxDatabase
    from pyrekordbox.masterdb import ContentCue, DjmdContent, DjmdCue
except ImportError:  # pyrekordbox < 0.5
    from pyrekordbox import Rekordbox6Database as RekordboxDatabase
    try:
        from pyrekordbox.db6 import ContentCue, DjmdContent, DjmdCue
    except ImportError:
        from pyrekordbox.db6.tables import ContentCue, DjmdContent, DjmdCue

try:
    from pyrekordbox.masterdb import DjmdPlaylist, DjmdSongPlaylist
except ImportError:
    try:
        from pyrekordbox.db6 import DjmdPlaylist, DjmdSongPlaylist
    except ImportError:
        from pyrekordbox.db6.tables import DjmdPlaylist, DjmdSongPlaylist

EXPECTED_PRIMING = 2112          # samples added by Apple AAC (qaac)
FILE_TYPE_WAV, FILE_TYPE_M4A = 11, 4           # djmdContent.FileType codes
ANLZ_EXTENSIONS = ("DAT", "EXT", "2EX")
NONE_LOOP = (None, -1, 0xFFFFFFFF)


def frame_of(milliseconds):
    """Rekordbox cue frames are 1/150 s: floor(ms*0.15) (verified on 22k cue rows)."""
    return (milliseconds * 15) // 100


def run(command, **keyword_arguments):
    return subprocess.run(command, capture_output=True, **keyword_arguments)


def ffprobe(path):
    probe_result = run([ffmpeg_tools.executable("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
            text=True, encoding="utf-8", errors="replace")
    if probe_result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {probe_result.stderr.strip()[:200]}")
    return json.loads(probe_result.stdout)


def audio_stream(probe_data):
    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "audio":
            return stream
    raise RuntimeError("no audio stream")


def parse_smpb(tag):
    fields = tag.split()
    return {"priming": int(fields[1], 16), "end_padding": int(fields[2], 16), "valid": int(fields[3], 16)}


def open_database(path, encryption_key):
    keyword_arguments = {"path": str(path)}
    if encryption_key:
        keyword_arguments["key"] = encryption_key
    return RekordboxDatabase(**keyword_arguments)


def share_directory_of(database, arguments):
    if arguments.share_directory:
        return Path(arguments.share_directory)
    known_share_directory = getattr(database, "_share_dir", None)
    return Path(known_share_directory) if known_share_directory else Path(arguments.database_path).parent / "share"


def anlz_files(share_directory, content):
    """Existing ANLZ files {ext: Path} for a content row, resolved against the share dir."""
    if not content.AnalysisDataPath:
        return {}
    relative_path = Path(content.AnalysisDataPath.replace("\\", "/").strip("/"))
    analysis_directory = share_directory / relative_path.parent
    files_by_extension = {}
    for extension in ANLZ_EXTENSIONS:
        for file_path in analysis_directory.glob(f"ANLZ*.{extension}"):
            files_by_extension[extension] = file_path
    return files_by_extension


def rekordbox_running():
    if os.name != "nt":
        return False
    process_result = run(["tasklist", "/FI", "IMAGENAME eq rekordbox.exe"], text=True, errors="replace")
    return "rekordbox.exe" in (process_result.stdout or "").lower()


def norm_path(path):
    return (path or "").replace("/", "\\").lower()


def artist_of(row):
    for attr in ("ArtistName",):
        try:
            artist_name = getattr(row, attr)
            if artist_name:
                return artist_name
        except Exception:
            pass
    try:
        return row.Artist.Name
    except Exception:
        return ""


def find_by_path(database, path):
    wanted_path = norm_path(path)
    for row in database.query(DjmdContent).filter(DjmdContent.FolderPath.like("%" + Path(path).stem[:20] + "%")).all():
        if norm_path(row.FolderPath) == wanted_path and not row.rb_local_deleted:
            return row
    return None


def find_by_text(database, text):
    """Rows whose file name or title contains text (case-insensitive, 'Artist - Title' matches the file name), not deleted."""
    lowered_text = text.strip().lower()
    found_rows = [row for row in database.query(DjmdContent).filter(DjmdContent.FolderPath.ilike("%" + text.strip() + "%")).all() if not row.rb_local_deleted]
    if not found_rows:
        title = lowered_text.split(" - ", 1)[-1]
        found_rows = [row for row in database.query(DjmdContent).filter(DjmdContent.Title.ilike("%" + title + "%")).all()
                if not row.rb_local_deleted and lowered_text.split(" - ", 1)[0] in (artist_of(row) or "").lower() + " " + str(row.FolderPath).lower()]
    return found_rows


def playlist_rows(database, name):
    playlists = [playlist for playlist in database.query(DjmdPlaylist).all() if not playlist.rb_local_deleted]
    exact_matches = [playlist for playlist in playlists if (playlist.Name or "").strip().lower() == name.lower()]
    if not exact_matches:
        return [], sorted({playlist.Name for playlist in playlists if name.lower()[:3] in (playlist.Name or "").lower()})
    content_rows, seen_content_ids = [], set()
    for playlist in exact_matches:
        for song_entry in database.query(DjmdSongPlaylist).filter(DjmdSongPlaylist.PlaylistID == playlist.ID).all():
            if song_entry.rb_local_deleted or song_entry.ContentID in seen_content_ids:
                continue
            seen_content_ids.add(song_entry.ContentID)
            content_row = database.query(DjmdContent).filter(DjmdContent.ID == song_entry.ContentID).first()
            if content_row is not None and not content_row.rb_local_deleted:
                content_rows.append(content_row)
    return content_rows, []
