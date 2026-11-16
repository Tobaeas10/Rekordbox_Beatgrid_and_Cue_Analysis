"""Round-trip check of anlz_grid on real ANLZ .DAT files: python dev/anlz_real_check.py path/ANLZ0000.DAT [...]"""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
import sys
from pathlib import Path
from lib.rekordbox import anlz_files

for anlz_path in sys.argv[1:]:
    file_bytes = Path(anlz_path).read_bytes()
    beats = anlz_files.read_pqtz(file_bytes)
    shifted = [(beat_number, beat_tempo, beat_time_ms + 5) for beat_number, beat_tempo, beat_time_ms in beats]
    print(anlz_path, 'beats', len(beats), 'structure', anlz_files.check_structure(file_bytes), 'identity rebuild', anlz_files.set_pqtz(file_bytes, beats) == file_bytes, 'shifted ok', anlz_files.roundtrip_ok(file_bytes, shifted))
