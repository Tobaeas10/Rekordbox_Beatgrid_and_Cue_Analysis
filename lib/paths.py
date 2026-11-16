"""Single source of truth for every path in the project and for reading the .env file.
Layout (project root = the folder containing lib/):
  data/.env, data/models/{beatgrid,cues}, data/state, data/backups, data/cache   = needed to run
  logs/   = reports written by the daily scripts       debug/ = generated analysis output (git-ignored)"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
ENV_FILE = DATA / '.env'
MODELS_BEATGRID = DATA / 'models' / 'beatgrid'
MODELS_CUES = DATA / 'models' / 'cues'
STATE_DIR = DATA / 'state'
BACKUPS = DATA / 'backups'
CACHE = DATA / 'cache'
LOGS = ROOT / 'logs'
DEBUG = ROOT / 'debug'
# input data of the dev scripts (user-provided or produced by tools; all git-ignored) and their intermediates
WORK = DEBUG / 'work'                        # intermediate datasets, CV outputs and newly trained models (copy to data/models by hand)
EVAL_REPORTS = DEBUG / 'eval_reports'        # eval_report*.json written by dev/tools/evaluate_beatgrid_on_library.py
EVENT_CACHES = DEBUG / 'event_caches'        # <set name>/<track id>.ff2.pkl event caches (sets: old961, new698, extra, all_songs)
BEAT_FEATURES = DEBUG / 'beat_features'      # per-track .npz written by evaluate_beatgrid_on_library.py --beat-features
GENRE_PLAYLISTS = DEBUG / 'genre_playlists'  # Rekordbox playlist exports (UTF-16 TSV, one file per genre)
CUE_TRAINING = DEBUG / 'cue_training'        # cues.jsonl, slimmed beat/band feature dumps for cue model training
for _directory in (STATE_DIR, BACKUPS, CACHE, LOGS, DEBUG, WORK):
    _directory.mkdir(parents=True, exist_ok=True)


def load_env(path=None):
    """KEY=VALUE pairs of the .env (default data/.env); a sibling file named `_env` only fills missing keys."""
    path = Path(path) if path else ENV_FILE
    values = {}
    for candidate_path in (path, path.with_name('_env')):
        if candidate_path.exists():
            for line in candidate_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    values.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return values
