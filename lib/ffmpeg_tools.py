"""Locates ffmpeg / ffprobe (and downloads them during setup if they are not installed). Standard library only.

At run time `executable(tool)` returns the program to call, in this order:
  1. FFMPEG_PATH / FFPROBE_PATH (environment variable, else data/.env)
  2. the program on PATH
  3. the copy downloaded into data/ffmpeg/bin
  4. the bare program name (the call then fails with a normal "not found" error)
The PATH of the user's system is never modified.
"""
import glob
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

from lib import paths

TOOLS = ('ffmpeg', 'ffprobe')
ENV_KEYS = {'ffmpeg': 'FFMPEG_PATH', 'ffprobe': 'FFPROBE_PATH'}
DOWNLOAD_DIR = paths.DATA / 'ffmpeg'
DOWNLOAD_BIN = DOWNLOAD_DIR / 'bin'

_BUILDS = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/'
_ARCHITECTURES = {'amd64': 'x64', 'x86_64': 'x64', 'arm64': 'arm64', 'aarch64': 'arm64'}
_DOWNLOAD_NAMES = {
    ('Windows', 'x64'): 'ffmpeg-master-latest-win64-lgpl.zip',
    ('Windows', 'arm64'): 'ffmpeg-master-latest-winarm64-lgpl.zip',
    ('Linux', 'x64'): 'ffmpeg-master-latest-linux64-lgpl.tar.xz',
    ('Linux', 'arm64'): 'ffmpeg-master-latest-linuxarm64-lgpl.tar.xz',
}


def program_name(tool):
    return tool + ('.exe' if os.name == 'nt' else '')


def works(program):
    """True if `program -version` runs."""
    try:
        return subprocess.run([str(program), '-version'], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def known_directories():
    """Folders where ffmpeg is commonly installed (nothing here is required to be on PATH)."""
    directories = []
    if os.name == 'nt':
        local_app_data = os.environ.get('LOCALAPPDATA', '')
        user_profile = os.environ.get('USERPROFILE', str(Path.home()))
        program_data = os.environ.get('ProgramData', r'C:\ProgramData')
        directories += [r'C:\ffmpeg\bin', r'C:\Program Files\ffmpeg\bin', r'C:\Program Files (x86)\ffmpeg\bin',
                        os.path.join(user_profile, 'scoop', 'shims'),
                        os.path.join(user_profile, 'scoop', 'apps', 'ffmpeg', 'current', 'bin'),
                        os.path.join(program_data, 'chocolatey', 'bin')]
        if local_app_data:
            directories.append(os.path.join(local_app_data, 'Microsoft', 'WinGet', 'Links'))
            directories += glob.glob(os.path.join(local_app_data, 'Microsoft', 'WinGet', 'Packages', '*FFmpeg*', '*', 'bin'))
            directories += glob.glob(os.path.join(local_app_data, 'Microsoft', 'WinGet', 'Packages', '*FFmpeg*', 'bin'))
    elif sys.platform == 'darwin':
        directories += ['/opt/homebrew/bin', '/usr/local/bin', '/opt/local/bin']
    else:
        directories += ['/usr/bin', '/usr/local/bin', '/snap/bin']
    directories.append(str(DOWNLOAD_BIN))
    return directories


def find_program(tool, preferred_directory=None):
    """Returns (path, source description) of a working `tool`, or (None, None)."""
    if preferred_directory:
        sibling = Path(preferred_directory) / program_name(tool)
        if sibling.is_file() and works(sibling):
            return str(sibling), 'next to ffmpeg'
    on_path = shutil.which(tool)
    if on_path and works(on_path):
        return on_path, 'found on PATH'
    for directory in known_directories():
        candidate = Path(directory) / program_name(tool)
        if candidate.is_file() and works(candidate):
            return str(candidate), 'found in ' + directory
    return None, None


def download_url():
    system = platform.system()
    architecture = _ARCHITECTURES.get(platform.machine().lower())
    name = _DOWNLOAD_NAMES.get((system, architecture))
    return _BUILDS + name if name else None


def _download_file(url, destination):
    with urllib.request.urlopen(url, timeout=60) as response, open(destination, 'wb') as output:
        total = int(response.headers.get('Content-Length') or 0)
        received, last_percent = 0, -1
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            output.write(chunk)
            received += len(chunk)
            if total:
                percent = received * 100 // total
                if percent // 5 != last_percent // 5:
                    print(f'          downloading ffmpeg: {percent}% ({received >> 20} of {total >> 20} MB)', flush=True)
                    last_percent = percent


def _extract_programs(archive_path):
    """Copies ffmpeg and ffprobe out of the downloaded archive into DOWNLOAD_BIN."""
    DOWNLOAD_BIN.mkdir(parents=True, exist_ok=True)
    wanted = {program_name(tool) for tool in TOOLS}
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                if os.path.basename(member) in wanted and '/bin/' in member:
                    with archive.open(member) as source, open(DOWNLOAD_BIN / os.path.basename(member), 'wb') as target:
                        shutil.copyfileobj(source, target)
    else:
        with tarfile.open(archive_path) as archive:
            for member in archive:
                if member.isfile() and os.path.basename(member.name) in wanted and '/bin/' in member.name:
                    source = archive.extractfile(member)
                    with open(DOWNLOAD_BIN / os.path.basename(member.name), 'wb') as target:
                        shutil.copyfileobj(source, target)
    for tool in TOOLS:
        program = DOWNLOAD_BIN / program_name(tool)
        if program.is_file():
            program.chmod(program.stat().st_mode | 0o111)


def download():
    """Downloads ffmpeg + ffprobe into data/ffmpeg/bin. Returns an error text, or None on success."""
    url = download_url()
    if url is None:
        return f'no ready-made ffmpeg download for {platform.system()} {platform.machine()}; install ffmpeg yourself (macOS: brew install ffmpeg)'
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=DOWNLOAD_DIR) as temporary_directory:
            archive_path = Path(temporary_directory) / 'ffmpeg_download'
            _download_file(url, archive_path)
            _extract_programs(archive_path)
    except Exception as error:
        return f'download failed ({type(error).__name__}: {error})'
    if not all(works(DOWNLOAD_BIN / program_name(tool)) for tool in TOOLS):
        return 'the downloaded archive did not contain a working ffmpeg and ffprobe'
    return None


def ensure_tools(current_values, force=False, allow_download=True):
    """Finds (or downloads) ffmpeg and ffprobe for setup. `current_values` = parsed .env.
    Returns {tool: (path or None, note)}."""
    results, directory = {}, None
    for attempt in (0, 1):
        results.clear()
        directory = None
        for tool in TOOLS:
            configured = current_values.get(ENV_KEYS[tool])
            if configured and not force and Path(configured).is_file() and works(configured):
                path, note = configured, 'kept from .env'
            else:
                path, note = find_program(tool, directory)
            if path:
                directory = directory or str(Path(path).parent)
            results[tool] = (path, note)
        if attempt == 0 and allow_download and not all(path for path, _ in results.values()):
            print('          ffmpeg / ffprobe not found on this computer; downloading a copy into data/ffmpeg ...', flush=True)
            error = download()
            if error:
                return {tool: (None, error) for tool in TOOLS}
            continue
        break
    for tool in TOOLS:
        path, note = results[tool]
        if path and Path(path).resolve().parent == DOWNLOAD_BIN.resolve():
            results[tool] = (path, 'downloaded to data/ffmpeg/bin')
    return results


@lru_cache(maxsize=None)
def executable(tool):
    """The ffmpeg or ffprobe program to call (see the module docstring for the lookup order)."""
    key = ENV_KEYS[tool]
    configured = os.environ.get(key) or paths.load_env().get(key)
    if configured and Path(configured).is_file():
        return configured
    on_path = shutil.which(tool)
    if on_path:
        return on_path
    downloaded = DOWNLOAD_BIN / program_name(tool)
    if downloaded.is_file():
        return str(downloaded)
    return tool


def is_available(tool):
    return shutil.which(executable(tool)) is not None
