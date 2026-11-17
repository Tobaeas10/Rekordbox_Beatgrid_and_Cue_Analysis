"""One-step installation, started by `SETUP (Windows).bat` or `SETUP (MacOS).command` (those also install Python 3.14 first if it is
missing); `python -m lib.setup_install` from the project folder does the same.

  1. checks the Python version
  2. creates the virtual environment .venv in this folder
  3. installs the packages of requirements.txt into it
  4. runs lib/populate_env.py inside the .venv: finds Rekordbox's database and key, finds (or downloads) ffmpeg
     and ffprobe, and writes everything to data/.env
  5. creates a shortcut to each of the three analysis scripts in this folder (Windows: .lnk, macOS: .command)

Run it again at any time to repair or update the installation, or after moving the project folder. Extra options are
passed on to populate_env (--force re-detects everything, --no-download never downloads ffmpeg); --no-pause
closes the window without waiting for Enter.
"""
import os
import shlex
import subprocess
import sys
import venv
from pathlib import Path

from lib import exit_pause

MINIMUM_PYTHON = (3, 14)
ROOT = Path(__file__).resolve().parents[1]
VENV_DIRECTORY = ROOT / '.venv'
VENV_PYTHON = VENV_DIRECTORY / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
LAUNCHERS = (
    ('Beatgrids and Cues', 'predict_beatgrids_and_cues.py', 'Fix the beat grids and set the cues of the tracks added in the last 8 hours'),
    ('Beatgrids only', 'predict_beatgrids.py', 'Fix the beat grids of the tracks added in the last 8 hours'),
    ('Cues only', 'predict_cues.py', 'Set the cues of the tracks added in the last 8 hours'),
)


def stop(message):
    print(f'\nSETUP STOPPED: {message}')
    sys.exit(1)


def venv_version():
    try:
        output = subprocess.run([str(VENV_PYTHON), '-c', 'import sys; print(sys.version_info[0], sys.version_info[1])'],
                                capture_output=True, text=True, timeout=60).stdout.split()
        return int(output[0]), int(output[1])
    except Exception:
        return None


def create_venv():
    existing = venv_version() if VENV_PYTHON.is_file() else None
    if existing and existing >= MINIMUM_PYTHON:
        print(f'[ok]  virtual environment .venv exists (Python {existing[0]}.{existing[1]})')
        return
    print('[..]  creating virtual environment .venv ...')
    try:
        venv.EnvBuilder(with_pip=True, clear=VENV_DIRECTORY.exists()).create(VENV_DIRECTORY)
    except Exception as error:
        stop(f'could not create the virtual environment: {error}')
    if not VENV_PYTHON.is_file():
        stop(f'the virtual environment has no {VENV_PYTHON.name}')
    print('[ok]  virtual environment created')


def install_requirements():
    print('[..]  installing packages (this can take a few minutes) ...')
    commands = ([str(VENV_PYTHON), '-m', 'pip', 'install', '--upgrade', 'pip'],
                [str(VENV_PYTHON), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements.txt')])
    for command in commands:
        if subprocess.call(command, cwd=ROOT) != 0:
            stop('pip could not install the packages (see the messages above). Check your internet connection and run SETUP again.')
    print('[ok]  packages installed')


def check_models():
    missing = [name for name, pattern in (('beatgrid', '*.json'), ('cues', '*.pkl'))
               if not list((ROOT / 'data' / 'models' / name).glob(pattern))]
    if missing:
        print(f'[WARN] no trained models found in data/models/{" and data/models/".join(missing)}. The download of the repository seems incomplete.')


def make_windows_shortcut(name, script, description):
    link = ROOT / f'{name}.lnk'
    environment = dict(os.environ, LINK_PATH=str(link), LINK_TARGET=str(VENV_PYTHON), LINK_ARGUMENTS=f'"{ROOT / "main" / script}"',
                       LINK_DIRECTORY=str(ROOT), LINK_DESCRIPTION=description)
    command = ('$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:LINK_PATH);'
               '$s.TargetPath = $env:LINK_TARGET; $s.Arguments = $env:LINK_ARGUMENTS; $s.WorkingDirectory = $env:LINK_DIRECTORY;'
               '$s.Description = $env:LINK_DESCRIPTION; $s.Save()')
    try:
        subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', command], env=environment, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pass
    if link.is_file():
        return link
    batch = ROOT / f'{name}.bat'
    batch.write_text(f'@echo off\r\nchcp 65001 >nul\r\n"{VENV_PYTHON}" "{ROOT / "main" / script}"\r\n', encoding='utf-8')
    return batch


def make_mac_command(name, script, description):
    command_file = ROOT / f'{name}.command'
    command_file.write_text(f'#!/bin/bash\ncd {shlex.quote(str(ROOT))} || exit 1\n'
                            f'{shlex.quote(str(VENV_PYTHON))} {shlex.quote(str(ROOT / "main" / script))} "$@"\n', encoding='utf-8')
    command_file.chmod(0o755)
    return command_file


def create_launchers():
    if sys.platform == 'win32':
        make = make_windows_shortcut
    elif sys.platform == 'darwin':
        make = make_mac_command
    else:
        return []
    created = []
    for name, script, description in LAUNCHERS:
        try:
            created.append(make(name, script, description))
        except OSError as error:
            print(f'[WARN] could not create the launcher "{name}": {error}')
    return created


def main():
    if sys.version_info < MINIMUM_PYTHON:
        stop(f'Python {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or newer is required (this is {sys.version_info[0]}.{sys.version_info[1]}). '
             'Install it from https://www.python.org/downloads/ and run SETUP again.')
    print(f'Setting up in {ROOT}\n')
    create_venv()
    install_requirements()
    check_models()
    print('[..]  detecting Rekordbox database, encryption key, ffmpeg and ffprobe ...\n')
    exit_code = subprocess.call([str(VENV_PYTHON), '-m', 'lib.populate_env', *sys.argv[1:]], cwd=ROOT)
    if exit_code != 0:
        print('\nSetup is not finished: fix what is listed above, then run SETUP again.')
        sys.exit(exit_code)
    launchers = create_launchers()
    print('\nSetup finished.')
    if launchers:
        print('\nTo analyse new tracks, double-click one of these in this folder (copy it to your Desktop or anywhere you like):')
        for launcher in launchers:
            print(f'   {launcher.name}')
        print('If you move the project folder, run SETUP again.')
    else:
        print('\nTo analyse new tracks run:  python main/predict_beatgrids_and_cues.py')


if __name__ == '__main__':
    exit_pause.run(main)
