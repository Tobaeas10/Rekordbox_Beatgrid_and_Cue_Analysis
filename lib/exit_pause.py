"""Keeps the console window open until Enter is pressed when a script ends, so a double-clicked script does not
vanish with its output. Standard library only.

run(entry_point) is meant for `if __name__ == '__main__':`. It does nothing special outside an interactive console,
and is skipped by --no-pause (removed from sys.argv before the script parses its options), by the environment
variable BEATGRID_NO_PAUSE, and when help (-h / --help) was asked for."""
import os
import sys
import traceback

NO_PAUSE_FLAG = '--no-pause'


def run(entry_point):
    pause = NO_PAUSE_FLAG not in sys.argv and not os.environ.get('BEATGRID_NO_PAUSE')
    sys.argv[:] = [argument for argument in sys.argv if argument != NO_PAUSE_FLAG]
    if any(argument in ('-h', '--help') for argument in sys.argv[1:]):
        pause = False
    exit_code = 0
    try:
        entry_point()
    except SystemExit as stop:
        exit_code = stop.code
        if isinstance(exit_code, str):
            print(exit_code, file=sys.stderr)
            exit_code = 1
    except KeyboardInterrupt:
        print('\nStopped.')
        exit_code = 130
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    if pause and sys.stdin and sys.stdin.isatty() and sys.stdout.isatty():
        try:
            input('\nPress Enter to close this window ... ')
        except (EOFError, KeyboardInterrupt):
            pass
    sys.exit(exit_code)
