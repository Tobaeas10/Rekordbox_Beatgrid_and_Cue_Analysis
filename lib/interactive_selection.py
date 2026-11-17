"""Menu that asks which tracks to analyse when the default selection (tracks added in the last 8 hours) has nothing
to do. Used by the analysis scripts, only in an interactive console and only when no selection option
(--since-hours, --from, --to, --playlist, --names) was given on the command line. Standard library only.

ask() returns the argument list to start the script again with, or None to quit. When the selection made in the
menu is empty as well, the menu is offered again (should_ask recognises the argument list the menu produced)."""
import os
import shlex
import sys
from datetime import datetime

SELECTION_OPTIONS = ('--since-hours', '--from', '--to', '--playlist', '--names')
DATE_HELP = 'Format: YYYY-MM-DD, for example 2026-11-01. A time of day can be added: "2026-11-01 14:30".'
_made_by_menu = []


def should_ask(argv):
    if not (sys.stdin and sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    if list(argv) == _made_by_menu:
        return True
    return not any(argument.split('=')[0] in SELECTION_OPTIONS for argument in argv)


def split_options(line):
    tokens = shlex.split(line, posix=(os.name != 'nt'))
    return [token[1:-1] if len(token) >= 2 and token[0] == token[-1] and token[0] in '"\'' else token for token in tokens]


def _ask_hours():
    while True:
        text = input('How many hours back? For example 24 or 72 (Enter = back to the menu): ').strip().replace(',', '.')
        if not text:
            return None
        try:
            hours = float(text)
        except ValueError:
            hours = 0
        if hours > 0:
            return ['--since-hours', f'{hours:g}']
        print('Please type a number of hours, for example 48.')


def _ask_date(prompt, allow_empty):
    while True:
        text = input(prompt).strip()
        if not text:
            return '' if allow_empty else None
        try:
            datetime.fromisoformat(text)
            return text
        except ValueError:
            print(f'That is not a valid date. {DATE_HELP}')


def _ask_dates():
    print(f'Tracks by the date they were added to Rekordbox.\n{DATE_HELP}')
    date_from = _ask_date('From (Enter = back to the menu): ', allow_empty=False)
    if date_from is None:
        return None
    date_to = _ask_date('To (a date without a time includes that whole day; Enter = until now): ', allow_empty=True)
    return ['--from', date_from] + (['--to', date_to] if date_to else [])


def _ask_playlist():
    name = input('Playlist name, exactly as shown in Rekordbox (Enter = back to the menu): ').strip()
    return ['--playlist', name] if name else None


def _ask_own_options(argument_parser, validate):
    print(argument_parser.format_help())
    print('Type the options you want, for example:  --names "Track name" --force')
    while True:
        line = input('Options (Enter = back to the menu): ').strip()
        if not line:
            return None
        try:
            tokens = split_options(line)
            validate(tokens)
            return tokens
        except (SystemExit, ValueError):
            print('That did not work (see the message above). Try again.')


def _without_selection(argv):
    """argv minus the selection options and their values (what an earlier menu answer put there)."""
    kept, skip_value = [], False
    for argument in argv:
        if skip_value:
            skip_value = False
        elif argument.split('=')[0] in SELECTION_OPTIONS:
            skip_value = '=' not in argument
        else:
            kept.append(argument)
    return kept


def ask(argument_parser, validate, argv, reason):
    """reason: why nothing is left to do. validate(tokens): parses tokens with the script's parser (raises SystemExit if invalid)."""
    print(f'\n{reason}')
    base = _without_selection(argv) if list(argv) == _made_by_menu else list(argv)
    try:
        while True:
            print('\nWhat do you want to analyse?')
            print('  1  tracks added in the last N hours')
            print('  2  tracks added between two dates')
            print('  3  all tracks of a playlist')
            print('  4  show all options (--help) and type my own')
            print('  Enter  quit')
            choice = input('Your choice: ').strip()
            if not choice:
                return None
            if choice == '4':
                tokens = _ask_own_options(argument_parser, validate)
                if tokens is not None:
                    _made_by_menu[:] = tokens
                    return tokens
                continue
            asker = {'1': _ask_hours, '2': _ask_dates, '3': _ask_playlist}.get(choice)
            if asker is None:
                print('Please type 1, 2, 3 or 4.')
                continue
            fragment = asker()
            if fragment is not None:
                result = base + fragment
                _made_by_menu[:] = result
                return result
    except (EOFError, KeyboardInterrupt):
        return None
