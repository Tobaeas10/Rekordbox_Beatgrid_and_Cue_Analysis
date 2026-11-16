"""Live progress table for predict_beatgrids.py (stdlib only; Windows 10/11 and macOS/Linux terminals).

The table is drawn in the terminal's alternate screen and redrawn in place once per second, so it never scrolls away.
Scrolling is done by the program itself: Up/Down, PgUp/PgDn (a page), Home/End (first/last row), F = follow the running tracks again; holding a key repeats it (OS key repeat);
the mouse wheel works in terminals that map it to arrow keys on the alternate screen (Windows Terminal, macOS Terminal, iTerm2).
When the run is over the alternate screen is left and the full final table is printed into the normal scrollback.
Not a TTY (pipe, log file) or --plain: the caller prints plain log lines instead.

Row dict keys: n, file, status (queued|waiting|running|done|failed|skipped|timeout), t_start, secs, flag, note, bpm, nsec.
Optional keys for the combined grid + cues run: stage ("grid"|"cues", shown while status is running; default "analyzing") and cues (cue count, or "-" when no cues
were written for the song; as soon as one row has this key the table gets a "Cues" column).
"""
import os
import shutil
import sys
import time

NAME_MAX = 40
try:
    "\u2713\u2026".encode(sys.stdout.encoding or "ascii")
    CHECK, ELL = "\u2713", "\u2026"
except Exception:
    CHECK, ELL = "OK", "~"

HEADERS = ("#", "Status", "File", "Flag", "Review note", "BPM", "Sections")
HEADERS_WITH_CUES = HEADERS + ("Cues",)


def with_cues(rows):
    return any("cues" in row for row in rows)


def fit(text, width):
    text = str(text)
    return text if len(text) <= width else (text[: max(0, width - 1)] + ELL if width > 0 else "")


def status_text(row, now):
    status = row["status"]
    if status == "running":
        return f"{row.get('stage', 'analyzing')} {int(now - row['t_start'])}s"
    if status == "done":
        return f"{CHECK} ({row['secs']:.0f}s)"
    return {"queued": "queued", "waiting": "waiting for RB", "failed": "FAILED", "skipped": "skipped", "timeout": "TIMEOUT"}[status]


def widths(rows, total_width):
    number_width = max(len(str(len(rows))), 1)
    status_width = max(15, max((len(status_text(row, time.time())) for row in rows), default=0))
    file_width = min(NAME_MAX, max([len(row["file"]) for row in rows] + [4]))
    flag_width, bpm_width, sections_width = 6, 7, 8
    cues_width = 5 if with_cues(rows) else 0
    separator_width = 3 * (7 if cues_width else 6) + 2  # " | " between the columns plus leading/trailing margin
    sections_width_total = sections_width + cues_width
    note_width = total_width - (number_width + status_width + file_width + flag_width + bpm_width + sections_width_total + separator_width)
    if note_width < 20:  # narrow terminal: shrink the file column first
        file_width = max(12, file_width - (20 - note_width))
        note_width = max(10, total_width - (number_width + status_width + file_width + flag_width + bpm_width + sections_width_total + separator_width))
    return [number_width, status_width, file_width, flag_width, note_width, bpm_width, sections_width] + ([cues_width] if cues_width else [])


def row_line(cells, column_widths):
    return " " + " | ".join(fit(cell, width).ljust(width) if column_index not in (0, 5, 6, 7) else fit(cell, width).rjust(width) for column_index, (cell, width) in enumerate(zip(cells, column_widths)))


def cells_of(row, now):
    return [row["n"], status_text(row, now), row["file"], row.get("flag", ""), row.get("note", ""),
            f"{row['bpm']:.2f}" if row.get("bpm") else "", row.get("nsec") or ""] + ([row.get("cues", "")] if "cues" in row else [])


def table_lines(rows, total_width, now):
    column_widths = widths(rows, total_width)
    header_line = row_line(HEADERS_WITH_CUES if with_cues(rows) else HEADERS, column_widths)
    separator_line = "-" + "-+-".join("-" * x for x in column_widths) + "-"
    return header_line, separator_line, [row_line(cells_of(row, now), column_widths) for row in rows], column_widths


def eta_text(rows, jobs, now):
    done = [row for row in rows if row["status"] == "done"]
    left = [row for row in rows if row["status"] in ("queued", "waiting", "running")]
    if not left:
        return "0 min"
    if not done:
        return "calculating..."
    average_seconds = sum(row["secs"] for row in done) / len(done)
    remaining_seconds = sum(max(0.0, average_seconds - (now - row["t_start"])) if row["status"] == "running" else average_seconds for row in left)
    minutes_left = remaining_seconds / max(1, min(jobs, len(left))) / 60
    return "< 1 min" if minutes_left < 1 else f"~{minutes_left:.0f} min"


def footer_lines(rows, jobs, start_time, now):
    done_count = sum(1 for row in rows if row["status"] == "done")
    failed_count = sum(1 for row in rows if row["status"] in ("failed", "timeout"))
    running_count = sum(1 for row in rows if row["status"] == "running")
    review_count = sum(1 for row in rows if row.get("flag"))
    finished_count = done_count + failed_count + sum(1 for row in rows if row["status"] == "skipped")
    elapsed_seconds = int(now - start_time)
    return [f"Done {done_count}/{len(rows)} ({100 * done_count // max(1, len(rows))}%)   running {running_count}   failed {failed_count}   review {review_count}   "
            f"elapsed {elapsed_seconds // 60}:{elapsed_seconds % 60:02d}   estimated time left: {eta_text(rows, jobs, now)}"]


class Live:
    def __init__(self, rows, header, jobs, start_time):
        self.rows, self.header, self.jobs, self.start_time = rows, header, jobs, start_time
        self.top, self.follow, self.on = 0, True, False
        self._old = None

    def start(self):
        if os.name == "nt":
            os.system("")  # enables VT escape handling in the Windows 10 console
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass
        else:
            import termios
            import tty
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        sys.stdout.write("\x1b[?1049h\x1b[?1007h\x1b[?25l")
        sys.stdout.flush()
        self.on = True

    def stop(self):
        if not self.on:
            return
        sys.stdout.write("\x1b[?1007l\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()
        if self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        self.on = False

    def keys(self):
        pressed_keys = []
        if os.name == "nt":
            import msvcrt
            while msvcrt.kbhit():
                character = msvcrt.getwch()
                if character in ("\xe0", "\x00"):
                    pressed_keys.append({"H": "up", "P": "down", "I": "pgup", "Q": "pgdn", "G": "home", "O": "end"}.get(msvcrt.getwch()))
                else:
                    pressed_keys.append("follow" if character in "fF" else None)
        else:
            import select
            while select.select([sys.stdin], [], [], 0)[0]:
                text = os.read(self._fd, 32).decode("utf-8", "ignore")
                if not text:
                    break
                index = 0
                while index < len(text):
                    if text[index] == "\x1b" and text[index + 1:index + 2] in ("[", "O"):
                        sequence_end = index + 2
                        while sequence_end < len(text) and not (text[sequence_end].isalpha() or text[sequence_end] == "~"):
                            sequence_end += 1
                        escape_sequence = text[index + 2:sequence_end + 1]
                        pressed_keys.append({"A": "up", "B": "down", "H": "home", "F": "end", "5~": "pgup", "6~": "pgdn", "1~": "home", "4~": "end", "7~": "home", "8~": "end"}.get(escape_sequence))
                        index = sequence_end + 1
                    else:
                        pressed_keys.append("follow" if text[index] in "fF" else None)
                        index += 1
        return [key for key in pressed_keys if key]

    def handle(self, page_size):
        for key in self.keys():
            if key == "follow":
                self.follow = True
                continue
            self.follow = False
            self.top += {"up": -1, "down": 1, "pgup": -page_size, "pgdn": page_size, "home": -10 ** 9, "end": 10 ** 9}.get(key, 0)

    def render(self):
        now = time.time()
        columns, terminal_lines = shutil.get_terminal_size((100, 30))
        line_width = columns - 1
        header_line, separator_line, body_lines, _ = table_lines(self.rows, line_width, now)
        footer = footer_lines(self.rows, self.jobs() if callable(self.jobs) else self.jobs, self.start_time, now)
        hint = "Up/Down: scroll   PgUp/PgDn: page   Home/End: first/last row   F: follow progress   Ctrl+C: stop (finished tracks are kept)"
        header_lines = [line() if callable(line) else line for line in self.header]   # a header line may be a function, so it shows live values (e.g. the worker count)
        fixed_line_count = len(header_lines) + 2 + len(footer) + 2
        page_size = max(3, terminal_lines - fixed_line_count)
        self.handle(page_size)
        if self.follow:
            active_indexes = [row_index for row_index, row in enumerate(self.rows) if row["status"] in ("running", "waiting", "queued")]
            first_active_index = active_indexes[0] if active_indexes else max(0, len(self.rows) - page_size)
            self.top = max(0, first_active_index - page_size // 3)
        self.top = max(0, min(self.top, max(0, len(self.rows) - page_size)))
        visible_lines = body_lines[self.top:self.top + page_size]
        position_text = f" rows {self.top + 1}-{self.top + len(visible_lines)} of {len(self.rows)}" + ("" if self.follow else " (scrolled, F = follow)")
        screen_lines = header_lines + [header_line, separator_line] + visible_lines + [" " * 0] * (page_size - len(visible_lines)) + ["-" * min(line_width, len(separator_line))] + footer + [hint[:line_width], position_text]
        sys.stdout.write("\x1b[H" + "".join(fit(line, line_width) + "\x1b[K\r\n" for line in screen_lines[:terminal_lines - 1]) + "\x1b[J")
        sys.stdout.flush()


def final_table(rows, width=None):
    """Plain full table for the normal scrollback (notes are not cut, long ones wrap under the row)."""
    width = width or shutil.get_terminal_size((110, 30)).columns - 1
    now = time.time()
    header_line, separator_line, _, column_widths = table_lines(rows, width, now)
    out = [header_line, separator_line]
    import textwrap
    padding = " " * (1 + sum(column_widths[:4]) + 3 * 4)
    for row in rows:
        cells = cells_of(row, now)
        if len(cells) < len(column_widths):
            cells.append("")
        note_lines = textwrap.wrap(cells[4], column_widths[4]) or [""]
        cells[4] = note_lines[0]
        out.append(row_line(cells, column_widths))
        out += [padding + p for p in note_lines[1:]]
    return "\n".join(out)
