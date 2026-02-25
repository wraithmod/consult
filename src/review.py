#!/usr/bin/env python3
"""
review.py — terminal review UI for GP SOAP note approval.

Usage:
    python3 src/review.py --note notes/20250225_143022_soap.md
    python3 src/review.py --note notes/20250225_143022_soap.md --transcript transcripts/20250225_143022.txt
"""

import argparse
import curses
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Pane:
    title: str
    path: Path | None
    text: str
    scroll: int = 0
    _wrap_width: int = 0
    _wrapped_lines: list[str] | None = None

    def wrapped_lines(self, width: int) -> list[str]:
        width = max(1, width)
        if self._wrapped_lines is not None and self._wrap_width == width:
            return self._wrapped_lines

        wrapped: list[str] = []
        raw_lines = self.text.splitlines()
        if not raw_lines:
            raw_lines = [""]

        for raw_line in raw_lines:
            expanded = raw_line.expandtabs(4)
            parts = textwrap.wrap(
                expanded,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            )
            if not parts:
                wrapped.append("")
            else:
                wrapped.extend(part.rstrip() for part in parts)

        self._wrap_width = width
        self._wrapped_lines = wrapped
        return wrapped

    def set_text(self, text: str) -> None:
        self.text = text
        self._wrapped_lines = None
        self._wrap_width = 0
        self.scroll = 0

    def reload_from_disk(self) -> None:
        if self.path is None:
            return
        self.set_text(self.path.read_text(encoding="utf-8"))

    def scroll_by(self, delta: int, view_height: int, view_width: int) -> None:
        lines = self.wrapped_lines(view_width)
        max_scroll = max(0, len(lines) - max(1, view_height))
        self.scroll = max(0, min(max_scroll, self.scroll + delta))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review and approve a generated GP SOAP note in a terminal UI."
    )
    parser.add_argument(
        "--note",
        required=True,
        type=Path,
        help="Path to the generated SOAP note markdown file.",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Optional transcript text file for reference.",
    )
    return parser.parse_args()


def load_text_file(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path.read_text(encoding="utf-8")


def approved_sidecar_path(note_path: Path) -> Path:
    return note_path.with_suffix(".approved")


def write_approved_marker(note_path: Path) -> Path:
    sidecar = approved_sidecar_path(note_path)
    sidecar.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
    return sidecar


class ReviewUI:
    def __init__(self, note_pane: Pane, transcript_pane: Pane | None) -> None:
        self.note_pane = note_pane
        self.transcript_pane = transcript_pane
        self.active_pane = 0
        self.status_message = ""

    def panes(self) -> list[Pane]:
        if self.transcript_pane is None:
            return [self.note_pane]
        return [self.note_pane, self.transcript_pane]

    def active(self) -> Pane:
        return self.panes()[self.active_pane]

    def _switch_pane(self) -> None:
        if self.transcript_pane is not None:
            self.active_pane = 1 - self.active_pane

    def _open_editor(self, stdscr: curses.window) -> None:
        editor = os.environ.get("EDITOR", "nano")
        try:
            curses.def_prog_mode()
            curses.endwin()
            subprocess.run([editor, str(self.note_pane.path)], check=False)
        except FileNotFoundError:
            self.status_message = f"Editor not found: {editor}"
        finally:
            curses.reset_prog_mode()
            stdscr.refresh()
            try:
                curses.curs_set(0)
            except curses.error:
                pass

        try:
            self.note_pane.reload_from_disk()
            self.status_message = "Note reloaded after editing."
        except OSError as exc:
            self.status_message = f"Could not reload note: {exc}"

    def _draw_pane(
        self,
        stdscr: curses.window,
        pane: Pane,
        start_row: int,
        height: int,
        width: int,
        is_active: bool,
    ) -> None:
        if height <= 0:
            return

        title_attr = curses.A_BOLD
        if is_active:
            title_attr |= curses.A_REVERSE

        title = f" {pane.title} "
        try:
            stdscr.addnstr(start_row, 0, title.ljust(width), width, title_attr)
        except curses.error:
            return

        content_height = max(0, height - 1)
        lines = pane.wrapped_lines(max(1, width))
        max_scroll = max(0, len(lines) - max(1, content_height))
        if pane.scroll > max_scroll:
            pane.scroll = max_scroll

        for i in range(content_height):
            row = start_row + 1 + i
            line_index = pane.scroll + i
            line = lines[line_index] if line_index < len(lines) else ""
            try:
                stdscr.addnstr(row, 0, line.ljust(width), width)
            except curses.error:
                pass

        if content_height > 0 and len(lines) > content_height and width >= 12:
            position = pane.scroll + 1
            footer = f"{position}/{len(lines)}"
            try:
                stdscr.addnstr(start_row, max(0, width - len(footer) - 1), footer, len(footer))
            except curses.error:
                pass

    def _draw_status_bar(self, stdscr: curses.window, row: int, width: int) -> None:
        keys = "e Edit  a Approve  r Reject  q Quit  ↑/↓ or j/k Scroll"
        if self.transcript_pane is not None:
            keys += "  Tab Switch pane"
        text = keys
        if self.status_message:
            text = f"{keys} | {self.status_message}"
        try:
            stdscr.addnstr(row, 0, text.ljust(width), width, curses.A_REVERSE)
        except curses.error:
            pass

    def _draw(self, stdscr: curses.window) -> None:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 4 or width < 20:
            message = "Terminal too small. Resize or press q to quit."
            try:
                stdscr.addnstr(0, 0, message[:width], width)
                stdscr.addnstr(max(0, height - 1), 0, "q Quit".ljust(width), width, curses.A_REVERSE)
            except curses.error:
                pass
            stdscr.refresh()
            return

        status_row = height - 1
        body_height = height - 1

        if self.transcript_pane is None:
            note_height = body_height
            transcript_height = 0
        else:
            note_height = body_height // 2
            transcript_height = body_height - note_height
            if note_height < 2:
                note_height = max(1, body_height - 1)
                transcript_height = body_height - note_height

        self._draw_pane(
            stdscr,
            self.note_pane,
            start_row=0,
            height=note_height,
            width=width,
            is_active=(self.active_pane == 0),
        )
        if self.transcript_pane is not None and transcript_height > 0:
            self._draw_pane(
                stdscr,
                self.transcript_pane,
                start_row=note_height,
                height=transcript_height,
                width=width,
                is_active=(self.active_pane == 1),
            )

        self._draw_status_bar(stdscr, status_row, width)
        stdscr.refresh()

    def run(self, stdscr: curses.window) -> tuple[str, int]:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)

        while True:
            self._draw(stdscr)
            ch = stdscr.getch()

            if ch in (ord("q"), ord("Q")):
                return ("Quit", 2)
            if ch in (ord("a"), ord("A")):
                return ("Approved", 0)
            if ch in (ord("r"), ord("R")):
                return ("Rejected", 1)
            if ch in (ord("e"), ord("E")):
                self._open_editor(stdscr)
                continue
            if ch in (curses.KEY_UP, ord("k"), ord("K")):
                height, width = stdscr.getmaxyx()
                pane_height = self._active_content_height(height)
                self.active().scroll_by(-1, pane_height, width)
                continue
            if ch in (curses.KEY_DOWN, ord("j"), ord("J")):
                height, width = stdscr.getmaxyx()
                pane_height = self._active_content_height(height)
                self.active().scroll_by(1, pane_height, width)
                continue
            if ch == curses.KEY_PPAGE:
                height, width = stdscr.getmaxyx()
                pane_height = self._active_content_height(height)
                self.active().scroll_by(-max(1, pane_height - 1), pane_height, width)
                continue
            if ch == curses.KEY_NPAGE:
                height, width = stdscr.getmaxyx()
                pane_height = self._active_content_height(height)
                self.active().scroll_by(max(1, pane_height - 1), pane_height, width)
                continue
            if ch in (curses.KEY_LEFT, curses.KEY_RIGHT, ord("\t")):
                self._switch_pane()
                continue
            if ch == curses.KEY_RESIZE:
                continue

    def _active_content_height(self, total_height: int) -> int:
        body_height = max(1, total_height - 1)
        if self.transcript_pane is None:
            return max(1, body_height - 1)
        note_height = body_height // 2
        transcript_height = body_height - note_height
        if self.active_pane == 0:
            return max(1, note_height - 1)
        return max(1, transcript_height - 1)


def run_review_ui(note_path: Path, transcript_path: Path | None) -> tuple[str, int]:
    note_text = load_text_file(note_path, "Note")
    note_pane = Pane(title="SOAP note", path=note_path, text=note_text)

    transcript_pane = None
    if transcript_path is not None:
        transcript_text = load_text_file(transcript_path, "Transcript")
        transcript_pane = Pane(title="Transcript (reference)", path=transcript_path, text=transcript_text)

    ui = ReviewUI(note_pane, transcript_pane)
    return curses.wrapper(ui.run)


def main() -> None:
    args = parse_args()

    try:
        status, exit_code = run_review_ui(args.note, args.transcript)
        if status == "Approved":
            write_approved_marker(args.note)
        elif status == "Rejected":
            args.note.unlink()
        print(status)
        sys.exit(exit_code)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Quit")
        sys.exit(2)
    except curses.error as exc:
        print(f"Error: terminal UI failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
