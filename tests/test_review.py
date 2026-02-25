import argparse
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import pytest

from src import review


class FakeCursesError(Exception):
    pass


class FakeWindow:
    def __init__(self, keys, height=24, width=100):
        self._keys = list(keys)
        self._height = height
        self._width = width
        self.keypad_enabled = False
        self.refresh_count = 0

    def keypad(self, flag):
        self.keypad_enabled = bool(flag)

    def erase(self):
        return None

    def getmaxyx(self):
        return (self._height, self._width)

    def addnstr(self, *args, **kwargs):
        return None

    def refresh(self):
        self.refresh_count += 1

    def getch(self):
        if self._keys:
            return self._keys.pop(0)
        return ord("q")

def patch_curses(fake_window):
    stack = ExitStack()
    stack.enter_context(
        mock.patch.object(
            review.curses,
            "wrapper",
            side_effect=lambda func: func(fake_window),
            create=True,
        )
    )
    stack.enter_context(mock.patch.object(review.curses, "curs_set", return_value=None, create=True))
    stack.enter_context(mock.patch.object(review.curses, "def_prog_mode", return_value=None, create=True))
    stack.enter_context(mock.patch.object(review.curses, "endwin", return_value=None, create=True))
    stack.enter_context(mock.patch.object(review.curses, "reset_prog_mode", return_value=None, create=True))
    stack.enter_context(mock.patch.object(review.curses, "A_BOLD", 1, create=True))
    stack.enter_context(mock.patch.object(review.curses, "A_REVERSE", 2, create=True))
    stack.enter_context(mock.patch.object(review.curses, "KEY_UP", 259, create=True))
    stack.enter_context(mock.patch.object(review.curses, "KEY_DOWN", 258, create=True))
    stack.enter_context(mock.patch.object(review.curses, "KEY_LEFT", 260, create=True))
    stack.enter_context(mock.patch.object(review.curses, "KEY_RIGHT", 261, create=True))
    stack.enter_context(mock.patch.object(review.curses, "KEY_PPAGE", 339, create=True))
    stack.enter_context(mock.patch.object(review.curses, "KEY_NPAGE", 338, create=True))
    stack.enter_context(mock.patch.object(review.curses, "KEY_RESIZE", 410, create=True))
    stack.enter_context(mock.patch.object(review.curses, "error", FakeCursesError, create=True))
    return stack


def run_main_with_keys(tmp_path: Path, keys, transcript=False):
    note_path = tmp_path / "note.md"
    note_path.write_text("original note\n", encoding="utf-8")
    transcript_path = None
    if transcript:
        transcript_path = tmp_path / "transcript.txt"
        transcript_path.write_text("Doctor: test\nPatient: test\n", encoding="utf-8")

    fake_window = FakeWindow(keys)
    args = argparse.Namespace(note=note_path, transcript=transcript_path)

    with patch_curses(fake_window), mock.patch.object(review, "parse_args", return_value=args):
        with pytest.raises(SystemExit) as excinfo:
            review.main()

    return excinfo.value.code, note_path, transcript_path, fake_window


def test_approve_flow_creates_sidecar_and_exits_zero(tmp_path):
    exit_code, note_path, _, _ = run_main_with_keys(tmp_path, [ord("a")], transcript=True)

    assert exit_code == 0
    assert note_path.exists()
    sidecar = review.approved_sidecar_path(note_path)
    assert sidecar.exists()
    assert sidecar.read_text(encoding="utf-8").strip()


def test_reject_flow_deletes_note_and_exits_one(tmp_path):
    exit_code, note_path, _, _ = run_main_with_keys(tmp_path, [ord("r")], transcript=False)

    assert exit_code == 1
    assert not note_path.exists()
    assert not review.approved_sidecar_path(note_path).exists()


def test_quit_flow_keeps_files_unchanged_and_exits_two(tmp_path):
    original = "keep me\n"
    note_path = tmp_path / "note.md"
    note_path.write_text(original, encoding="utf-8")
    fake_window = FakeWindow([ord("q")])
    args = argparse.Namespace(note=note_path, transcript=None)

    with patch_curses(fake_window), mock.patch.object(review, "parse_args", return_value=args):
        with pytest.raises(SystemExit) as excinfo:
            review.main()

    assert excinfo.value.code == 2
    assert note_path.exists()
    assert note_path.read_text(encoding="utf-8") == original
    assert not review.approved_sidecar_path(note_path).exists()


def test_reload_after_edit_reloads_note_content(tmp_path):
    note_path = tmp_path / "note.md"
    note_path.write_text("before edit\n", encoding="utf-8")
    note_pane = review.Pane(title="SOAP note", path=note_path, text=note_path.read_text(encoding="utf-8"))
    ui = review.ReviewUI(note_pane=note_pane, transcript_pane=None)
    fake_window = FakeWindow([ord("e"), ord("q")])

    def fake_editor_run(cmd, check=False):
        note_path.write_text("after edit\n", encoding="utf-8")
        return mock.Mock(returncode=0)

    with patch_curses(fake_window), mock.patch.object(review.subprocess, "run", side_effect=fake_editor_run):
        status, exit_code = ui.run(fake_window)

    assert (status, exit_code) == ("Quit", 2)
    assert ui.note_pane.text == "after edit\n"
    assert ui.status_message == "Note reloaded after editing."


def test_scroll_keys_and_pane_switch_do_not_crash(tmp_path):
    note_path = tmp_path / "note.md"
    transcript_path = tmp_path / "transcript.txt"
    note_path.write_text("\n".join(f"Note line {i}" for i in range(80)), encoding="utf-8")
    transcript_path.write_text("\n".join(f"Transcript line {i}" for i in range(80)), encoding="utf-8")

    note_pane = review.Pane(title="SOAP note", path=note_path, text=note_path.read_text(encoding="utf-8"))
    transcript_pane = review.Pane(
        title="Transcript (reference)",
        path=transcript_path,
        text=transcript_path.read_text(encoding="utf-8"),
    )
    ui = review.ReviewUI(note_pane=note_pane, transcript_pane=transcript_pane)
    fake_window = FakeWindow(
        [
            review.curses.KEY_DOWN,
            review.curses.KEY_DOWN,
            review.curses.KEY_UP,
            review.curses.KEY_NPAGE,
            review.curses.KEY_PPAGE,
            review.curses.KEY_RIGHT,
            review.curses.KEY_DOWN,
            review.curses.KEY_LEFT,
            review.curses.KEY_RESIZE,
            ord("q"),
        ],
        height=18,
        width=60,
    )

    with patch_curses(fake_window):
        status, exit_code = ui.run(fake_window)

    assert (status, exit_code) == ("Quit", 2)
    assert fake_window.keypad_enabled is True
    assert ui.note_pane.scroll >= 0
    assert ui.transcript_pane.scroll >= 0
