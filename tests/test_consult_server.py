from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient


def _load_server_module():
    module_path = Path("consult-server/main.py").resolve()
    spec = importlib.util.spec_from_file_location("consult_server_main_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_root_serves_web_ui(monkeypatch):
    module = _load_server_module()
    monkeypatch.setattr(module.summarise, "warmup_ollama_model", lambda _m: None)

    with TestClient(module.app) as client:
        resp = client.get("/")

    assert resp.status_code == 200
    assert "Consult Web Recorder" in resp.text


def test_process_accepts_webm_upload(monkeypatch):
    module = _load_server_module()
    monkeypatch.setattr(module.summarise, "warmup_ollama_model", lambda _m: None)
    process_mock = mock.Mock(
        return_value={
            "status": "success",
            "audio_file": "x.webm",
            "transcript_file": "x.txt",
            "note_file": "x_soap.md",
            "note": "ok",
        }
    )
    monkeypatch.setattr(module, "_process_saved_audio", process_mock)

    with TestClient(module.app) as client:
        resp = client.post(
            "/process",
            files={"file": ("recording.webm", io.BytesIO(b"webm-bytes"), "audio/webm")},
        )

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    saved_path = process_mock.call_args[0][0]
    assert saved_path.suffix == ".webm"


def test_process_rejects_unsupported_extension(monkeypatch):
    module = _load_server_module()
    monkeypatch.setattr(module.summarise, "warmup_ollama_model", lambda _m: None)

    with TestClient(module.app) as client:
        resp = client.post(
            "/process",
            files={"file": ("bad.txt", io.BytesIO(b"bad"), "text/plain")},
        )

    assert resp.status_code == 400
    assert "Unsupported audio format" in resp.json()["detail"]
