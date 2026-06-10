from __future__ import annotations

import math
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from services.pipeline_app import PipelineConfig, create_app


def write_tone(path: Path, *, seconds: float = 1.0, sample_rate: int = 16_000) -> None:
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for index in range(frames):
            value = int(12_000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            wav.writeframesraw(value.to_bytes(2, "little", signed=True))


def make_app(tmp_path: Path, *, backend: str = "mock") -> tuple[TestClient, object]:
    workspace_dir = tmp_path / "workspace"
    output_dir = tmp_path / "outputs"
    config = PipelineConfig(
        workspace_dir=workspace_dir,
        output_dir=output_dir,
        db_path=tmp_path / "pipeline.sqlite3",
        backend=backend,
        worker_enabled=False,
        chunk_ms=1_000,
        overlap_ms=200,
        afnext_endpoint="http://127.0.0.1:9/v1/audio-flamingo/ask",
    )
    workspace_dir.mkdir(parents=True)
    app = create_app(config)
    return TestClient(app), app.state.pipeline_runtime


def test_dashboard_summary_initializes_schema(tmp_path: Path) -> None:
    client, _ = make_app(tmp_path)
    with client:
        response = client.get("/api/dashboard")

    assert response.status_code == 200
    data = response.json()
    assert data["totals"]["jobs"] == 0
    assert data["totals"]["chunks"] == 0
    assert data["totals"]["stems"] == 0
    assert data["queues"]["sound_gate"]["pending"] == 0
    assert data["stages"]["sound_gate"] == 0


def test_mock_pipeline_e2e_creates_target_and_residual_outputs(tmp_path: Path) -> None:
    client, runtime = make_app(tmp_path)
    audio_path = tmp_path / "workspace" / "tone.wav"
    write_tone(audio_path, seconds=1.2)

    with client:
        response = client.post("/api/jobs", json={"audio_path": str(audio_path)})
        assert response.status_code == 200
        job_id = response.json()["job"]["id"]

        processed = runtime.process_until_idle(max_tasks=20)
        assert processed >= 3

        detail = client.get(f"/api/jobs/{job_id}").json()
        dashboard = client.get("/api/dashboard").json()

    assert detail["job"]["status"] == "complete"
    assert detail["chunks"][0]["stage"] == "complete"
    assert detail["stems"]
    stem = detail["stems"][0]
    assert Path(stem["target"]["wav"]["path"]).is_file()
    assert Path(stem["residual"]["wav"]["path"]).is_file()
    assert Path(stem["zip"]["path"]).is_file()
    assert dashboard["tasks"]["completed"] >= 3
    assert dashboard["totals"]["stems"] == len(detail["stems"])
    assert dashboard["recent_outputs"][0]["prompt"] == "horse hooves"


def test_failed_task_can_be_retried(tmp_path: Path) -> None:
    client, runtime = make_app(tmp_path, backend="real")
    audio_path = tmp_path / "workspace" / "tone.wav"
    write_tone(audio_path, seconds=1.0)

    with client:
        response = client.post("/api/jobs", json={"audio_path": str(audio_path)})
        assert response.status_code == 200
        job_id = response.json()["job"]["id"]

        assert runtime.process_pending_once() is True
        assert runtime.process_pending_once() is True

        detail = client.get(f"/api/jobs/{job_id}").json()
        failed_tasks = [task for task in detail["tasks"] if task["status"] == "failed"]
        assert failed_tasks
        failed_id = failed_tasks[0]["id"]

        retry = client.post(f"/api/tasks/{failed_id}/retry")
        assert retry.status_code == 200
        detail = client.get(f"/api/jobs/{job_id}").json()

    retried = [task for task in detail["tasks"] if task["id"] == failed_id][0]
    assert retried["status"] == "pending"
    assert detail["job"]["status"] == "queued"
