from __future__ import annotations

import math
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from services.pipeline_app import PipelineConfig, create_app
from services.storage import S3Storage


def write_tone(path: Path, *, seconds: float = 1.0, sample_rate: int = 16_000, amplitude: int = 12_000) -> None:
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for index in range(frames):
            value = int(amplitude * math.sin(2 * math.pi * 440 * index / sample_rate))
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
    assert data["totals"]["artifacts"] == 0
    assert data["jobs"]["queued"] == 0
    assert data["jobs"]["running"] == 0
    assert data["jobs"]["complete"] == 0
    assert data["queues"]["sound_gate"]["pending"] == 0
    assert data["purposes"]["describe_scene"]["pending"] == 0
    assert data["stages"]["sound_gate"] == 0
    assert data["storage_backend"] == "local"
    assert data["performance"]["overall"]["completed_tasks"] == 0
    assert data["performance"]["queues"]["sam_audio"]["audio_seconds_per_minute"] == 0
    assert data["performance"]["purposes"]["separate_music"]["avg_task_seconds"] == 0


def test_dashboard_graph_collapses_empty_paths_into_one_skipped_block(tmp_path: Path) -> None:
    client, _ = make_app(tmp_path)

    with client:
        response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert html.count('data-node="skipped"') == 1
    assert 'data-node="skipped_silent"' not in html
    assert 'data-node="skipped_music"' not in html
    assert 'data-node="skipped_sfx_voice"' not in html
    assert "empty audio" in html
    assert html.count(">empty</text>") == 5
    assert "music track" in html
    assert 'data-node="describe_music"' in html
    assert 'data-node="separate_voices"' in html
    assert 'data-node="gate_voice"' in html
    assert 'data-node="gate_sfx"' in html
    assert 'data-node="transcribe_voice"' in html
    assert 'data-node="sfx_ready"' in html
    assert 'data-node="list_sfx"' in html
    assert 'data-node="separate_sfx"' in html
    assert 'data-node="gate_remaining_sfx"' in html
    assert "music description" in html
    assert "voice transcription" in html
    assert "list sound" in html
    assert "remaining sfx" in html


def test_mock_pipeline_e2e_creates_scene_and_music_split_outputs(tmp_path: Path) -> None:
    client, runtime = make_app(tmp_path)
    audio_path = tmp_path / "workspace" / "tone.wav"
    write_tone(audio_path, seconds=1.2)

    with client:
        response = client.post("/api/jobs", json={"audio_path": str(audio_path)})
        assert response.status_code == 200
        job_id = response.json()["job"]["id"]
        queued_dashboard = client.get("/api/dashboard").json()

        processed = runtime.process_until_idle(max_tasks=100)
        assert processed >= 67

        detail = client.get(f"/api/jobs/{job_id}").json()
        dashboard = client.get("/api/dashboard").json()

    assert queued_dashboard["jobs"]["queued"] == 1
    assert queued_dashboard["purposes"]["describe_scene"]["pending"] == 1
    assert detail["job"]["status"] == "complete"
    assert {chunk["stage"] for chunk in detail["chunks"]} == {"sfx_iteration_limit"}

    artifacts = detail["artifacts"]
    scene = [artifact for artifact in artifacts if artifact["kind"] == "scene_description"]
    music_tracks = [artifact for artifact in artifacts if artifact["kind"] == "music_track"]
    sfx_voice_tracks = [artifact for artifact in artifacts if artifact["kind"] == "sfx_voice_track"]
    music_descriptions = [artifact for artifact in artifacts if artifact["kind"] == "music_description"]
    voice_tracks = [artifact for artifact in artifacts if artifact["kind"] == "voice_track"]
    sfx_tracks = [artifact for artifact in artifacts if artifact["kind"] == "sfx_track"]
    voice_transcriptions = [artifact for artifact in artifacts if artifact["kind"] == "voice_transcription"]
    sfx_lists = [artifact for artifact in artifacts if artifact["kind"] == "sfx_list"]
    sfx_isolated_tracks = [artifact for artifact in artifacts if artifact["kind"] == "sfx_isolated_track"]
    sfx_remaining_tracks = [artifact for artifact in artifacts if artifact["kind"] == "sfx_remaining_track"]
    sfx_loop_debug = [artifact for artifact in artifacts if artifact["kind"] == "sfx_loop_debug"]
    branch_gates = [
        artifact
        for artifact in artifacts
        if artifact["kind"] == "sound_gate"
        and artifact["prompt"] in {"gate_music", "gate_sfx_voice", "gate_voice", "gate_sfx", "gate_remaining_sfx"}
    ]
    assert len(scene) == 1
    assert len(music_tracks) == len(detail["chunks"])
    assert len(sfx_voice_tracks) == len(detail["chunks"])
    assert len(music_descriptions) == len(detail["chunks"])
    assert len(voice_tracks) == len(detail["chunks"])
    assert len(sfx_tracks) == len(detail["chunks"])
    assert len(voice_transcriptions) == len(detail["chunks"])
    assert len(sfx_lists) == len(detail["chunks"]) * 8
    assert len(sfx_isolated_tracks) == len(detail["chunks"]) * 8
    assert len(sfx_remaining_tracks) == len(detail["chunks"]) * 8
    assert len(sfx_loop_debug) == len(detail["chunks"])
    assert len(branch_gates) == len(detail["chunks"]) * 12
    assert all(
        Path(artifact["path"]).is_file()
        for artifact in music_tracks + sfx_voice_tracks + voice_tracks + sfx_tracks + sfx_isolated_tracks + sfx_remaining_tracks
    )
    assert all(
        artifact["storage_ref"]["backend"] == "local"
        and artifact["path_ref"] == artifact["storage_ref"]
        and artifact["path_ref"]["url"].startswith("/files/")
        for artifact in music_tracks + sfx_voice_tracks + voice_tracks + sfx_tracks + sfx_isolated_tracks + sfx_remaining_tracks
    )
    assert not any(task["payload"].get("purpose") == "describe_sfx" for task in detail["tasks"])
    assert not detail["stems"]

    assert dashboard["tasks"]["completed"] >= 67
    assert dashboard["jobs"]["complete"] == 1
    assert dashboard["totals"]["artifacts"] == len(detail["artifacts"])
    assert dashboard["purposes"]["separate_music"]["completed"] == len(detail["chunks"])
    assert dashboard["purposes"]["gate_music"]["completed"] == len(detail["chunks"])
    assert dashboard["purposes"]["gate_sfx_voice"]["completed"] == len(detail["chunks"])
    assert dashboard["purposes"]["describe_music"]["completed"] == len(detail["chunks"])
    assert dashboard["purposes"]["separate_voices"]["completed"] == len(detail["chunks"])
    assert dashboard["purposes"]["gate_voice"]["completed"] == len(detail["chunks"])
    assert dashboard["purposes"]["gate_sfx"]["completed"] == len(detail["chunks"])
    assert dashboard["purposes"]["transcribe_voice"]["completed"] == len(detail["chunks"])
    assert dashboard["purposes"]["list_sfx"]["completed"] == len(detail["chunks"]) * 8
    assert dashboard["purposes"]["separate_sfx"]["completed"] == len(detail["chunks"]) * 8
    assert dashboard["purposes"]["gate_remaining_sfx"]["completed"] == len(detail["chunks"]) * 8
    assert dashboard["performance"]["overall"]["completed_tasks"] >= 67
    assert dashboard["performance"]["overall"]["audio_seconds_per_minute"] > 0
    assert dashboard["performance"]["queues"]["sam_audio"]["completed_tasks"] == len(detail["chunks"]) * 10
    assert dashboard["performance"]["queues"]["sam_audio"]["audio_seconds_per_minute"] > 0
    assert dashboard["performance"]["queues"]["audio_flamingo"]["completed_tasks"] == 1 + len(detail["chunks"]) * 10
    assert dashboard["performance"]["queues"]["audio_flamingo"]["audio_seconds_per_minute"] > 0
    assert dashboard["performance"]["purposes"]["separate_music"]["completed_tasks"] == len(detail["chunks"])
    assert dashboard["performance"]["purposes"]["separate_music"]["avg_audio_seconds"] > 0
    assert dashboard["performance"]["purposes"]["separate_music"]["avg_task_seconds"] > 0
    assert dashboard["performance"]["purposes"]["describe_music"]["completed_tasks"] == len(detail["chunks"])
    assert dashboard["recent_outputs"][0]["kind"] in {
        "sound_gate",
        "music_track",
        "sfx_voice_track",
        "scene_description",
        "music_description",
        "voice_track",
        "sfx_track",
        "voice_transcription",
        "sfx_list",
        "sfx_isolated_track",
        "sfx_remaining_track",
        "sfx_loop_debug",
    }


def test_uploaded_audio_job_is_persisted_and_processed(tmp_path: Path) -> None:
    client, runtime = make_app(tmp_path)
    audio_path = tmp_path / "tone_upload.wav"
    write_tone(audio_path, seconds=1.0)

    with client:
        with audio_path.open("rb") as audio_file:
            response = client.post(
                "/api/jobs/upload",
                data={"prompt": "List sounds and choose one target."},
                files={"audio_file": ("tone_upload.wav", audio_file, "audio/wav")},
            )
        assert response.status_code == 200
        job_id = response.json()["job"]["id"]

        processed = runtime.process_until_idle(max_tasks=50)
        assert processed >= 34

        detail = client.get(f"/api/jobs/{job_id}").json()

    source_path = Path(detail["job"]["source_audio_path"])
    assert source_path.is_file()
    assert source_path.parent.name == "uploads"
    assert detail["job"]["status"] == "complete"
    assert any(artifact["kind"] == "scene_description" for artifact in detail["artifacts"])
    assert any(artifact["kind"] == "music_track" for artifact in detail["artifacts"])
    assert any(artifact["kind"] == "sfx_voice_track" for artifact in detail["artifacts"])
    assert any(artifact["kind"] == "music_description" for artifact in detail["artifacts"])
    assert any(artifact["kind"] == "voice_track" for artifact in detail["artifacts"])
    assert any(artifact["kind"] == "sfx_track" for artifact in detail["artifacts"])
    assert any(artifact["kind"] == "voice_transcription" for artifact in detail["artifacts"])
    assert any(artifact["kind"] == "sfx_isolated_track" for artifact in detail["artifacts"])
    assert any(artifact["kind"] == "sfx_remaining_track" for artifact in detail["artifacts"])


def test_sound_gate_skips_digital_silence(tmp_path: Path) -> None:
    client, runtime = make_app(tmp_path)
    audio_path = tmp_path / "workspace" / "silence.wav"
    write_tone(audio_path, seconds=1.0, amplitude=0)

    with client:
        response = client.post("/api/jobs", json={"audio_path": str(audio_path)})
        assert response.status_code == 200
        job_id = response.json()["job"]["id"]

        assert runtime.process_until_idle(max_tasks=20) == 2
        detail = client.get(f"/api/jobs/{job_id}").json()

    assert detail["job"]["status"] == "complete"
    assert detail["chunks"][0]["stage"] == "skipped_silent"
    gate_task = [task for task in detail["tasks"] if task["queue"] == "sound_gate"][0]
    assert len(detail["tasks"]) == 2
    assert gate_task["result"]["has_sound"] is False
    assert gate_task["result"]["dbfs"] is None
    assert any(artifact["kind"] == "scene_description" for artifact in detail["artifacts"])
    assert not any(artifact["kind"] == "music_track" for artifact in detail["artifacts"])
    assert not detail["stems"]


def test_sound_gate_skips_barely_hearable_audio(tmp_path: Path) -> None:
    client, runtime = make_app(tmp_path)
    audio_path = tmp_path / "workspace" / "quiet.wav"
    write_tone(audio_path, seconds=1.0, amplitude=8)

    with client:
        response = client.post("/api/jobs", json={"audio_path": str(audio_path)})
        assert response.status_code == 200
        job_id = response.json()["job"]["id"]

        assert runtime.process_until_idle(max_tasks=20) == 2
        detail = client.get(f"/api/jobs/{job_id}").json()

    result = [task for task in detail["tasks"] if task["queue"] == "sound_gate"][0]["result"]
    assert detail["chunks"][0]["stage"] == "skipped_silent"
    assert result["has_sound"] is False
    assert result["peak_dbfs"] < result["thresholds"]["min_peak_dbfs"]
    assert not any(artifact["kind"] == "music_track" for artifact in detail["artifacts"])
    assert not detail["stems"]


def test_branch_gate_skips_silent_sfx_voice_residual(tmp_path: Path) -> None:
    client, runtime = make_app(tmp_path)
    audio_path = tmp_path / "workspace" / "tone.wav"
    write_tone(audio_path, seconds=1.0)

    def mock_sam_with_silent_residual(audio_path: Path, prompt: str, output_prefix: str, job_id: str | None = None) -> dict:
        output_dir = runtime.config.output_dir / "pipeline" / "jobs" / (job_id or "mock") / "silent_residual"
        output_dir.mkdir(parents=True, exist_ok=True)
        target_wav = output_dir / f"{output_prefix}_target.wav"
        residual_wav = output_dir / f"{output_prefix}_residual.wav"
        write_tone(target_wav, seconds=1.0)
        write_tone(residual_wav, seconds=1.0, amplitude=0)
        return {
            "model_id": "mock/sam-audio-large",
            "audio_path": str(audio_path),
            "description": prompt,
            "target": {"wav": {"path": str(target_wav)}},
            "residual": {"wav": {"path": str(residual_wav)}},
        }

    runtime.mock_sam_audio = mock_sam_with_silent_residual

    with client:
        response = client.post("/api/jobs", json={"audio_path": str(audio_path)})
        assert response.status_code == 200
        job_id = response.json()["job"]["id"]

        assert runtime.process_until_idle(max_tasks=20) == 6
        detail = client.get(f"/api/jobs/{job_id}").json()

    assert detail["job"]["status"] == "complete"
    assert detail["chunks"][0]["stage"] == "music_described"
    residual_gate = [
        artifact
        for artifact in detail["artifacts"]
        if artifact["kind"] == "sound_gate" and artifact["prompt"] == "gate_sfx_voice"
    ][0]
    assert residual_gate["metadata"]["has_sound"] is False
    assert any(artifact["kind"] == "music_track" for artifact in detail["artifacts"])
    assert any(artifact["kind"] == "sfx_voice_track" for artifact in detail["artifacts"])
    assert any(artifact["kind"] == "music_description" for artifact in detail["artifacts"])
    assert not any(artifact["kind"] == "voice_track" for artifact in detail["artifacts"])
    assert not any(artifact["kind"] == "voice_transcription" for artifact in detail["artifacts"])


def test_voice_gate_skips_silent_voice_target(tmp_path: Path) -> None:
    client, runtime = make_app(tmp_path)
    audio_path = tmp_path / "workspace" / "tone.wav"
    write_tone(audio_path, seconds=1.0)

    def mock_sam_with_silent_voice(audio_path: Path, prompt: str, output_prefix: str, job_id: str | None = None) -> dict:
        output_dir = runtime.config.output_dir / "pipeline" / "jobs" / (job_id or "mock") / "silent_voice"
        output_dir.mkdir(parents=True, exist_ok=True)
        target_wav = output_dir / f"{output_prefix}_target.wav"
        residual_wav = output_dir / f"{output_prefix}_residual.wav"
        if prompt == "human voice":
            write_tone(target_wav, seconds=1.0, amplitude=0)
            write_tone(residual_wav, seconds=1.0)
        else:
            write_tone(target_wav, seconds=1.0)
            write_tone(residual_wav, seconds=1.0)
        return {
            "model_id": "mock/sam-audio-large",
            "audio_path": str(audio_path),
            "description": prompt,
            "target": {"wav": {"path": str(target_wav)}},
            "residual": {"wav": {"path": str(residual_wav)}},
        }

    runtime.mock_sam_audio = mock_sam_with_silent_voice

    with client:
        response = client.post("/api/jobs", json={"audio_path": str(audio_path)})
        assert response.status_code == 200
        job_id = response.json()["job"]["id"]

        assert runtime.process_until_idle(max_tasks=50) == 33
        detail = client.get(f"/api/jobs/{job_id}").json()

    assert detail["job"]["status"] == "complete"
    assert detail["chunks"][0]["stage"] == "sfx_iteration_limit"
    voice_gate = [
        artifact
        for artifact in detail["artifacts"]
        if artifact["kind"] == "sound_gate" and artifact["prompt"] == "gate_voice"
    ][0]
    assert voice_gate["metadata"]["has_sound"] is False
    assert any(artifact["kind"] == "voice_track" for artifact in detail["artifacts"])
    assert any(artifact["kind"] == "sfx_track" for artifact in detail["artifacts"])
    assert any(artifact["kind"] == "sfx_isolated_track" for artifact in detail["artifacts"])
    assert not any(artifact["kind"] == "voice_transcription" for artifact in detail["artifacts"])
    assert not any(task["payload"].get("purpose") == "describe_sfx" for task in detail["tasks"])


def test_sfx_gate_skips_silent_sfx_residual(tmp_path: Path) -> None:
    client, runtime = make_app(tmp_path)
    audio_path = tmp_path / "workspace" / "tone.wav"
    write_tone(audio_path, seconds=1.0)

    def mock_sam_with_silent_sfx(audio_path: Path, prompt: str, output_prefix: str, job_id: str | None = None) -> dict:
        output_dir = runtime.config.output_dir / "pipeline" / "jobs" / (job_id or "mock") / "silent_sfx"
        output_dir.mkdir(parents=True, exist_ok=True)
        target_wav = output_dir / f"{output_prefix}_target.wav"
        residual_wav = output_dir / f"{output_prefix}_residual.wav"
        write_tone(target_wav, seconds=1.0)
        if prompt == "human voice":
            write_tone(residual_wav, seconds=1.0, amplitude=0)
        else:
            write_tone(residual_wav, seconds=1.0)
        return {
            "model_id": "mock/sam-audio-large",
            "audio_path": str(audio_path),
            "description": prompt,
            "target": {"wav": {"path": str(target_wav)}},
            "residual": {"wav": {"path": str(residual_wav)}},
        }

    runtime.mock_sam_audio = mock_sam_with_silent_sfx

    with client:
        response = client.post("/api/jobs", json={"audio_path": str(audio_path)})
        assert response.status_code == 200
        job_id = response.json()["job"]["id"]

        assert runtime.process_until_idle(max_tasks=20) == 10
        detail = client.get(f"/api/jobs/{job_id}").json()

    assert detail["job"]["status"] == "complete"
    assert detail["chunks"][0]["stage"] == "voice_transcribed"
    sfx_gate = [
        artifact
        for artifact in detail["artifacts"]
        if artifact["kind"] == "sound_gate" and artifact["prompt"] == "gate_sfx"
    ][0]
    assert sfx_gate["metadata"]["has_sound"] is False
    assert any(artifact["kind"] == "voice_track" for artifact in detail["artifacts"])
    assert any(artifact["kind"] == "sfx_track" for artifact in detail["artifacts"])
    assert any(artifact["kind"] == "voice_transcription" for artifact in detail["artifacts"])
    assert not any(task["payload"].get("purpose") == "describe_sfx" for task in detail["tasks"])


def test_sfx_loop_exhausts_when_remaining_residual_is_silent(tmp_path: Path) -> None:
    client, runtime = make_app(tmp_path)
    audio_path = tmp_path / "workspace" / "tone.wav"
    write_tone(audio_path, seconds=1.0)

    def mock_sam_with_silent_remaining(audio_path: Path, prompt: str, output_prefix: str, job_id: str | None = None) -> dict:
        output_dir = runtime.config.output_dir / "pipeline" / "jobs" / (job_id or "mock") / "silent_remaining"
        output_dir.mkdir(parents=True, exist_ok=True)
        target_wav = output_dir / f"{output_prefix}_target.wav"
        residual_wav = output_dir / f"{output_prefix}_residual.wav"
        write_tone(target_wav, seconds=1.0)
        write_tone(residual_wav, seconds=1.0, amplitude=0 if prompt == "horse hooves" else 12_000)
        return {
            "model_id": "mock/sam-audio-large",
            "audio_path": str(audio_path),
            "description": prompt,
            "target": {"wav": {"path": str(target_wav)}},
            "residual": {"wav": {"path": str(residual_wav)}},
        }

    runtime.mock_sam_audio = mock_sam_with_silent_remaining

    with client:
        response = client.post("/api/jobs", json={"audio_path": str(audio_path)})
        assert response.status_code == 200
        job_id = response.json()["job"]["id"]

        assert runtime.process_until_idle(max_tasks=30) == 13
        detail = client.get(f"/api/jobs/{job_id}").json()

    assert detail["job"]["status"] == "complete"
    assert detail["chunks"][0]["stage"] == "sfx_exhausted"
    assert len([task for task in detail["tasks"] if task["payload"].get("purpose") == "list_sfx"]) == 1
    assert len([task for task in detail["tasks"] if task["payload"].get("purpose") == "separate_sfx"]) == 1
    assert len([task for task in detail["tasks"] if task["payload"].get("purpose") == "gate_remaining_sfx"]) == 1
    assert any(artifact["kind"] == "sfx_isolated_track" for artifact in detail["artifacts"])
    assert any(artifact["kind"] == "sfx_remaining_track" for artifact in detail["artifacts"])
    assert any(
        artifact["kind"] == "sfx_loop_debug" and artifact["metadata"]["reason"] == "residual_empty"
        for artifact in detail["artifacts"]
    )


def test_sfx_loop_exhausts_on_invalid_sfx_json(tmp_path: Path) -> None:
    client, runtime = make_app(tmp_path)
    audio_path = tmp_path / "workspace" / "tone.wav"
    write_tone(audio_path, seconds=1.0)
    original_mock_audio_flamingo = runtime.mock_audio_flamingo

    def mock_audio_flamingo_invalid_json(audio_path: Path, prompt: str) -> dict:
        if "strict json" in prompt.lower() and "effects" in prompt.lower():
            return {"model_id": "mock/audio-flamingo-next", "audio_path": str(audio_path), "prompt": prompt, "text": "not json"}
        return original_mock_audio_flamingo(audio_path, prompt)

    runtime.mock_audio_flamingo = mock_audio_flamingo_invalid_json

    with client:
        response = client.post("/api/jobs", json={"audio_path": str(audio_path)})
        assert response.status_code == 200
        job_id = response.json()["job"]["id"]

        assert runtime.process_until_idle(max_tasks=30) == 11
        detail = client.get(f"/api/jobs/{job_id}").json()

    assert detail["job"]["status"] == "complete"
    assert detail["chunks"][0]["stage"] == "sfx_exhausted"
    assert any(artifact["kind"] == "sfx_list" and artifact["metadata"]["parse_error"] for artifact in detail["artifacts"])
    assert any(
        artifact["kind"] == "sfx_loop_debug" and artifact["metadata"]["reason"] == "parse_error"
        for artifact in detail["artifacts"]
    )
    assert not any(task["payload"].get("purpose") == "separate_sfx" for task in detail["tasks"])


def test_sfx_loop_exhausts_on_empty_sfx_effects(tmp_path: Path) -> None:
    client, runtime = make_app(tmp_path)
    audio_path = tmp_path / "workspace" / "tone.wav"
    write_tone(audio_path, seconds=1.0)
    original_mock_audio_flamingo = runtime.mock_audio_flamingo

    def mock_audio_flamingo_empty_effects(audio_path: Path, prompt: str) -> dict:
        if "strict json" in prompt.lower() and "effects" in prompt.lower():
            return {"model_id": "mock/audio-flamingo-next", "audio_path": str(audio_path), "prompt": prompt, "text": '{"effects":[]}'}
        return original_mock_audio_flamingo(audio_path, prompt)

    runtime.mock_audio_flamingo = mock_audio_flamingo_empty_effects

    with client:
        response = client.post("/api/jobs", json={"audio_path": str(audio_path)})
        assert response.status_code == 200
        job_id = response.json()["job"]["id"]

        assert runtime.process_until_idle(max_tasks=30) == 11
        detail = client.get(f"/api/jobs/{job_id}").json()

    assert detail["job"]["status"] == "complete"
    assert detail["chunks"][0]["stage"] == "sfx_exhausted"
    assert any(artifact["kind"] == "sfx_list" and artifact["metadata"]["parsed"] == {"effects": []} for artifact in detail["artifacts"])
    assert any(
        artifact["kind"] == "sfx_loop_debug" and artifact["metadata"]["reason"] == "empty_effects"
        for artifact in detail["artifacts"]
    )
    assert not any(task["payload"].get("purpose") == "separate_sfx" for task in detail["tasks"])


def test_failed_task_can_be_retried(tmp_path: Path) -> None:
    client, runtime = make_app(tmp_path, backend="real")
    audio_path = tmp_path / "workspace" / "tone.wav"
    write_tone(audio_path, seconds=1.0)

    with client:
        response = client.post("/api/jobs", json={"audio_path": str(audio_path)})
        assert response.status_code == 200
        job_id = response.json()["job"]["id"]

        assert runtime.process_pending_once() is True

        detail = client.get(f"/api/jobs/{job_id}").json()
        failed_tasks = [task for task in detail["tasks"] if task["status"] == "failed"]
        assert failed_tasks
        assert failed_tasks[0]["payload"]["purpose"] == "describe_scene"
        failed_id = failed_tasks[0]["id"]

        retry = client.post(f"/api/tasks/{failed_id}/retry")
        assert retry.status_code == 200
        detail = client.get(f"/api/jobs/{job_id}").json()

    retried = [task for task in detail["tasks"] if task["id"] == failed_id][0]
    assert retried["status"] == "pending"
    assert detail["job"]["status"] == "queued"


def test_s3_storage_adapter_uploads_artifact_and_returns_s3_ref(tmp_path: Path) -> None:
    artifact_file = tmp_path / "target.wav"
    write_tone(artifact_file, seconds=0.1)

    class FakeS3Client:
        def __init__(self) -> None:
            self.uploads = []

        def upload_file(self, filename: str, bucket: str, key: str, ExtraArgs: dict | None = None) -> None:
            self.uploads.append({"filename": filename, "bucket": bucket, "key": key, "extra_args": ExtraArgs})

    fake_client = FakeS3Client()
    storage = S3Storage(
        bucket="test-bucket",
        prefix="pipeline-artifacts",
        public_base_url="https://cdn.example.test",
        client=fake_client,
    )

    ref = storage.store_artifact_file(
        artifact_file,
        artifact_id="artifact123",
        job_id="job123",
        chunk_id="chunk123",
        kind="music_track",
    )

    assert len(fake_client.uploads) == 1
    upload = fake_client.uploads[0]
    assert upload["filename"] == str(artifact_file.resolve())
    assert upload["bucket"] == "test-bucket"
    assert upload["key"].startswith("pipeline-artifacts/job123/chunk123/music_track/artifact123_target.wav")
    assert upload["extra_args"] == {"ContentType": "audio/x-wav"}
    assert ref["backend"] == "s3"
    assert ref["bucket"] == "test-bucket"
    assert ref["uri"] == f"s3://test-bucket/{upload['key']}"
    assert ref["path"] == ref["uri"]
    assert ref["local_path"] == str(artifact_file.resolve())
    assert ref["url"] == f"https://cdn.example.test/{upload['key']}"
