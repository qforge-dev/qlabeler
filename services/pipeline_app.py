from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
import threading
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydub import AudioSegment

from services.common import OUTPUT_DIR, WORKSPACE_DIR, exception_detail, parse_bool, resolve_local_path
from services.storage import ArtifactStorage, create_storage_adapter, local_file_ref


TASK_SOUND_GATE = "sound_gate"
TASK_AUDIO_FLAMINGO = "audio_flamingo"
TASK_SAM_AUDIO = "sam_audio"
TASK_QUEUES = (TASK_SOUND_GATE, TASK_AUDIO_FLAMINGO, TASK_SAM_AUDIO)

STAGE_SOUND_GATE = "sound_gate"
STAGE_DESCRIBE_SCENE = "describe_scene"
STAGE_SEPARATE_MUSIC = "separate_music"
STAGE_GATE_MUSIC = "gate_music"
STAGE_GATE_SFX_VOICE = "gate_sfx_voice"
STAGE_MUSIC_READY = "music_ready"
STAGE_SFX_VOICE_READY = "sfx_voice_ready"
STAGE_DESCRIBE_MUSIC = "describe_music"
STAGE_MUSIC_DESCRIBED = "music_described"
STAGE_SEPARATE_VOICES = "separate_voices"
STAGE_GATE_VOICE = "gate_voice"
STAGE_GATE_SFX = "gate_sfx"
STAGE_TRANSCRIBE_VOICE = "transcribe_voice"
STAGE_VOICE_TRANSCRIBED = "voice_transcribed"
STAGE_SFX_READY = "sfx_ready"
STAGE_LIST_SFX = "list_sfx"
STAGE_GATE_REMAINING_SFX = "gate_remaining_sfx"
STAGE_SFX_EXHAUSTED = "sfx_exhausted"
STAGE_SFX_ITERATION_LIMIT = "sfx_iteration_limit"
STAGE_SFX_LOOP_FAILED = "sfx_loop_failed"
STAGE_SKIPPED_MUSIC = "skipped_music"
STAGE_SKIPPED_SFX_VOICE = "skipped_sfx_voice"
STAGE_SKIPPED_VOICE = "skipped_voice"
STAGE_SKIPPED_SFX = "skipped_sfx"
STAGE_DESCRIBE_SFX = "describe_sfx"
STAGE_SEPARATE_SFX = "separate_sfx"
STAGE_COMPLETE = "complete"
STAGE_SKIPPED_SILENT = "skipped_silent"
STAGE_FAILED = "failed"
STAGES = (
    STAGE_SOUND_GATE,
    STAGE_DESCRIBE_SCENE,
    STAGE_SEPARATE_MUSIC,
    STAGE_GATE_MUSIC,
    STAGE_GATE_SFX_VOICE,
    STAGE_MUSIC_READY,
    STAGE_SFX_VOICE_READY,
    STAGE_DESCRIBE_MUSIC,
    STAGE_MUSIC_DESCRIBED,
    STAGE_SEPARATE_VOICES,
    STAGE_GATE_VOICE,
    STAGE_GATE_SFX,
    STAGE_TRANSCRIBE_VOICE,
    STAGE_VOICE_TRANSCRIBED,
    STAGE_SFX_READY,
    STAGE_LIST_SFX,
    STAGE_GATE_REMAINING_SFX,
    STAGE_SFX_EXHAUSTED,
    STAGE_SFX_ITERATION_LIMIT,
    STAGE_SFX_LOOP_FAILED,
    STAGE_SKIPPED_MUSIC,
    STAGE_SKIPPED_SFX_VOICE,
    STAGE_SKIPPED_VOICE,
    STAGE_SKIPPED_SFX,
    STAGE_DESCRIBE_SFX,
    STAGE_SEPARATE_SFX,
    STAGE_COMPLETE,
    STAGE_SKIPPED_SILENT,
    STAGE_FAILED,
)

PURPOSE_CHUNK_SOUND_GATE = "chunk_sound_gate"
PURPOSE_DESCRIBE_SCENE = "describe_scene"
PURPOSE_SEPARATE_MUSIC = "separate_music"
PURPOSE_GATE_MUSIC = "gate_music"
PURPOSE_GATE_SFX_VOICE = "gate_sfx_voice"
PURPOSE_DESCRIBE_MUSIC = "describe_music"
PURPOSE_SEPARATE_VOICES = "separate_voices"
PURPOSE_GATE_VOICE = "gate_voice"
PURPOSE_GATE_SFX = "gate_sfx"
PURPOSE_TRANSCRIBE_VOICE = "transcribe_voice"
PURPOSE_LIST_SFX = "list_sfx"
PURPOSE_GATE_REMAINING_SFX = "gate_remaining_sfx"
PURPOSE_DESCRIBE_SFX = "describe_sfx"
PURPOSE_SEPARATE_SFX = "separate_sfx"
PURPOSES = (
    PURPOSE_DESCRIBE_SCENE,
    PURPOSE_CHUNK_SOUND_GATE,
    PURPOSE_SEPARATE_MUSIC,
    PURPOSE_GATE_MUSIC,
    PURPOSE_GATE_SFX_VOICE,
    PURPOSE_DESCRIBE_MUSIC,
    PURPOSE_SEPARATE_VOICES,
    PURPOSE_GATE_VOICE,
    PURPOSE_GATE_SFX,
    PURPOSE_TRANSCRIBE_VOICE,
    PURPOSE_LIST_SFX,
    PURPOSE_GATE_REMAINING_SFX,
    PURPOSE_DESCRIBE_SFX,
    PURPOSE_SEPARATE_SFX,
)

ARTIFACT_SCENE_DESCRIPTION = "scene_description"
ARTIFACT_MUSIC_TRACK = "music_track"
ARTIFACT_MUSIC_TRACK_RAW = "music_track_raw"
ARTIFACT_SFX_VOICE_TRACK = "sfx_voice_track"
ARTIFACT_MUSIC_DESCRIPTION = "music_description"
ARTIFACT_VOICE_TRACK = "voice_track"
ARTIFACT_VOICE_TRACK_RAW = "voice_track_raw"
ARTIFACT_SFX_TRACK = "sfx_track"
ARTIFACT_VOICE_TRANSCRIPTION = "voice_transcription"
ARTIFACT_SFX_LIST = "sfx_list"
ARTIFACT_SFX_ISOLATED_TRACK = "sfx_isolated_track"
ARTIFACT_SFX_REMAINING_TRACK = "sfx_remaining_track"
ARTIFACT_SFX_LOOP_DEBUG = "sfx_loop_debug"
ARTIFACT_SOUND_GATE = "sound_gate"
SOUND_AUDIO_ARTIFACT_KINDS = (
    ARTIFACT_MUSIC_TRACK,
    ARTIFACT_MUSIC_TRACK_RAW,
    ARTIFACT_SFX_VOICE_TRACK,
    ARTIFACT_VOICE_TRACK,
    ARTIFACT_VOICE_TRACK_RAW,
    ARTIFACT_SFX_TRACK,
    ARTIFACT_SFX_ISOLATED_TRACK,
    ARTIFACT_SFX_REMAINING_TRACK,
)

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

SFX_LOOP_MAX_ITERATIONS = 8


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_id() -> str:
    return uuid.uuid4().hex


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def json_loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def clean_prefix(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return cleaned[:96] or "pipeline"


def clean_upload_name(filename: str | None) -> str:
    source = Path(filename or "audio").name
    stem = clean_prefix(Path(source).stem or "audio")
    suffix = re.sub(r"[^A-Za-z0-9.]+", "", Path(source).suffix.lower())[:16]
    return f"{new_id()}_{stem}{suffix}"


def path_ref(path: str | None, output_dir: Path) -> dict[str, str] | None:
    return local_file_ref(path, output_dir)


def post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    request = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from {url}: {body[:500]}") from exc


@dataclass(frozen=True)
class PipelineConfig:
    workspace_dir: Path = WORKSPACE_DIR
    output_dir: Path = OUTPUT_DIR
    db_path: Path = WORKSPACE_DIR / "pipeline.sqlite3"
    backend: str = "mock"
    afnext_endpoint: str = "http://127.0.0.1:8001/v1/audio-flamingo/ask"
    afnext_batch_endpoint: str = "http://127.0.0.1:8001/v1/audio-flamingo/ask_batch"
    sam_audio_endpoint: str = "http://127.0.0.1:8002/v1/sam-audio/separate"
    sam_audio_batch_endpoint: str = "http://127.0.0.1:8002/v1/sam-audio/separate_batch"
    worker_enabled: bool = True
    worker_interval_seconds: float = 1.0
    sam_batch_size: int = 4
    afnext_batch_size: int = 1
    chunk_ms: int = 30_000
    overlap_ms: int = 5_000
    min_chunk_ms: int = 10_000
    sound_gate_min_dbfs: float = -40.0
    sound_gate_min_peak_dbfs: float = -35.0
    sound_gate_window_ms: int = 100
    sound_gate_min_active_ms: int = 1000
    sound_gate_min_active_ratio: float = 0.05
    request_timeout_seconds: float = 600.0
    storage_backend: str = "local"
    s3_bucket: str | None = None
    s3_prefix: str = "qlabeler"
    s3_region: str | None = None
    s3_endpoint_url: str | None = None
    s3_public_base_url: str | None = None
    s3_presign_seconds: int = 0

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        workspace_dir = Path(os.environ.get("WORKSPACE_DIR", str(WORKSPACE_DIR))).expanduser().resolve()
        output_dir = Path(os.environ.get("OUTPUT_DIR", str(OUTPUT_DIR))).expanduser().resolve()
        db_path = Path(os.environ.get("PIPELINE_DB_PATH", str(workspace_dir / "pipeline.sqlite3"))).expanduser().resolve()
        backend = os.environ.get("PIPELINE_BACKEND", "mock").strip().lower() or "mock"
        storage_backend = os.environ.get("PIPELINE_STORAGE_BACKEND", "local").strip().lower() or "local"
        return cls(
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            db_path=db_path,
            backend=backend,
            afnext_endpoint=os.environ.get("AFNEXT_ENDPOINT", "http://127.0.0.1:8001/v1/audio-flamingo/ask"),
            afnext_batch_endpoint=os.environ.get("AFNEXT_BATCH_ENDPOINT", "http://127.0.0.1:8001/v1/audio-flamingo/ask_batch"),
            sam_audio_endpoint=os.environ.get("SAM_AUDIO_ENDPOINT", "http://127.0.0.1:8002/v1/sam-audio/separate"),
            sam_audio_batch_endpoint=os.environ.get("SAM_AUDIO_BATCH_ENDPOINT", "http://127.0.0.1:8002/v1/sam-audio/separate_batch"),
            worker_enabled=parse_bool(os.environ.get("PIPELINE_WORKER_ENABLED"), default=True),
            worker_interval_seconds=float(os.environ.get("PIPELINE_WORKER_INTERVAL_SECONDS", "1.0")),
            sam_batch_size=max(1, int(os.environ.get("PIPELINE_SAM_BATCH_SIZE", "4"))),
            afnext_batch_size=max(1, int(os.environ.get("PIPELINE_AFNEXT_BATCH_SIZE", "1"))),
            chunk_ms=int(os.environ.get("PIPELINE_CHUNK_SECONDS", "30")) * 1000,
            overlap_ms=int(os.environ.get("PIPELINE_OVERLAP_SECONDS", "5")) * 1000,
            min_chunk_ms=int(os.environ.get("PIPELINE_MIN_CHUNK_SECONDS", "10")) * 1000,
            sound_gate_min_dbfs=float(os.environ.get("PIPELINE_SOUND_GATE_MIN_DBFS", "-50")),
            sound_gate_min_peak_dbfs=float(os.environ.get("PIPELINE_SOUND_GATE_MIN_PEAK_DBFS", "-55")),
            sound_gate_window_ms=int(os.environ.get("PIPELINE_SOUND_GATE_WINDOW_MS", "100")),
            sound_gate_min_active_ms=int(os.environ.get("PIPELINE_SOUND_GATE_MIN_ACTIVE_MS", "250")),
            sound_gate_min_active_ratio=float(os.environ.get("PIPELINE_SOUND_GATE_MIN_ACTIVE_RATIO", "0.01")),
            request_timeout_seconds=float(os.environ.get("PIPELINE_REQUEST_TIMEOUT_SECONDS", "600")),
            storage_backend=storage_backend,
            s3_bucket=os.environ.get("S3_BUCKET") or None,
            s3_prefix=os.environ.get("S3_PREFIX", "qlabeler"),
            s3_region=os.environ.get("S3_REGION") or os.environ.get("AWS_REGION") or None,
            s3_endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
            s3_public_base_url=os.environ.get("S3_PUBLIC_BASE_URL") or None,
            s3_presign_seconds=int(os.environ.get("S3_PRESIGN_SECONDS", "0")),
        )


class JobCreateRequest(BaseModel):
    audio_path: str = Field(description="Local path or file:// URL to an audio file.")
    prompt: str | None = Field(default=None, description="Optional prompt override for the Audio Flamingo stage.")


class RetryResponse(BaseModel):
    task_id: str
    status: str


class MockAudioRequest(BaseModel):
    audio_path: str | None = None
    file_path: str | None = None
    file: str | None = None
    audio_url: str | None = None
    prompt: str | None = None
    input: str | None = None
    question: str | None = None
    description: str | None = None
    output_prefix: str | None = None

    def audio_ref(self) -> str:
        value = self.audio_path or self.file_path or self.file or self.audio_url
        if not value:
            raise ValueError("Provide audio_path, file_path, file, or audio_url.")
        return value

    def prompt_text(self, default: str = "sound effects") -> str:
        value = self.prompt or self.input or self.question or self.description
        return value.strip() if value and value.strip() else default


class PipelineRuntime:
    def __init__(self, config: PipelineConfig, storage: ArtifactStorage | None = None):
        self.config = config
        self.storage = storage or create_storage_adapter(
            backend=config.storage_backend,
            output_dir=config.output_dir,
            s3_bucket=config.s3_bucket,
            s3_prefix=config.s3_prefix,
            s3_region=config.s3_region,
            s3_endpoint_url=config.s3_endpoint_url,
            s3_public_base_url=config.s3_public_base_url,
            s3_presign_seconds=config.s3_presign_seconds,
        )
        self._audio_duration_cache: dict[str, float] = {}
        self._audio_duration_cache_lock = threading.Lock()
        self._worker_threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._worker_lock = threading.Lock()

    def connect(self) -> sqlite3.Connection:
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.config.db_path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def init_db(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    source_audio_path TEXT NOT NULL,
                    prompt TEXT,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    has_music INTEGER DEFAULT NULL,
                    has_voices INTEGER DEFAULT NULL,
                    music_description TEXT DEFAULT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    audio_path TEXT NOT NULL,
                    start_ms INTEGER NOT NULL,
                    end_ms INTEGER NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(job_id, chunk_index)
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    chunk_id TEXT REFERENCES chunks(id) ON DELETE CASCADE,
                    queue TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS stems (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    prompt TEXT NOT NULL,
                    target_wav TEXT,
                    target_mp3 TEXT,
                    residual_wav TEXT,
                    residual_mp3 TEXT,
                    zip_path TEXT,
                    backend TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    chunk_id TEXT REFERENCES chunks(id) ON DELETE CASCADE,
                    task_id TEXT REFERENCES tasks(id) ON DELETE SET NULL,
                    kind TEXT NOT NULL,
                    path TEXT,
                    text TEXT,
                    prompt TEXT,
                    backend TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE,
                    chunk_id TEXT REFERENCES chunks(id) ON DELETE CASCADE,
                    task_id TEXT REFERENCES tasks(id) ON DELETE CASCADE,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_job_stage ON chunks(job_id, stage);
                CREATE INDEX IF NOT EXISTS idx_tasks_status_queue ON tasks(status, queue, created_at);
                CREATE INDEX IF NOT EXISTS idx_tasks_job ON tasks(job_id);
                CREATE INDEX IF NOT EXISTS idx_stems_job ON stems(job_id);
                CREATE INDEX IF NOT EXISTS idx_artifacts_job ON artifacts(job_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind, created_at);
                CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id, created_at);
                """
            )
            self._migrate_tasks_nullable_chunk_id(conn)

    def _migrate_tasks_nullable_chunk_id(self, conn: sqlite3.Connection) -> None:
        task_columns = conn.execute("PRAGMA table_info(tasks)").fetchall()
        chunk_column = next((column for column in task_columns if column["name"] == "chunk_id"), None)
        if chunk_column is None or int(chunk_column["notnull"]) == 0:
            return

        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            conn.executescript(
                """
            DROP INDEX IF EXISTS idx_tasks_status_queue;
            DROP INDEX IF EXISTS idx_tasks_job;

            CREATE TABLE tasks_new (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                chunk_id TEXT REFERENCES chunks(id) ON DELETE CASCADE,
                queue TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL DEFAULT '{}',
                result_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            );

            INSERT INTO tasks_new (
                id, job_id, chunk_id, queue, status, attempts, payload_json, result_json,
                error, created_at, updated_at, started_at, completed_at
            )
            SELECT
                id, job_id, chunk_id, queue, status, attempts, payload_json, result_json,
                error, created_at, updated_at, started_at, completed_at
            FROM tasks;

            DROP TABLE tasks;
            ALTER TABLE tasks_new RENAME TO tasks;

            CREATE INDEX IF NOT EXISTS idx_tasks_status_queue ON tasks(status, queue, created_at);
            CREATE INDEX IF NOT EXISTS idx_tasks_job ON tasks(job_id);
            """
            )
        finally:
            conn.execute("PRAGMA foreign_keys=ON")

    def start_worker(self) -> None:
        if not self.config.worker_enabled:
            return
        with self._worker_lock:
            if any(thread.is_alive() for thread in self._worker_threads):
                return
            self._stop_event.clear()
            self._worker_threads = [
                threading.Thread(
                    target=self.worker_loop,
                    args=(queue,),
                    name=f"pipeline-worker-{queue}",
                    daemon=True,
                )
                for queue in TASK_QUEUES
            ]
            for thread in self._worker_threads:
                thread.start()

    def stop_worker(self) -> None:
        self._stop_event.set()
        for thread in self._worker_threads:
            if thread.is_alive():
                thread.join(timeout=5)

    def worker_loop(self, queue: str | None = None) -> None:
        while not self._stop_event.is_set():
            did_work = self.process_pending_once(queue)
            if not did_work:
                self._stop_event.wait(self.config.worker_interval_seconds)

    def process_until_idle(self, *, max_tasks: int = 1000) -> int:
        processed = 0
        for _ in range(max_tasks):
            if not self.process_pending_once():
                break
            processed += 1
        return processed

    def create_job(self, audio_ref: str, prompt: str | None = None) -> dict[str, Any]:
        if self.config.chunk_ms <= 0:
            raise ValueError("PIPELINE_CHUNK_SECONDS must be greater than 0.")
        if self.config.overlap_ms < 0 or self.config.overlap_ms >= self.config.chunk_ms:
            raise ValueError("PIPELINE_OVERLAP_SECONDS must be at least 0 and smaller than PIPELINE_CHUNK_SECONDS.")

        audio_path = resolve_local_path(audio_ref, base_dir=self.config.workspace_dir)
        segment = AudioSegment.from_file(str(audio_path))
        duration_ms = len(segment)
        if duration_ms <= 0:
            raise ValueError(f"Audio has no duration: {audio_path}")

        job_id = new_id()
        created_at = now_iso()
        prompt_text = prompt.strip() if prompt and prompt.strip() else None
        job_dir = self.config.output_dir / "pipeline" / "jobs" / job_id
        chunk_dir = job_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=False)

        chunks = []
        starts = [0] if duration_ms <= self.config.chunk_ms else list(range(0, duration_ms, self.config.chunk_ms - self.config.overlap_ms))
        for chunk_index, start_ms in enumerate(starts, start=1):
            end_ms = min(start_ms + self.config.chunk_ms, duration_ms)
            if end_ms <= start_ms:
                continue
            chunk_duration_ms = end_ms - start_ms
            if chunk_duration_ms < self.config.min_chunk_ms:
                continue
            chunk_path = chunk_dir / f"chunk_{chunk_index:04d}_{start_ms:08d}-{end_ms:08d}.wav"
            segment[start_ms:end_ms].export(str(chunk_path), format="wav")
            chunks.append(
                {
                    "id": new_id(),
                    "chunk_index": chunk_index,
                    "audio_path": str(chunk_path),
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": chunk_duration_ms,
                }
            )

        if not chunks:
            raise ValueError(f"Could not create chunks for audio: {audio_path}")

        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO jobs (id, source_audio_path, prompt, status, chunk_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, str(audio_path), prompt_text, "queued", len(chunks), created_at, created_at),
            )
            self._insert_task(
                conn,
                job_id=job_id,
                chunk_id=None,
                queue=TASK_AUDIO_FLAMINGO,
                payload={
                    "purpose": PURPOSE_DESCRIBE_SCENE,
                    "audio_path": str(audio_path),
                    "prompt": self.default_scene_prompt(prompt_text),
                },
                created_at=created_at,
            )
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO chunks (
                        id, job_id, chunk_index, audio_path, start_ms, end_ms, duration_ms, stage, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk["id"],
                        job_id,
                        chunk["chunk_index"],
                        chunk["audio_path"],
                        chunk["start_ms"],
                        chunk["end_ms"],
                        chunk["duration_ms"],
                        STAGE_SOUND_GATE,
                        created_at,
                        created_at,
                    ),
                )
                self._insert_task(
                    conn,
                    job_id=job_id,
                    chunk_id=chunk["id"],
                    queue=TASK_SOUND_GATE,
                    payload={"purpose": PURPOSE_CHUNK_SOUND_GATE, "audio_path": chunk["audio_path"]},
                    created_at=created_at,
                )
            self._insert_event(
                conn,
                job_id=job_id,
                chunk_id=None,
                task_id=None,
                level="info",
                message=f"Created job with {len(chunks)} chunk(s)",
                data={"source_audio_path": str(audio_path)},
                created_at=created_at,
            )
            conn.execute("COMMIT")

        return self.job_detail(job_id)

    def save_upload(self, upload: UploadFile) -> Path:
        upload_dir = self.config.output_dir / "pipeline" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / clean_upload_name(upload.filename)

        try:
            upload.file.seek(0)
            with upload_path.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
        except Exception:
            upload_path.unlink(missing_ok=True)
            raise

        if upload_path.stat().st_size <= 0:
            upload_path.unlink(missing_ok=True)
            raise ValueError("Uploaded audio file is empty.")
        return upload_path

    def _insert_task(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        chunk_id: str | None,
        queue: str,
        payload: dict[str, Any],
        created_at: str | None = None,
    ) -> str:
        task_id = new_id()
        timestamp = created_at or now_iso()
        conn.execute(
            """
            INSERT INTO tasks (id, job_id, chunk_id, queue, status, payload_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (task_id, job_id, chunk_id, queue, STATUS_PENDING, json_dumps(payload), timestamp, timestamp),
        )
        return task_id

    def _insert_event(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str | None,
        chunk_id: str | None,
        task_id: str | None,
        level: str,
        message: str,
        data: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO events (job_id, chunk_id, task_id, level, message, data_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, chunk_id, task_id, level, message, json_dumps(data or {}), created_at or now_iso()),
        )

    def _insert_artifact(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        chunk_id: str | None,
        task_id: str | None,
        kind: str,
        path: str | None = None,
        text: str | None = None,
        prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> str:
        artifact_id = new_id()
        timestamp = created_at or now_iso()
        stored_metadata = dict(metadata or {})
        if path and Path(path).expanduser().is_file():
            stored_metadata.setdefault(
                "storage",
                self.storage.store_artifact_file(
                    Path(path),
                    artifact_id=artifact_id,
                    job_id=job_id,
                    chunk_id=chunk_id,
                    kind=kind,
                ),
            )
        conn.execute(
            """
            INSERT INTO artifacts (
                id, job_id, chunk_id, task_id, kind, path, text, prompt, backend, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                job_id,
                chunk_id,
                task_id,
                kind,
                path,
                text,
                prompt,
                self.config.backend,
                json_dumps(stored_metadata),
                timestamp,
            ),
        )
        return artifact_id

    def _has_incomplete_purpose_conn(self, conn: sqlite3.Connection, chunk_id: str, purpose: str) -> bool:
        rows = conn.execute(
            """
            SELECT payload_json
            FROM tasks
            WHERE chunk_id = ? AND status IN (?, ?)
            """,
            (chunk_id, STATUS_PENDING, STATUS_RUNNING),
        ).fetchall()
        for row in rows:
            payload = json_loads(row["payload_json"], {})
            if payload.get("purpose") == purpose:
                return True
        return False

    def claim_next_tasks(self, queue: str | None = None, *, limit: int = 1) -> list[dict[str, Any]]:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            queue_clause = " AND queue = ?" if queue is not None else ""
            params: list[Any] = [STATUS_PENDING]
            if queue is not None:
                params.append(queue)
            params.append(max(1, int(limit)))
            rows = conn.execute(
                f"""
                SELECT * FROM tasks
                WHERE status = ?{queue_clause}
                ORDER BY created_at ASC, rowid ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
            if not rows:
                conn.execute("COMMIT")
                return []

            timestamp = now_iso()
            claimed: list[dict[str, Any]] = []
            for row in rows:
                conn.execute(
                    """
                    UPDATE tasks
                    SET status = ?, attempts = attempts + 1, started_at = ?, updated_at = ?, error = NULL
                    WHERE id = ?
                    """,
                    (STATUS_RUNNING, timestamp, timestamp, row["id"]),
                )
                conn.execute(
                    "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
                    ("running", timestamp, row["job_id"]),
                )
                claimed.append(dict(row) | {"status": STATUS_RUNNING, "attempts": int(row["attempts"]) + 1})
            conn.execute("COMMIT")
            return claimed

    def claim_next_task(self, queue: str | None = None) -> dict[str, Any] | None:
        claimed = self.claim_next_tasks(queue, limit=1)
        return claimed[0] if claimed else None

    def queue_batch_size(self, queue: str | None) -> int:
        if queue == TASK_SAM_AUDIO:
            return max(1, self.config.sam_batch_size)
        if queue == TASK_AUDIO_FLAMINGO:
            return max(1, self.config.afnext_batch_size)
        return 1

    def process_pending_once(self, queue: str | None = None) -> bool:
        tasks = self.claim_next_tasks(queue, limit=self.queue_batch_size(queue))
        if not tasks:
            return False

        if len(tasks) > 1 and queue == TASK_SAM_AUDIO:
            self.process_sam_audio_batch(tasks)
        elif len(tasks) > 1 and queue == TASK_AUDIO_FLAMINGO:
            self.process_audio_flamingo_batch(tasks)
        else:
            for task in tasks:
                self._process_claimed_task(task)
        return True

    def _process_claimed_task(self, task: dict[str, Any]) -> None:
        try:
            if task["queue"] == TASK_SOUND_GATE:
                self.process_sound_gate(task)
            elif task["queue"] == TASK_AUDIO_FLAMINGO:
                self.process_audio_flamingo(task)
            elif task["queue"] == TASK_SAM_AUDIO:
                self.process_sam_audio(task)
            else:
                raise RuntimeError(f"Unknown task queue: {task['queue']}")
        except Exception as exc:
            self.fail_task(task, exception_detail(exc))

    def process_sound_gate(self, task: dict[str, Any]) -> None:
        payload = json_loads(task["payload_json"], {})
        purpose = payload.get("purpose") or PURPOSE_CHUNK_SOUND_GATE
        audio_path = Path(payload["audio_path"])
        result = self.sound_gate(audio_path)
        if purpose in {
            PURPOSE_GATE_MUSIC,
            PURPOSE_GATE_SFX_VOICE,
            PURPOSE_GATE_VOICE,
            PURPOSE_GATE_SFX,
            PURPOSE_GATE_REMAINING_SFX,
        }:
            self.process_track_sound_gate(task, payload, result)
            return

        timestamp = now_iso()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if result["has_sound"]:
                # Check job-level has_music flag from scene description.
                job_row = conn.execute(
                    "SELECT has_music, has_voices FROM jobs WHERE id = ?", (task["job_id"],)
                ).fetchone()
                has_music = job_row["has_music"] if job_row and job_row["has_music"] is not None else 1
                has_voices = job_row["has_voices"] if job_row and job_row["has_voices"] is not None else 1

                if has_music:
                    # Use the detailed music description from scene analysis as the SAM prompt.
                    job_row2 = conn.execute(
                        "SELECT music_description FROM jobs WHERE id = ?", (task["job_id"],)
                    ).fetchone()
                    music_prompt = (job_row2["music_description"] if job_row2 and job_row2["music_description"] else None) or "music"
                    # Normal flow: separate music first.
                    conn.execute(
                        "UPDATE chunks SET stage = ?, error = NULL, updated_at = ? WHERE id = ?",
                        (STAGE_SEPARATE_MUSIC, timestamp, task["chunk_id"]),
                    )
                    self._insert_task(
                        conn,
                        job_id=task["job_id"],
                        chunk_id=task["chunk_id"],
                        queue=TASK_SAM_AUDIO,
                        payload={
                            "purpose": PURPOSE_SEPARATE_MUSIC,
                            "audio_path": str(audio_path),
                            "prompt": music_prompt,
                        },
                        created_at=timestamp,
                    )
                elif has_voices:
                    # No music but has voices: separate voices directly from source.
                    conn.execute(
                        "UPDATE chunks SET stage = ?, error = NULL, updated_at = ? WHERE id = ?",
                        (STAGE_SEPARATE_VOICES, timestamp, task["chunk_id"]),
                    )
                    self._insert_task(
                        conn,
                        job_id=task["job_id"],
                        chunk_id=task["chunk_id"],
                        queue=TASK_SAM_AUDIO,
                        payload={
                            "purpose": PURPOSE_SEPARATE_VOICES,
                            "audio_path": str(audio_path),
                            "prompt": "human voice",
                        },
                        created_at=timestamp,
                    )
                else:
                    # No music, no voices: skip to SFX listing.
                    conn.execute(
                        "UPDATE chunks SET stage = ?, error = NULL, updated_at = ? WHERE id = ?",
                        (STAGE_LIST_SFX, timestamp, task["chunk_id"]),
                    )
                    self._insert_task(
                        conn,
                        job_id=task["job_id"],
                        chunk_id=task["chunk_id"],
                        queue=TASK_AUDIO_FLAMINGO,
                        payload={
                            "purpose": PURPOSE_LIST_SFX,
                            "audio_path": str(audio_path),
                        },
                        created_at=timestamp,
                    )
                message = "Sound gate passed"
            else:
                conn.execute(
                    "UPDATE chunks SET stage = ?, error = NULL, updated_at = ? WHERE id = ?",
                    (STAGE_SKIPPED_SILENT, timestamp, task["chunk_id"]),
                )
                message = "Sound gate skipped silent chunk"

            self._insert_artifact(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                kind=ARTIFACT_SOUND_GATE,
                prompt=purpose,
                metadata=result,
                created_at=timestamp,
            )
            self._complete_task_conn(conn, task["id"], result, timestamp)
            self._insert_event(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                level="info",
                message=message,
                data=result,
                created_at=timestamp,
            )
            self._update_job_status_conn(conn, task["job_id"], timestamp)
            conn.execute("COMMIT")

    def process_track_sound_gate(self, task: dict[str, Any], payload: dict[str, Any], result: dict[str, Any]) -> None:
        purpose = payload["purpose"]
        track_type = payload.get("track_type") or {
            PURPOSE_GATE_MUSIC: "music",
            PURPOSE_GATE_SFX_VOICE: "sfx_voice",
            PURPOSE_GATE_VOICE: "voice",
            PURPOSE_GATE_SFX: "sfx",
            PURPOSE_GATE_REMAINING_SFX: "remaining_sfx",
        }.get(purpose, "track")
        audio_path = payload["audio_path"]
        iteration = int(payload.get("iteration") or 0)
        timestamp = now_iso()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._insert_artifact(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                kind=ARTIFACT_SOUND_GATE,
                prompt=purpose,
                metadata={"track_type": track_type, **result},
                created_at=timestamp,
            )

            if track_type == "music":
                if result["has_sound"]:
                    self._insert_task(
                        conn,
                        job_id=task["job_id"],
                        chunk_id=task["chunk_id"],
                        queue=TASK_AUDIO_FLAMINGO,
                        payload={
                            "purpose": PURPOSE_DESCRIBE_MUSIC,
                            "audio_path": audio_path,
                            "prompt": self.default_music_description_prompt(),
                            "source_artifact_id": payload.get("artifact_id"),
                        },
                        created_at=timestamp,
                    )
                    stage = STAGE_DESCRIBE_MUSIC
                    message = "Music track gate passed"
                else:
                    # Music target is empty — separation produced nothing useful.
                    # Use the original chunk audio for subsequent stages instead of the residual.
                    chunk_row = conn.execute(
                        "SELECT audio_path FROM chunks WHERE id = ?", (task["chunk_id"],)
                    ).fetchone()
                    original_audio = chunk_row["audio_path"] if chunk_row else audio_path
                    job_row = conn.execute(
                        "SELECT has_voices FROM jobs WHERE id = ?", (task["job_id"],)
                    ).fetchone()
                    has_voices = job_row["has_voices"] if job_row and job_row["has_voices"] is not None else 1
                    if has_voices:
                        self._insert_task(
                            conn,
                            job_id=task["job_id"],
                            chunk_id=task["chunk_id"],
                            queue=TASK_SAM_AUDIO,
                            payload={
                                "purpose": PURPOSE_SEPARATE_VOICES,
                                "audio_path": original_audio,
                                "prompt": "human voice",
                            },
                            created_at=timestamp,
                        )
                        stage = STAGE_SEPARATE_VOICES
                    else:
                        self._insert_task(
                            conn,
                            job_id=task["job_id"],
                            chunk_id=task["chunk_id"],
                            queue=TASK_AUDIO_FLAMINGO,
                            payload={
                                "purpose": PURPOSE_LIST_SFX,
                                "audio_path": original_audio,
                            },
                            created_at=timestamp,
                        )
                        stage = STAGE_LIST_SFX
                    message = "Music track gate skipped (empty) — using original audio for next stage"
            elif track_type == "sfx_voice":
                if result["has_sound"]:
                    # Check job-level has_voices before separating voice.
                    job_row = conn.execute(
                        "SELECT has_voices FROM jobs WHERE id = ?", (task["job_id"],)
                    ).fetchone()
                    has_voices = job_row["has_voices"] if job_row and job_row["has_voices"] is not None else 1
                    if has_voices:
                        self._insert_task(
                            conn,
                            job_id=task["job_id"],
                            chunk_id=task["chunk_id"],
                            queue=TASK_SAM_AUDIO,
                            payload={
                                "purpose": PURPOSE_SEPARATE_VOICES,
                                "audio_path": audio_path,
                                "prompt": "human voice",
                                "source_artifact_id": payload.get("artifact_id"),
                            },
                            created_at=timestamp,
                        )
                        stage = STAGE_SEPARATE_VOICES
                    else:
                        # No voices: skip to SFX listing on this track.
                        self._insert_task(
                            conn,
                            job_id=task["job_id"],
                            chunk_id=task["chunk_id"],
                            queue=TASK_AUDIO_FLAMINGO,
                            payload={
                                "purpose": PURPOSE_LIST_SFX,
                                "audio_path": audio_path,
                                "source_artifact_id": payload.get("artifact_id"),
                            },
                            created_at=timestamp,
                        )
                        stage = STAGE_LIST_SFX
                else:
                    stage = STAGE_SKIPPED_SFX_VOICE
                message = "SFX+voice track gate passed" if result["has_sound"] else "SFX+voice track gate skipped silent output"
            elif track_type == "voice":
                if result["has_sound"]:
                    self._insert_task(
                        conn,
                        job_id=task["job_id"],
                        chunk_id=task["chunk_id"],
                        queue=TASK_AUDIO_FLAMINGO,
                        payload={
                            "purpose": PURPOSE_TRANSCRIBE_VOICE,
                            "audio_path": audio_path,
                            "prompt": self.default_voice_transcription_prompt(),
                            "source_artifact_id": payload.get("artifact_id"),
                        },
                        created_at=timestamp,
                    )
                    stage = STAGE_TRANSCRIBE_VOICE
                else:
                    # Voice target is empty — use the input audio (sfx_voice or original) for SFX.
                    # Get the audio that was fed into voice separation (the parent track).
                    parent_audio = payload.get("source_audio_path")
                    if not parent_audio:
                        chunk_row = conn.execute(
                            "SELECT audio_path FROM chunks WHERE id = ?", (task["chunk_id"],)
                        ).fetchone()
                        parent_audio = chunk_row["audio_path"] if chunk_row else audio_path
                    self._insert_task(
                        conn,
                        job_id=task["job_id"],
                        chunk_id=task["chunk_id"],
                        queue=TASK_AUDIO_FLAMINGO,
                        payload={
                            "purpose": PURPOSE_LIST_SFX,
                            "audio_path": parent_audio,
                        },
                        created_at=timestamp,
                    )
                    stage = STAGE_LIST_SFX
                message = "Voice track gate passed" if result["has_sound"] else "Voice track gate skipped (empty) — using parent audio for SFX"
            elif track_type == "sfx":
                if result["has_sound"]:
                    self._insert_task(
                        conn,
                        job_id=task["job_id"],
                        chunk_id=task["chunk_id"],
                        queue=TASK_AUDIO_FLAMINGO,
                        payload={
                            "purpose": PURPOSE_LIST_SFX,
                            "audio_path": audio_path,
                            "prompt": self.default_sfx_list_prompt(),
                            "iteration": 1,
                            "source_artifact_id": payload.get("artifact_id"),
                        },
                        created_at=timestamp,
                    )
                    stage = STAGE_LIST_SFX
                else:
                    stage = STAGE_SKIPPED_SFX
                message = "SFX track gate passed" if result["has_sound"] else "SFX track gate skipped silent output"
            elif track_type == "remaining_sfx":
                if result["has_sound"]:
                    if iteration >= SFX_LOOP_MAX_ITERATIONS:
                        self._insert_artifact(
                            conn,
                            job_id=task["job_id"],
                            chunk_id=task["chunk_id"],
                            task_id=task["id"],
                            kind=ARTIFACT_SFX_LOOP_DEBUG,
                            path=audio_path,
                            text=f"Stopped after {iteration} SFX extraction iteration(s).",
                            prompt=purpose,
                            metadata={
                                "reason": "iteration_limit",
                                "iteration": iteration,
                                "max_iterations": SFX_LOOP_MAX_ITERATIONS,
                                "source_artifact_id": payload.get("artifact_id"),
                            },
                            created_at=timestamp,
                        )
                        stage = STAGE_SFX_ITERATION_LIMIT
                        message = "SFX loop stopped at iteration limit"
                    else:
                        self._insert_task(
                            conn,
                            job_id=task["job_id"],
                            chunk_id=task["chunk_id"],
                            queue=TASK_AUDIO_FLAMINGO,
                            payload={
                                "purpose": PURPOSE_LIST_SFX,
                                "audio_path": audio_path,
                                "prompt": self.default_sfx_list_prompt(),
                                "iteration": iteration + 1,
                                "source_artifact_id": payload.get("artifact_id"),
                            },
                            created_at=timestamp,
                        )
                        stage = STAGE_LIST_SFX
                        message = "Remaining SFX gate passed"
                else:
                    self._insert_artifact(
                        conn,
                        job_id=task["job_id"],
                        chunk_id=task["chunk_id"],
                        task_id=task["id"],
                        kind=ARTIFACT_SFX_LOOP_DEBUG,
                        path=audio_path,
                        text="Remaining SFX residual is empty.",
                        prompt=purpose,
                        metadata={
                            "reason": "residual_empty",
                            "iteration": iteration,
                            "source_artifact_id": payload.get("artifact_id"),
                        },
                        created_at=timestamp,
                    )
                    stage = STAGE_SFX_EXHAUSTED
                    message = "SFX loop exhausted remaining audio"
            else:
                raise RuntimeError(f"Unknown track gate type: {track_type}")

            conn.execute(
                "UPDATE chunks SET stage = ?, error = NULL, updated_at = ? WHERE id = ?",
                (stage, timestamp, task["chunk_id"]),
            )
            self._complete_task_conn(conn, task["id"], result, timestamp)
            self._insert_event(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                level="info",
                message=message,
                data={"track_type": track_type, **result},
                created_at=timestamp,
            )
            self._update_job_status_conn(conn, task["job_id"], timestamp)
            conn.execute("COMMIT")

    def _audio_flamingo_prompt(self, task: dict[str, Any], payload: dict[str, Any]) -> str:
        purpose = payload.get("purpose") or PURPOSE_DESCRIBE_SFX
        return payload.get("prompt") or self.default_prompt_for_audio_flamingo_purpose(purpose, task["job_id"])

    def process_audio_flamingo(self, task: dict[str, Any]) -> None:
        payload = json_loads(task["payload_json"], {})
        audio_path = Path(payload["audio_path"])
        prompt = self._audio_flamingo_prompt(task, payload)
        if self.config.backend == "mock":
            response = self.mock_audio_flamingo(audio_path, prompt)
        else:
            response = post_json(
                self.config.afnext_endpoint,
                {
                    "audio_path": str(audio_path),
                    "input": prompt,
                    "max_new_tokens": 256,
                    "repetition_penalty": 1.2,
                },
                timeout=self.config.request_timeout_seconds,
            )
        self._complete_audio_flamingo_task(task, payload, response)

    def process_audio_flamingo_batch(self, tasks: list[dict[str, Any]]) -> None:
        prepared: list[tuple[dict[str, Any], dict[str, Any], str, str]] = []
        for task in tasks:
            try:
                payload = json_loads(task["payload_json"], {})
                audio_path = str(Path(payload["audio_path"]))
                prompt = self._audio_flamingo_prompt(task, payload)
                prepared.append((task, payload, audio_path, prompt))
            except Exception as exc:
                self.fail_task(task, exception_detail(exc))
        if not prepared:
            return

        if self.config.backend == "mock":
            for task, payload, audio_path, prompt in prepared:
                try:
                    response = self.mock_audio_flamingo(Path(audio_path), prompt)
                    self._complete_audio_flamingo_task(task, payload, response)
                except Exception as exc:
                    self.fail_task(task, exception_detail(exc))
            return

        try:
            batch_response = post_json(
                self.config.afnext_batch_endpoint,
                {
                    "items": [
                        {"audio_path": audio_path, "prompt": prompt}
                        for _, _, audio_path, prompt in prepared
                    ],
                    "max_new_tokens": 256,
                    "repetition_penalty": 1.2,
                },
                timeout=self.config.request_timeout_seconds,
            )
            results = batch_response.get("results")
            if not isinstance(results, list) or len(results) != len(prepared):
                raise RuntimeError(
                    f"Audio Flamingo batch endpoint returned {len(results) if isinstance(results, list) else 'no'} "
                    f"results for {len(prepared)} items"
                )
        except Exception as exc:
            error = exception_detail(exc)
            for task, _, _, _ in prepared:
                self.fail_task(task, error)
            return

        for (task, payload, _, _), entry in zip(prepared, results):
            try:
                response = self._unwrap_batch_entry(entry)
                self._complete_audio_flamingo_task(task, payload, response)
            except Exception as exc:
                self.fail_task(task, exception_detail(exc))

    @staticmethod
    def _unwrap_batch_entry(entry: Any) -> dict[str, Any]:
        if not isinstance(entry, dict):
            raise RuntimeError(f"Invalid batch result entry: {entry!r}")
        if entry.get("error"):
            raise RuntimeError(str(entry["error"]))
        result = entry.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"Batch result entry missing result payload: {entry!r}")
        return result

    def _complete_audio_flamingo_task(self, task: dict[str, Any], payload: dict[str, Any], response: dict[str, Any]) -> None:
        purpose = payload.get("purpose") or PURPOSE_DESCRIBE_SFX
        audio_path = Path(payload["audio_path"])
        text = str(response.get("text", "")).strip()
        if purpose == PURPOSE_DESCRIBE_SCENE:
            self.complete_scene_description(task, text, response)
            return
        if purpose == PURPOSE_DESCRIBE_MUSIC:
            self.complete_music_description(task, text, response)
            return
        if purpose == PURPOSE_TRANSCRIBE_VOICE:
            self.complete_voice_transcription(task, text, response)
            return
        if purpose == PURPOSE_LIST_SFX:
            self.complete_sfx_list(task, text, response)
            return

        target_prompt = self.extract_sam_prompt(text)
        timestamp = now_iso()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE chunks SET stage = ?, error = NULL, updated_at = ? WHERE id = ?",
                (STAGE_SEPARATE_SFX, timestamp, task["chunk_id"]),
            )
            self._insert_task(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                queue=TASK_SAM_AUDIO,
                payload={
                    "purpose": PURPOSE_SEPARATE_SFX,
                    "audio_path": str(audio_path),
                    "prompt": target_prompt,
                    "audio_flamingo_text": text,
                },
                created_at=timestamp,
            )
            self._complete_task_conn(
                conn,
                task["id"],
                {"text": text, "sam_prompt": target_prompt, "backend": self.config.backend},
                timestamp,
            )
            self._insert_event(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                level="info",
                message="Audio Flamingo produced SAM prompt",
                data={"sam_prompt": target_prompt},
                created_at=timestamp,
            )
            self._update_job_status_conn(conn, task["job_id"], timestamp)
            conn.execute("COMMIT")

    def complete_sfx_list(self, task: dict[str, Any], text: str, response: dict[str, Any]) -> None:
        payload = json_loads(task["payload_json"], {})
        iteration = int(payload.get("iteration") or 1)
        parsed, parse_error = self.parse_sfx_list_text(text)
        effects = parsed.get("effects", []) if parsed else []
        first_effect = effects[0] if effects else None
        sam_prompt = str((first_effect or {}).get("sam_prompt") or "").strip()
        timestamp = now_iso()

        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            sfx_list_artifact_id = self._insert_artifact(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                kind=ARTIFACT_SFX_LIST,
                path=payload.get("audio_path"),
                text=text,
                prompt=payload.get("prompt"),
                metadata={
                    "iteration": iteration,
                    "parsed": parsed,
                    "parse_error": parse_error,
                    "response": response,
                    "source_artifact_id": payload.get("source_artifact_id"),
                },
                created_at=timestamp,
            )

            if parse_error:
                stage = STAGE_SFX_EXHAUSTED
                result = {"text": text, "parsed": parsed, "parse_error": parse_error, "backend": self.config.backend}
                self._insert_artifact(
                    conn,
                    job_id=task["job_id"],
                    chunk_id=task["chunk_id"],
                    task_id=task["id"],
                    kind=ARTIFACT_SFX_LOOP_DEBUG,
                    path=payload.get("audio_path"),
                    text=text,
                    prompt=payload.get("prompt"),
                    metadata={
                        "reason": "parse_error",
                        "iteration": iteration,
                        "parse_error": parse_error,
                        "source_artifact_id": payload.get("source_artifact_id"),
                    },
                    created_at=timestamp,
                )
                message = "Audio Flamingo SFX list parse failed; loop exhausted"
            elif not first_effect:
                stage = STAGE_SFX_EXHAUSTED
                result = {"text": text, "parsed": parsed, "selected_effect": None, "backend": self.config.backend}
                self._insert_artifact(
                    conn,
                    job_id=task["job_id"],
                    chunk_id=task["chunk_id"],
                    task_id=task["id"],
                    kind=ARTIFACT_SFX_LOOP_DEBUG,
                    path=payload.get("audio_path"),
                    text="Audio Flamingo returned no SFX candidates.",
                    prompt=payload.get("prompt"),
                    metadata={
                        "reason": "empty_effects",
                        "iteration": iteration,
                        "source_artifact_id": payload.get("source_artifact_id"),
                    },
                    created_at=timestamp,
                )
                message = "Audio Flamingo returned no SFX candidates"
            elif not sam_prompt:
                stage = STAGE_SFX_LOOP_FAILED
                result = {"text": text, "parsed": parsed, "selected_effect": first_effect, "backend": self.config.backend}
                self._insert_artifact(
                    conn,
                    job_id=task["job_id"],
                    chunk_id=task["chunk_id"],
                    task_id=task["id"],
                    kind=ARTIFACT_SFX_LOOP_DEBUG,
                    path=payload.get("audio_path"),
                    text=text,
                    prompt=payload.get("prompt"),
                    metadata={
                        "reason": "missing_sam_prompt",
                        "iteration": iteration,
                        "selected_effect": first_effect,
                        "source_artifact_id": payload.get("source_artifact_id"),
                    },
                    created_at=timestamp,
                )
                message = "Audio Flamingo SFX candidate missing SAM prompt"
            else:
                stage = STAGE_SEPARATE_SFX
                result = {
                    "text": text,
                    "parsed": parsed,
                    "selected_effect": first_effect,
                    "sam_prompt": sam_prompt,
                    "backend": self.config.backend,
                }
                self._insert_task(
                    conn,
                    job_id=task["job_id"],
                    chunk_id=task["chunk_id"],
                    queue=TASK_SAM_AUDIO,
                    payload={
                        "purpose": PURPOSE_SEPARATE_SFX,
                        "audio_path": payload["audio_path"],
                        "prompt": sam_prompt,
                        "iteration": iteration,
                        "selected_effect": first_effect,
                        "sfx_list_artifact_id": sfx_list_artifact_id,
                        "source_artifact_id": payload.get("source_artifact_id"),
                    },
                    created_at=timestamp,
                )
                message = "Audio Flamingo listed SFX candidate"

            conn.execute(
                "UPDATE chunks SET stage = ?, error = NULL, updated_at = ? WHERE id = ?",
                (stage, timestamp, task["chunk_id"]),
            )
            self._complete_task_conn(conn, task["id"], result, timestamp)
            self._insert_event(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                level="info",
                message=message,
                data=result,
                created_at=timestamp,
            )
            self._update_job_status_conn(conn, task["job_id"], timestamp)
            conn.execute("COMMIT")

    def complete_scene_description(self, task: dict[str, Any], text: str, response: dict[str, Any]) -> None:
        timestamp = now_iso()
        # Parse has_music / has_voices / music_description from the structured response.
        has_music = self._parse_scene_bool(text, "HAS_MUSIC")
        has_voices = self._parse_scene_bool(text, "HAS_VOICES")
        music_description = self._parse_scene_field(text, "MUSIC_DESCRIPTION")
        if music_description and music_description.lower() in ("none", "n/a", ""):
            music_description = None
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._insert_artifact(
                conn,
                job_id=task["job_id"],
                chunk_id=None,
                task_id=task["id"],
                kind=ARTIFACT_SCENE_DESCRIPTION,
                text=text,
                prompt=(json_loads(task["payload_json"], {}) or {}).get("prompt"),
                metadata={**response, "has_music": has_music, "has_voices": has_voices, "music_description": music_description},
                created_at=timestamp,
            )
            # Store has_music/has_voices/music_description on the job for downstream decisions.
            conn.execute(
                "UPDATE jobs SET has_music = ?, has_voices = ?, music_description = ?, updated_at = ? WHERE id = ?",
                (int(has_music), int(has_voices), music_description, timestamp, task["job_id"]),
            )
            self._complete_task_conn(
                conn,
                task["id"],
                {"text": text, "backend": self.config.backend, "has_music": has_music, "has_voices": has_voices, "music_description": music_description},
                timestamp,
            )
            self._insert_event(
                conn,
                job_id=task["job_id"],
                chunk_id=None,
                task_id=task["id"],
                level="info",
                message=f"Scene described: has_music={has_music}, has_voices={has_voices}, music='{music_description or ''}'",
                data={"text": text, "has_music": has_music, "has_voices": has_voices, "music_description": music_description},
                created_at=timestamp,
            )
            self._update_job_status_conn(conn, task["job_id"], timestamp)
            conn.execute("COMMIT")

    @staticmethod
    def _parse_scene_field(text: str, key: str) -> str | None:
        """Parse a field like 'MUSIC_DESCRIPTION: some text here' from the scene description."""
        import re
        pattern = rf"(?i){re.escape(key)}\s*:\s*(.+)"
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            return value if value else None
        return None

    @staticmethod
    def _parse_scene_bool(text: str, key: str) -> bool:
        """Parse a boolean like 'HAS_MUSIC: true' from the scene description text."""
        import re
        pattern = rf"(?i){re.escape(key)}\s*:\s*(true|false|yes|no|1|0)"
        match = re.search(pattern, text)
        if match:
            return match.group(1).lower() in ("true", "yes", "1")
        # Fallback: assume true if not explicitly stated (conservative).
        return True

    def complete_music_description(self, task: dict[str, Any], text: str, response: dict[str, Any]) -> None:
        payload = json_loads(task["payload_json"], {})
        timestamp = now_iso()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._insert_artifact(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                kind=ARTIFACT_MUSIC_DESCRIPTION,
                text=text,
                prompt=payload.get("prompt"),
                metadata={
                    "source_artifact_id": payload.get("source_artifact_id"),
                    "response": response,
                },
                created_at=timestamp,
            )
            conn.execute(
                "UPDATE chunks SET stage = ?, error = NULL, updated_at = ? WHERE id = ?",
                (STAGE_MUSIC_DESCRIBED, timestamp, task["chunk_id"]),
            )
            self._complete_task_conn(
                conn,
                task["id"],
                {"text": text, "backend": self.config.backend},
                timestamp,
            )
            self._insert_event(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                level="info",
                message="Audio Flamingo described music track",
                data={"text": text},
                created_at=timestamp,
            )
            self._update_job_status_conn(conn, task["job_id"], timestamp)
            conn.execute("COMMIT")

    def complete_voice_transcription(self, task: dict[str, Any], text: str, response: dict[str, Any]) -> None:
        payload = json_loads(task["payload_json"], {})
        timestamp = now_iso()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._insert_artifact(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                kind=ARTIFACT_VOICE_TRANSCRIPTION,
                text=text,
                prompt=payload.get("prompt"),
                metadata={
                    "source_artifact_id": payload.get("source_artifact_id"),
                    "response": response,
                },
                created_at=timestamp,
            )
            conn.execute(
                "UPDATE chunks SET stage = ?, error = NULL, updated_at = ? WHERE id = ?",
                (STAGE_VOICE_TRANSCRIBED, timestamp, task["chunk_id"]),
            )
            self._complete_task_conn(
                conn,
                task["id"],
                {"text": text, "backend": self.config.backend},
                timestamp,
            )
            self._insert_event(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                level="info",
                message="Audio Flamingo transcribed voice track with diarization",
                data={"text": text},
                created_at=timestamp,
            )
            self._update_job_status_conn(conn, task["job_id"], timestamp)
            conn.execute("COMMIT")

    def _sam_audio_request(self, task: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        purpose = payload.get("purpose") or PURPOSE_SEPARATE_SFX
        prompt = payload["prompt"]
        chunk = self.chunk(task["chunk_id"])
        output_prefix = clean_prefix(f"job_{task['job_id']}_chunk_{chunk['chunk_index']:04d}_{purpose}_{prompt}")
        return {
            "audio_path": str(Path(payload["audio_path"])),
            "input": prompt,
            "output_prefix": output_prefix,
            "max_audio_seconds": max(35, int((chunk["duration_ms"] + 999) / 1000)),
            "predict_spans": False,
            "reranking_candidates": 1,
        }

    def process_sam_audio(self, task: dict[str, Any]) -> None:
        payload = json_loads(task["payload_json"], {})
        request = self._sam_audio_request(task, payload)

        if self.config.backend == "mock":
            response = self.mock_sam_audio(
                Path(request["audio_path"]), request["input"], request["output_prefix"], task["job_id"]
            )
        else:
            response = post_json(
                self.config.sam_audio_endpoint,
                request,
                timeout=self.config.request_timeout_seconds,
            )
        self._complete_sam_audio_task(task, payload, response)

    def process_sam_audio_batch(self, tasks: list[dict[str, Any]]) -> None:
        prepared: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for task in tasks:
            try:
                payload = json_loads(task["payload_json"], {})
                request = self._sam_audio_request(task, payload)
                prepared.append((task, payload, request))
            except Exception as exc:
                self.fail_task(task, exception_detail(exc))
        if not prepared:
            return

        if self.config.backend == "mock":
            for task, payload, request in prepared:
                try:
                    response = self.mock_sam_audio(
                        Path(request["audio_path"]), request["input"], request["output_prefix"], task["job_id"]
                    )
                    self._complete_sam_audio_task(task, payload, response)
                except Exception as exc:
                    self.fail_task(task, exception_detail(exc))
            return

        try:
            batch_response = post_json(
                self.config.sam_audio_batch_endpoint,
                {
                    "items": [
                        {
                            "audio_path": request["audio_path"],
                            "prompt": request["input"],
                            "output_prefix": request["output_prefix"],
                            "max_audio_seconds": request["max_audio_seconds"],
                        }
                        for _, _, request in prepared
                    ],
                    "predict_spans": False,
                    "reranking_candidates": 1,
                },
                timeout=self.config.request_timeout_seconds,
            )
            results = batch_response.get("results")
            if not isinstance(results, list) or len(results) != len(prepared):
                raise RuntimeError(
                    f"SAM-Audio batch endpoint returned {len(results) if isinstance(results, list) else 'no'} "
                    f"results for {len(prepared)} items"
                )
        except Exception as exc:
            error = exception_detail(exc)
            for task, _, _ in prepared:
                self.fail_task(task, error)
            return

        for (task, payload, _), entry in zip(prepared, results):
            try:
                response = self._unwrap_batch_entry(entry)
                self._complete_sam_audio_task(task, payload, response)
            except Exception as exc:
                self.fail_task(task, exception_detail(exc))

    def _complete_sam_audio_task(self, task: dict[str, Any], payload: dict[str, Any], response: dict[str, Any]) -> None:
        purpose = payload.get("purpose") or PURPOSE_SEPARATE_SFX
        prompt = payload["prompt"]
        if purpose == PURPOSE_SEPARATE_MUSIC:
            self.complete_music_separation(task, prompt, response)
            return
        if purpose == PURPOSE_SEPARATE_VOICES:
            self.complete_voice_separation(task, prompt, response)
            return
        if purpose == PURPOSE_SEPARATE_SFX:
            self.complete_sfx_separation(task, prompt, response)
            return

        timestamp = now_iso()
        target = response.get("target") or {}
        residual = response.get("residual") or {}
        zip_ref = response.get("zip") or {}
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO stems (
                    id, job_id, chunk_id, prompt, target_wav, target_mp3, residual_wav,
                    residual_mp3, zip_path, backend, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id(),
                    task["job_id"],
                    task["chunk_id"],
                    prompt,
                    (target.get("wav") or {}).get("path"),
                    (target.get("mp3") or {}).get("path"),
                    (residual.get("wav") or {}).get("path"),
                    (residual.get("mp3") or {}).get("path"),
                    zip_ref.get("path"),
                    self.config.backend,
                    json_dumps(response),
                    timestamp,
                ),
            )
            conn.execute(
                "UPDATE chunks SET stage = ?, error = NULL, updated_at = ? WHERE id = ?",
                (STAGE_COMPLETE, timestamp, task["chunk_id"]),
            )
            self._complete_task_conn(conn, task["id"], response, timestamp)
            self._insert_event(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                level="info",
                message="SAM-Audio separation completed",
                data={"prompt": prompt},
                created_at=timestamp,
            )
            self._update_job_status_conn(conn, task["job_id"], timestamp)
            conn.execute("COMMIT")

    def complete_music_separation(self, task: dict[str, Any], prompt: str, response: dict[str, Any]) -> None:
        timestamp = now_iso()
        target = response.get("target") or {}
        residual = response.get("residual") or {}
        raw_target = response.get("raw_target") or {}
        music_path = self.preferred_audio_path(target)
        sfx_voice_path = self.preferred_audio_path(residual)
        if not music_path:
            raise RuntimeError(f"SAM-Audio music response missing target audio path: {response}")
        if not sfx_voice_path:
            raise RuntimeError(f"SAM-Audio music response missing residual audio path: {response}")

        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            music_artifact_id = self._insert_artifact(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                kind=ARTIFACT_MUSIC_TRACK,
                path=music_path,
                prompt=prompt,
                metadata={"role": "target", "refs": target, "response": response},
                created_at=timestamp,
            )
            # Store raw model output for comparison in the UI.
            raw_music_path = self.preferred_audio_path(raw_target)
            if raw_music_path:
                self._insert_artifact(
                    conn,
                    job_id=task["job_id"],
                    chunk_id=task["chunk_id"],
                    task_id=task["id"],
                    kind=ARTIFACT_MUSIC_TRACK_RAW,
                    path=raw_music_path,
                    prompt=f"{prompt} (raw model)",
                    metadata={"role": "raw_target", "refs": raw_target},
                    created_at=timestamp,
                )
            sfx_voice_artifact_id = self._insert_artifact(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                kind=ARTIFACT_SFX_VOICE_TRACK,
                path=sfx_voice_path,
                prompt="residual after music",
                metadata={"role": "residual", "refs": residual, "response": response},
                created_at=timestamp,
            )
            self._insert_task(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                queue=TASK_SOUND_GATE,
                payload={
                    "purpose": PURPOSE_GATE_MUSIC,
                    "track_type": "music",
                    "audio_path": music_path,
                    "artifact_id": music_artifact_id,
                },
                created_at=timestamp,
            )
            self._insert_task(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                queue=TASK_SOUND_GATE,
                payload={
                    "purpose": PURPOSE_GATE_SFX_VOICE,
                    "track_type": "sfx_voice",
                    "audio_path": sfx_voice_path,
                    "artifact_id": sfx_voice_artifact_id,
                },
                created_at=timestamp,
            )
            conn.execute(
                "UPDATE chunks SET stage = ?, error = NULL, updated_at = ? WHERE id = ?",
                (STAGE_GATE_MUSIC, timestamp, task["chunk_id"]),
            )
            self._complete_task_conn(conn, task["id"], response, timestamp)
            self._insert_event(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                level="info",
                message="SAM-Audio separated music track",
                data={"prompt": prompt, "music_path": music_path, "sfx_voice_path": sfx_voice_path},
                created_at=timestamp,
            )
            self._update_job_status_conn(conn, task["job_id"], timestamp)
            conn.execute("COMMIT")

    def complete_voice_separation(self, task: dict[str, Any], prompt: str, response: dict[str, Any]) -> None:
        timestamp = now_iso()
        target = response.get("target") or {}
        residual = response.get("residual") or {}
        raw_target = response.get("raw_target") or {}
        voice_path = self.preferred_audio_path(target)
        sfx_path = self.preferred_audio_path(residual)
        if not voice_path:
            raise RuntimeError(f"SAM-Audio voice response missing target audio path: {response}")
        if not sfx_path:
            raise RuntimeError(f"SAM-Audio voice response missing residual audio path: {response}")

        payload = json_loads(task["payload_json"], {})
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            voice_artifact_id = self._insert_artifact(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                kind=ARTIFACT_VOICE_TRACK,
                path=voice_path,
                prompt=prompt,
                metadata={
                    "role": "target",
                    "source_artifact_id": payload.get("source_artifact_id"),
                    "refs": target,
                    "response": response,
                },
                created_at=timestamp,
            )
            # Store raw model output for comparison.
            raw_voice_path = self.preferred_audio_path(raw_target)
            if raw_voice_path:
                self._insert_artifact(
                    conn,
                    job_id=task["job_id"],
                    chunk_id=task["chunk_id"],
                    task_id=task["id"],
                    kind=ARTIFACT_VOICE_TRACK_RAW,
                    path=raw_voice_path,
                    prompt=f"{prompt} (raw model)",
                    metadata={"role": "raw_target", "refs": raw_target},
                    created_at=timestamp,
                )
            sfx_artifact_id = self._insert_artifact(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                kind=ARTIFACT_SFX_TRACK,
                path=sfx_path,
                prompt="residual after voice",
                metadata={
                    "role": "residual",
                    "source_artifact_id": payload.get("source_artifact_id"),
                    "refs": residual,
                    "response": response,
                },
                created_at=timestamp,
            )
            self._insert_task(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                queue=TASK_SOUND_GATE,
                payload={
                    "purpose": PURPOSE_GATE_VOICE,
                    "track_type": "voice",
                    "audio_path": voice_path,
                    "artifact_id": voice_artifact_id,
                    "source_audio_path": payload.get("audio_path"),
                },
                created_at=timestamp,
            )
            self._insert_task(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                queue=TASK_SOUND_GATE,
                payload={
                    "purpose": PURPOSE_GATE_SFX,
                    "track_type": "sfx",
                    "audio_path": sfx_path,
                    "artifact_id": sfx_artifact_id,
                },
                created_at=timestamp,
            )
            conn.execute(
                "UPDATE chunks SET stage = ?, error = NULL, updated_at = ? WHERE id = ?",
                (STAGE_GATE_VOICE, timestamp, task["chunk_id"]),
            )
            self._complete_task_conn(conn, task["id"], response, timestamp)
            self._insert_event(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                level="info",
                message="SAM-Audio separated voice and SFX tracks",
                data={"prompt": prompt, "voice_path": voice_path, "sfx_path": sfx_path},
                created_at=timestamp,
            )
            self._update_job_status_conn(conn, task["job_id"], timestamp)
            conn.execute("COMMIT")

    def complete_sfx_separation(self, task: dict[str, Any], prompt: str, response: dict[str, Any]) -> None:
        timestamp = now_iso()
        target = response.get("target") or {}
        residual = response.get("residual") or {}
        isolated_path = self.preferred_audio_path(target)
        remaining_path = self.preferred_audio_path(residual)
        if not isolated_path:
            raise RuntimeError(f"SAM-Audio SFX response missing target audio path: {response}")
        if not remaining_path:
            raise RuntimeError(f"SAM-Audio SFX response missing residual audio path: {response}")

        # Check if the isolated SFX target actually has meaningful audio.
        # If it's empty/silent, the separation produced nothing — stop the loop.
        isolated_gate = self.sound_gate(Path(isolated_path))

        payload = json_loads(task["payload_json"], {})
        iteration = int(payload.get("iteration") or 1)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")

            if not isolated_gate["has_sound"]:
                # Isolated target is empty — separation failed to extract anything.
                # Don't store the empty artifact. Stop the loop here.
                self._insert_artifact(
                    conn,
                    job_id=task["job_id"],
                    chunk_id=task["chunk_id"],
                    task_id=task["id"],
                    kind=ARTIFACT_SFX_LOOP_DEBUG,
                    path=isolated_path,
                    text=f"SFX extraction produced empty target at iteration {iteration}. Stopping.",
                    prompt=prompt,
                    metadata={
                        "reason": "target_empty",
                        "iteration": iteration,
                        "gate": isolated_gate,
                    },
                    created_at=timestamp,
                )
                conn.execute(
                    "UPDATE chunks SET stage = ?, error = NULL, updated_at = ? WHERE id = ?",
                    (STAGE_SFX_EXHAUSTED, timestamp, task["chunk_id"]),
                )
                self._complete_task_conn(conn, task["id"], response, timestamp)
                self._insert_event(
                    conn,
                    job_id=task["job_id"],
                    chunk_id=task["chunk_id"],
                    task_id=task["id"],
                    level="info",
                    message=f"SFX loop stopped: isolated target empty at iteration {iteration}",
                    data={"prompt": prompt, "gate": isolated_gate},
                    created_at=timestamp,
                )
                self._update_job_status_conn(conn, task["job_id"], timestamp)
                conn.execute("COMMIT")
                return

            isolated_artifact_id = self._insert_artifact(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                kind=ARTIFACT_SFX_ISOLATED_TRACK,
                path=isolated_path,
                prompt=prompt,
                metadata={
                    "role": "target",
                    "iteration": iteration,
                    "selected_effect": payload.get("selected_effect"),
                    "sfx_list_artifact_id": payload.get("sfx_list_artifact_id"),
                    "source_artifact_id": payload.get("source_artifact_id"),
                    "refs": target,
                    "response": response,
                    "gate": isolated_gate,
                },
                created_at=timestamp,
            )

            remaining_artifact_id = self._insert_artifact(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                kind=ARTIFACT_SFX_REMAINING_TRACK,
                path=remaining_path,
                prompt="residual after sfx",
                metadata={
                    "role": "residual",
                    "iteration": iteration,
                    "selected_effect": payload.get("selected_effect"),
                    "sfx_list_artifact_id": payload.get("sfx_list_artifact_id"),
                    "source_artifact_id": payload.get("source_artifact_id"),
                    "isolated_artifact_id": isolated_artifact_id,
                    "refs": residual,
                    "response": response,
                },
                created_at=timestamp,
            )
            self._insert_task(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                queue=TASK_SOUND_GATE,
                payload={
                    "purpose": PURPOSE_GATE_REMAINING_SFX,
                    "track_type": "remaining_sfx",
                    "audio_path": remaining_path,
                    "artifact_id": remaining_artifact_id,
                    "isolated_artifact_id": isolated_artifact_id,
                    "iteration": iteration,
                },
                created_at=timestamp,
            )
            conn.execute(
                "UPDATE chunks SET stage = ?, error = NULL, updated_at = ? WHERE id = ?",
                (STAGE_GATE_REMAINING_SFX, timestamp, task["chunk_id"]),
            )
            self._complete_task_conn(conn, task["id"], response, timestamp)
            self._insert_event(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                level="info",
                message="SAM-Audio separated one SFX candidate",
                data={
                    "prompt": prompt,
                    "iteration": iteration,
                    "isolated_path": isolated_path,
                    "remaining_path": remaining_path,
                },
                created_at=timestamp,
            )
            self._update_job_status_conn(conn, task["job_id"], timestamp)
            conn.execute("COMMIT")

    def fail_task(self, task: dict[str, Any], error: str) -> None:
        timestamp = now_iso()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, error = ?, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (STATUS_FAILED, error, timestamp, timestamp, task["id"]),
            )
            if task.get("chunk_id"):
                conn.execute(
                    "UPDATE chunks SET stage = ?, error = ?, updated_at = ? WHERE id = ?",
                    (STAGE_FAILED, error, timestamp, task["chunk_id"]),
                )
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (STATUS_FAILED, error, timestamp, task["job_id"]),
            )
            self._insert_event(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task["id"],
                level="error",
                message="Task failed",
                data={"error": error, "queue": task["queue"]},
                created_at=timestamp,
            )
            conn.execute("COMMIT")

    def retry_task(self, task_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if task is None:
                conn.execute("ROLLBACK")
                raise KeyError(task_id)
            if task["status"] != STATUS_FAILED:
                conn.execute("ROLLBACK")
                raise ValueError(f"Only failed tasks can be retried; task is {task['status']}")

            payload = json_loads(task["payload_json"], {})
            purpose = payload.get("purpose")
            stage = {
                PURPOSE_CHUNK_SOUND_GATE: STAGE_SOUND_GATE,
                PURPOSE_DESCRIBE_SCENE: STAGE_DESCRIBE_SCENE,
                PURPOSE_SEPARATE_MUSIC: STAGE_SEPARATE_MUSIC,
                PURPOSE_GATE_MUSIC: STAGE_GATE_MUSIC,
                PURPOSE_GATE_SFX_VOICE: STAGE_GATE_SFX_VOICE,
                PURPOSE_DESCRIBE_MUSIC: STAGE_DESCRIBE_MUSIC,
                PURPOSE_SEPARATE_VOICES: STAGE_SEPARATE_VOICES,
                PURPOSE_GATE_VOICE: STAGE_GATE_VOICE,
                PURPOSE_GATE_SFX: STAGE_GATE_SFX,
                PURPOSE_TRANSCRIBE_VOICE: STAGE_TRANSCRIBE_VOICE,
                PURPOSE_LIST_SFX: STAGE_LIST_SFX,
                PURPOSE_GATE_REMAINING_SFX: STAGE_GATE_REMAINING_SFX,
                PURPOSE_DESCRIBE_SFX: STAGE_DESCRIBE_SFX,
                PURPOSE_SEPARATE_SFX: STAGE_SEPARATE_SFX,
            }.get(
                purpose,
                {
                    TASK_SOUND_GATE: STAGE_SOUND_GATE,
                    TASK_AUDIO_FLAMINGO: STAGE_DESCRIBE_SFX,
                    TASK_SAM_AUDIO: STAGE_SEPARATE_SFX,
                }.get(task["queue"], STAGE_FAILED),
            )
            timestamp = now_iso()
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, error = NULL, result_json = NULL, started_at = NULL,
                    completed_at = NULL, updated_at = ?
                WHERE id = ?
                """,
                (STATUS_PENDING, timestamp, task_id),
            )
            if task["chunk_id"]:
                conn.execute(
                    "UPDATE chunks SET stage = ?, error = NULL, updated_at = ? WHERE id = ?",
                    (stage, timestamp, task["chunk_id"]),
                )
            conn.execute(
                "UPDATE jobs SET status = ?, error = NULL, updated_at = ? WHERE id = ?",
                ("queued", timestamp, task["job_id"]),
            )
            self._insert_event(
                conn,
                job_id=task["job_id"],
                chunk_id=task["chunk_id"],
                task_id=task_id,
                level="info",
                message="Task queued for retry",
                data={"queue": task["queue"]},
                created_at=timestamp,
            )
            conn.execute("COMMIT")
        return {"task_id": task_id, "status": STATUS_PENDING}

    def _complete_task_conn(self, conn: sqlite3.Connection, task_id: str, result: dict[str, Any], timestamp: str) -> None:
        conn.execute(
            """
            UPDATE tasks
            SET status = ?, result_json = ?, error = NULL, updated_at = ?, completed_at = ?
            WHERE id = ?
            """,
            (STATUS_COMPLETED, json_dumps(result), timestamp, timestamp, task_id),
        )

    def _update_job_status_conn(self, conn: sqlite3.Connection, job_id: str, timestamp: str) -> None:
        counts = {
            row["status"]: row["count"]
            for row in conn.execute(
                "SELECT status, COUNT(*) AS count FROM tasks WHERE job_id = ? GROUP BY status",
                (job_id,),
            ).fetchall()
        }
        if counts.get(STATUS_FAILED, 0):
            status = STATUS_FAILED
        elif counts.get(STATUS_RUNNING, 0):
            status = "running"
        elif counts.get(STATUS_PENDING, 0):
            status = "queued"
        else:
            status = "complete"

        conn.execute(
            "UPDATE jobs SET status = ?, error = CASE WHEN ? != ? THEN NULL ELSE error END, updated_at = ? WHERE id = ?",
            (status, status, STATUS_FAILED, timestamp, job_id),
        )

    @staticmethod
    def parse_timestamp(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    @staticmethod
    def elapsed_seconds(started_at: str | None, completed_at: str | None) -> float | None:
        started = PipelineRuntime.parse_timestamp(started_at)
        completed = PipelineRuntime.parse_timestamp(completed_at)
        if not started or not completed:
            return None
        return max(0.001, (completed - started).total_seconds())

    @staticmethod
    def purpose_for_task(queue: str, payload: dict[str, Any]) -> str:
        purpose = payload.get("purpose")
        if purpose:
            return str(purpose)
        return {
            TASK_SOUND_GATE: PURPOSE_CHUNK_SOUND_GATE,
            TASK_AUDIO_FLAMINGO: PURPOSE_DESCRIBE_SFX,
            TASK_SAM_AUDIO: PURPOSE_SEPARATE_SFX,
        }.get(queue, queue)

    def audio_duration_seconds_for_path(self, audio_path: str | None) -> float | None:
        if not audio_path:
            return None
        try:
            resolved = str(Path(audio_path).expanduser().resolve())
        except Exception:
            return None
        with self._audio_duration_cache_lock:
            if resolved in self._audio_duration_cache:
                return self._audio_duration_cache[resolved]
        path = Path(resolved)
        if not path.is_file():
            return None
        try:
            duration = len(AudioSegment.from_file(str(path))) / 1000.0
        except Exception:
            return None
        with self._audio_duration_cache_lock:
            self._audio_duration_cache[resolved] = duration
        return duration

    def task_audio_duration_seconds(self, row: sqlite3.Row, payload: dict[str, Any], result: dict[str, Any] | None) -> float | None:
        if row["chunk_duration_ms"] is not None:
            return max(0.0, float(row["chunk_duration_ms"]) / 1000.0)
        if isinstance(result, dict) and result.get("duration_ms") is not None:
            return max(0.0, float(result["duration_ms"]) / 1000.0)
        return self.audio_duration_seconds_for_path(payload.get("audio_path") or row["source_audio_path"])

    @staticmethod
    def empty_performance_metric() -> dict[str, Any]:
        return {
            "completed_tasks": 0,
            "audio_seconds": 0.0,
            "wall_seconds": 0.0,
            "audio_seconds_per_minute": 0.0,
            "realtime_factor": 0.0,
            "avg_task_seconds": 0.0,
            "avg_audio_seconds": 0.0,
        }

    @classmethod
    def add_performance_sample(cls, metric: dict[str, Any], *, audio_seconds: float, wall_seconds: float) -> None:
        metric["completed_tasks"] += 1
        metric["audio_seconds"] += audio_seconds
        metric["wall_seconds"] += wall_seconds

    @classmethod
    def finalize_performance_metric(cls, metric: dict[str, Any]) -> dict[str, Any]:
        completed = int(metric["completed_tasks"])
        audio_seconds = float(metric["audio_seconds"])
        wall_seconds = float(metric["wall_seconds"])
        audio_seconds_per_minute = (audio_seconds / wall_seconds * 60.0) if wall_seconds > 0 else 0.0
        realtime_factor = (audio_seconds / wall_seconds) if wall_seconds > 0 else 0.0
        avg_task_seconds = (wall_seconds / completed) if completed else 0.0
        avg_audio_seconds = (audio_seconds / completed) if completed else 0.0
        return {
            "completed_tasks": completed,
            "audio_seconds": round(audio_seconds, 3),
            "wall_seconds": round(wall_seconds, 3),
            "audio_seconds_per_minute": round(audio_seconds_per_minute, 3),
            "realtime_factor": round(realtime_factor, 3),
            "avg_task_seconds": round(avg_task_seconds, 3),
            "avg_audio_seconds": round(avg_audio_seconds, 3),
        }

    def performance_summary(self, conn: sqlite3.Connection) -> dict[str, Any]:
        overall = self.empty_performance_metric()
        queues = {queue: self.empty_performance_metric() for queue in TASK_QUEUES}
        purposes = {purpose: self.empty_performance_metric() for purpose in PURPOSES}

        rows = conn.execute(
            """
            SELECT
                t.queue,
                t.payload_json,
                t.result_json,
                t.started_at,
                t.completed_at,
                c.duration_ms AS chunk_duration_ms,
                j.source_audio_path AS source_audio_path
            FROM tasks t
            LEFT JOIN chunks c ON c.id = t.chunk_id
            LEFT JOIN jobs j ON j.id = t.job_id
            WHERE t.status = ?
              AND t.started_at IS NOT NULL
              AND t.completed_at IS NOT NULL
            """,
            (STATUS_COMPLETED,),
        ).fetchall()

        for row in rows:
            wall_seconds = self.elapsed_seconds(row["started_at"], row["completed_at"])
            if wall_seconds is None:
                continue
            payload = json_loads(row["payload_json"], {})
            result = json_loads(row["result_json"], None)
            audio_seconds = self.task_audio_duration_seconds(row, payload, result)
            if audio_seconds is None or audio_seconds <= 0:
                continue
            purpose = self.purpose_for_task(row["queue"], payload)
            queues.setdefault(row["queue"], self.empty_performance_metric())
            purposes.setdefault(purpose, self.empty_performance_metric())
            self.add_performance_sample(overall, audio_seconds=audio_seconds, wall_seconds=wall_seconds)
            self.add_performance_sample(queues[row["queue"]], audio_seconds=audio_seconds, wall_seconds=wall_seconds)
            self.add_performance_sample(purposes[purpose], audio_seconds=audio_seconds, wall_seconds=wall_seconds)

        return {
            "overall": self.finalize_performance_metric(overall),
            "queues": {name: self.finalize_performance_metric(metric) for name, metric in queues.items()},
            "purposes": {name: self.finalize_performance_metric(metric) for name, metric in purposes.items()},
        }

    def default_scene_prompt(self, override: str | None) -> str:
        if override and override.strip():
            return override.strip()
        return (
            "Listen to the whole audio and answer in exactly this format:\n"
            "HAS_MUSIC: true or false\n"
            "HAS_VOICES: true or false\n"
            "MUSIC_DESCRIPTION: <if HAS_MUSIC is true, describe the music in a short phrase: "
            "genre, instruments, tempo, mood. Example: 'upbeat electronic dance music with synth pads and four-on-the-floor drums'. "
            "If no music, write 'none'>\n"
            "DESCRIPTION: <one sentence describing the full audio scene>\n"
            "Reply with nothing else."
        )

    def default_audio_flamingo_prompt(self, job_id: str) -> str:
        with self.connect() as conn:
            row = conn.execute("SELECT prompt FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row and row["prompt"]:
            return row["prompt"]
        return "Identify audible sources and return exactly two lines: SOUNDS: <sources>; SAM_PROMPT: <one target sound only>."

    def default_prompt_for_audio_flamingo_purpose(self, purpose: str, job_id: str) -> str:
        if purpose == PURPOSE_DESCRIBE_SCENE:
            return self.default_scene_prompt(None)
        if purpose == PURPOSE_DESCRIBE_MUSIC:
            return self.default_music_description_prompt()
        if purpose == PURPOSE_TRANSCRIBE_VOICE:
            return self.default_voice_transcription_prompt()
        if purpose == PURPOSE_LIST_SFX:
            return self.default_sfx_list_prompt()
        return self.default_audio_flamingo_prompt(job_id)

    @staticmethod
    def default_music_description_prompt() -> str:
        return (
            "Describe the music in this audio. Include style or genre, instrumentation, mood, "
            "tempo, and whether it is foreground or background."
        )

    @staticmethod
    def default_voice_transcription_prompt() -> str:
        return (
            "Transcribe any speech in this audio with speaker labels and diarization when possible. "
            "If there is no speech, say so."
        )

    @staticmethod
    def default_sfx_list_prompt() -> str:
        return (
            "List audible sound effects in this audio. Return strict JSON only, with no markdown, "
            'using this shape: {"effects":[{"label":"...", "sam_prompt":"...", "explanation":"..."}]}. '
            "Use a concise SAM prompt for one separable sound effect."
        )

    @staticmethod
    def amplitude_dbfs(amplitude: float, max_possible_amplitude: float) -> float:
        if amplitude <= 0 or max_possible_amplitude <= 0:
            return float("-inf")
        return 20.0 * math.log10(amplitude / max_possible_amplitude)

    @staticmethod
    def finite_dbfs(value: float) -> float | None:
        return round(value, 2) if math.isfinite(value) else None

    def sound_gate(self, audio_path: Path) -> dict[str, Any]:
        audio = AudioSegment.from_file(str(audio_path)).set_channels(1)
        duration_ms = len(audio)
        max_possible = float(audio.max_possible_amplitude)
        overall_dbfs = self.amplitude_dbfs(float(audio.rms), max_possible)
        peak_dbfs = self.amplitude_dbfs(float(audio.max), max_possible)
        window_ms = max(10, self.config.sound_gate_window_ms)
        active_ms = 0
        active_windows = 0
        loudest_window_dbfs = float("-inf")
        loudest_window_peak_dbfs = float("-inf")

        for start_ms in range(0, duration_ms, window_ms):
            window = audio[start_ms : min(start_ms + window_ms, duration_ms)]
            if len(window) <= 0:
                continue
            window_dbfs = self.amplitude_dbfs(float(window.rms), max_possible)
            window_peak_dbfs = self.amplitude_dbfs(float(window.max), max_possible)
            loudest_window_dbfs = max(loudest_window_dbfs, window_dbfs)
            loudest_window_peak_dbfs = max(loudest_window_peak_dbfs, window_peak_dbfs)
            if (
                window_dbfs >= self.config.sound_gate_min_dbfs
                and window_peak_dbfs >= self.config.sound_gate_min_peak_dbfs
            ):
                active_ms += len(window)
                active_windows += 1

        active_ratio = active_ms / duration_ms if duration_ms else 0.0
        has_peak = peak_dbfs >= self.config.sound_gate_min_peak_dbfs
        has_enough_level = overall_dbfs >= self.config.sound_gate_min_dbfs or loudest_window_dbfs >= self.config.sound_gate_min_dbfs
        has_enough_active_audio = (
            active_ms >= self.config.sound_gate_min_active_ms
            and active_ratio >= self.config.sound_gate_min_active_ratio
        )
        has_sound = duration_ms > 0 and has_peak and has_enough_level and has_enough_active_audio

        return {
            "has_sound": has_sound,
            "duration_ms": duration_ms,
            "rms": int(audio.rms),
            "peak": int(audio.max),
            "dbfs": self.finite_dbfs(overall_dbfs),
            "peak_dbfs": self.finite_dbfs(peak_dbfs),
            "active_ms": active_ms,
            "active_ratio": round(active_ratio, 4),
            "active_windows": active_windows,
            "loudest_window_dbfs": self.finite_dbfs(loudest_window_dbfs),
            "loudest_window_peak_dbfs": self.finite_dbfs(loudest_window_peak_dbfs),
            "thresholds": {
                "min_dbfs": self.config.sound_gate_min_dbfs,
                "min_peak_dbfs": self.config.sound_gate_min_peak_dbfs,
                "window_ms": window_ms,
                "min_active_ms": self.config.sound_gate_min_active_ms,
                "min_active_ratio": self.config.sound_gate_min_active_ratio,
            },
        }

    def mock_sound_gate(self, audio_path: Path) -> dict[str, Any]:
        return self.sound_gate(audio_path)

    def mock_audio_flamingo(self, audio_path: Path, prompt: str) -> dict[str, str]:
        prompt_lower = prompt.lower()
        if "whole audio scene" in prompt_lower or "full audio scene" in prompt_lower:
            text = "A mock outdoor scene with rhythmic horse hooves, light ambient noise, and cinematic music."
        elif "describe the music" in prompt_lower or "music in this audio" in prompt_lower:
            text = "MUSIC: mock cinematic strings with a steady pulse, warm mood, and background-score placement."
        elif "transcribe" in prompt_lower or "diarization" in prompt_lower or "speaker labels" in prompt_lower:
            text = "SPEAKER_1: Mock diarized transcription for the audible voice track."
        elif "strict json" in prompt_lower and "effects" in prompt_lower:
            text = json_dumps(
                {
                    "effects": [
                        {
                            "label": "mock horse hooves",
                            "sam_prompt": "horse hooves",
                            "explanation": "A rhythmic hoofbeat-like sound remains in the SFX track.",
                        }
                    ]
                }
            )
        else:
            text = "SOUNDS: horse hooves, cinematic strings\nSAM_PROMPT: horse hooves"
        return {
            "model_id": "mock/audio-flamingo-next",
            "audio_path": str(audio_path),
            "prompt": prompt,
            "text": text,
        }

    def mock_sam_audio(self, audio_path: Path, prompt: str, output_prefix: str, job_id: str | None = None) -> dict[str, Any]:
        request_id = new_id()
        if job_id:
            output_dir = self.config.output_dir / "pipeline" / "jobs" / job_id / "stems" / request_id
        else:
            output_dir = self.config.output_dir / "pipeline" / "mock" / request_id
        output_dir.mkdir(parents=True, exist_ok=False)

        target_wav = output_dir / f"{output_prefix}_target.wav"
        residual_wav = output_dir / f"{output_prefix}_residual.wav"
        zip_path = output_dir / f"{output_prefix}_outputs.zip"
        shutil.copyfile(audio_path, target_wav)
        shutil.copyfile(audio_path, residual_wav)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.write(target_wav, arcname=target_wav.name)
            zip_file.write(residual_wav, arcname=residual_wav.name)

        return {
            "model_id": "mock/sam-audio-large",
            "request_id": request_id,
            "audio_path": str(audio_path),
            "description": prompt,
            "target": {"wav": path_ref(str(target_wav), self.config.output_dir)},
            "residual": {"wav": path_ref(str(residual_wav), self.config.output_dir)},
            "zip": path_ref(str(zip_path), self.config.output_dir),
        }

    @staticmethod
    def preferred_audio_path(refs: dict[str, Any]) -> str | None:
        for key in ("wav", "mp3"):
            value = refs.get(key)
            if isinstance(value, dict) and value.get("path"):
                return str(value["path"])
        if refs.get("path"):
            return str(refs["path"])
        return None

    def extract_sam_prompt(self, text: str) -> str:
        match = re.search(r"SAM_PROMPT\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            raise ValueError(f"Audio Flamingo response missing SAM_PROMPT: {text}")
        prompt = match.group(1).strip().splitlines()[0].strip().strip("\"'` .;")
        if not prompt:
            raise ValueError(f"Audio Flamingo response had an empty SAM_PROMPT: {text}")
        return prompt[:180]

    def parse_sfx_list_text(self, text: str) -> tuple[dict[str, Any] | None, str | None]:
        candidate = text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            candidate = fenced.group(1).strip()
        elif "{" in candidate and "}" in candidate:
            candidate = candidate[candidate.find("{") : candidate.rfind("}") + 1]

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return None, f"Invalid JSON: {exc.msg}"

        effects = parsed.get("effects") if isinstance(parsed, dict) else None
        if not isinstance(effects, list):
            return None, "Expected JSON object with an effects list"

        normalized_effects: list[dict[str, str]] = []
        for effect in effects:
            if not isinstance(effect, dict):
                continue
            label = str(effect.get("label") or "").strip()
            sam_prompt = str(effect.get("sam_prompt") or "").strip()
            explanation = str(effect.get("explanation") or "").strip()
            normalized_effects.append(
                {
                    "label": label,
                    "sam_prompt": sam_prompt,
                    "explanation": explanation,
                }
            )
        return {"effects": normalized_effects}, None

    def chunk(self, chunk_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row is None:
            raise KeyError(chunk_id)
        return dict(row)

    def job_detail(self, job_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(job_id)
            chunks = [dict(row) for row in conn.execute("SELECT * FROM chunks WHERE job_id = ? ORDER BY chunk_index", (job_id,))]
            tasks = [self._task_row(row) for row in conn.execute("SELECT * FROM tasks WHERE job_id = ? ORDER BY created_at", (job_id,))]
            stems = [self._stem_row(row) for row in conn.execute("SELECT * FROM stems WHERE job_id = ? ORDER BY created_at DESC", (job_id,))]
            artifacts = [
                self._artifact_row(row)
                for row in conn.execute("SELECT * FROM artifacts WHERE job_id = ? ORDER BY created_at DESC", (job_id,))
            ]
            events = [
                self._event_row(row)
                for row in conn.execute("SELECT * FROM events WHERE job_id = ? ORDER BY created_at DESC, id DESC LIMIT 100", (job_id,))
            ]
        return {"job": dict(job), "chunks": chunks, "tasks": tasks, "stems": stems, "artifacts": artifacts, "events": events}

    def dashboard_summary(self) -> dict[str, Any]:
        with self.connect() as conn:
            totals = {
                "jobs": conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0],
                "chunks": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
                "stems": conn.execute("SELECT COUNT(*) FROM stems").fetchone()[0],
                "artifacts": conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0],
            }
            job_counts = {"queued": 0, "running": 0, "complete": 0, STATUS_FAILED: 0}
            for row in conn.execute("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"):
                job_counts[row["status"]] = row["count"]

            task_counts = {status: 0 for status in (STATUS_PENDING, STATUS_RUNNING, STATUS_FAILED, STATUS_COMPLETED)}
            for row in conn.execute("SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"):
                task_counts[row["status"]] = row["count"]

            stage_counts = {stage: 0 for stage in STAGES}
            for row in conn.execute("SELECT stage, COUNT(*) AS count FROM chunks GROUP BY stage"):
                stage_counts[row["stage"]] = row["count"]

            queue_counts = {
                queue: {STATUS_PENDING: 0, STATUS_RUNNING: 0, STATUS_FAILED: 0, STATUS_COMPLETED: 0}
                for queue in TASK_QUEUES
            }
            for row in conn.execute("SELECT queue, status, COUNT(*) AS count FROM tasks GROUP BY queue, status"):
                queue_counts.setdefault(row["queue"], {})[row["status"]] = row["count"]

            purpose_counts = {
                purpose: {STATUS_PENDING: 0, STATUS_RUNNING: 0, STATUS_FAILED: 0, STATUS_COMPLETED: 0}
                for purpose in PURPOSES
            }
            for row in conn.execute("SELECT queue, status, payload_json FROM tasks"):
                payload = json_loads(row["payload_json"], {})
                purpose = payload.get("purpose")
                if not purpose:
                    purpose = {
                        TASK_SOUND_GATE: PURPOSE_CHUNK_SOUND_GATE,
                        TASK_AUDIO_FLAMINGO: PURPOSE_DESCRIBE_SFX,
                        TASK_SAM_AUDIO: PURPOSE_SEPARATE_SFX,
                    }.get(row["queue"], row["queue"])
                purpose_counts.setdefault(
                    purpose,
                    {STATUS_PENDING: 0, STATUS_RUNNING: 0, STATUS_FAILED: 0, STATUS_COMPLETED: 0},
                )
                purpose_counts[purpose][row["status"]] = purpose_counts[purpose].get(row["status"], 0) + 1

            recent_jobs = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT
                        j.*,
                        SUM(CASE WHEN c.stage IN (
                            'complete', 'skipped_silent', 'music_ready', 'music_described',
                            'sfx_voice_ready', 'voice_transcribed', 'sfx_ready',
                            'sfx_exhausted', 'sfx_iteration_limit', 'sfx_loop_failed',
                            'skipped_music', 'skipped_sfx_voice', 'skipped_voice', 'skipped_sfx'
                        ) THEN 1 ELSE 0 END) AS complete_chunks,
                        SUM(CASE WHEN c.stage = 'failed' THEN 1 ELSE 0 END) AS failed_chunks
                    FROM jobs j
                    LEFT JOIN chunks c ON c.job_id = j.id
                    GROUP BY j.id
                    ORDER BY j.created_at DESC
                    LIMIT 20
                    """
                )
            ]
            failures = [
                self._task_row(row)
                for row in conn.execute(
                    """
                    SELECT t.*, c.chunk_index, c.audio_path AS chunk_audio_path
                    FROM tasks t
                    LEFT JOIN chunks c ON c.id = t.chunk_id
                    WHERE t.status = ?
                    ORDER BY t.updated_at DESC
                    LIMIT 20
                    """,
                    (STATUS_FAILED,),
                )
            ]
            outputs = [
                self._stem_row(row)
                for row in conn.execute(
                    """
                    SELECT s.*, c.chunk_index
                    FROM stems s
                    JOIN chunks c ON c.id = s.chunk_id
                    ORDER BY s.created_at DESC
                    LIMIT 20
                    """
                )
            ]
            artifacts = [
                self._artifact_row(row)
                for row in conn.execute(
                    """
                    SELECT a.*, c.chunk_index
                    FROM artifacts a
                    LEFT JOIN chunks c ON c.id = a.chunk_id
                    ORDER BY a.created_at DESC
                    LIMIT 30
                    """
                )
            ]
            performance = self.performance_summary(conn)
        return {
            "backend": self.config.backend,
            "storage_backend": self.storage.backend,
            "db_path": str(self.config.db_path),
            "output_dir": str(self.config.output_dir),
            "totals": totals,
            "jobs": job_counts,
            "tasks": task_counts,
            "stages": stage_counts,
            "queues": queue_counts,
            "purposes": purpose_counts,
            "performance": performance,
            "recent_jobs": recent_jobs,
            "recent_failures": failures,
            "recent_outputs": artifacts,
            "recent_stems": outputs,
        }

    @staticmethod
    def _ref_for_format(refs: dict[str, Any], audio_format: str) -> dict[str, Any] | None:
        value = refs.get(audio_format)
        if isinstance(value, dict):
            return dict(value) | {"format": audio_format}
        return None

    def preferred_artifact_audio_ref(self, artifact: dict[str, Any]) -> dict[str, Any] | None:
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        refs = metadata.get("refs") if isinstance(metadata, dict) and isinstance(metadata.get("refs"), dict) else {}
        preferred = self._ref_for_format(refs, "mp3") or self._ref_for_format(refs, "wav")
        if preferred:
            return preferred

        path_ref_value = artifact.get("path_ref")
        if isinstance(path_ref_value, dict):
            suffix = Path(str(path_ref_value.get("path") or "")).suffix.lower().lstrip(".")
            return dict(path_ref_value) | {"format": suffix or "audio"}
        return None

    @staticmethod
    def sound_label_for_artifact(artifact: dict[str, Any]) -> str:
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        selected_effect = metadata.get("selected_effect") if isinstance(metadata, dict) else None
        if artifact.get("kind") == ARTIFACT_SFX_ISOLATED_TRACK and isinstance(selected_effect, dict):
            label = str(selected_effect.get("label") or "").strip()
            if label:
                return label
        return {
            ARTIFACT_MUSIC_TRACK: "music (STFT masked)",
            ARTIFACT_MUSIC_TRACK_RAW: "music (raw model)",
            ARTIFACT_SFX_VOICE_TRACK: "sfx+voice",
            ARTIFACT_VOICE_TRACK: "voice (STFT masked)",
            ARTIFACT_VOICE_TRACK_RAW: "voice (raw model)",
            ARTIFACT_SFX_TRACK: "sfx",
            ARTIFACT_SFX_ISOLATED_TRACK: str(artifact.get("prompt") or "isolated sfx").strip() or "isolated sfx",
            ARTIFACT_SFX_REMAINING_TRACK: "remaining sfx",
        }.get(str(artifact.get("kind") or ""), str(artifact.get("kind") or "audio"))

    def sound_row(self, row: sqlite3.Row) -> dict[str, Any]:
        chunk = dict(row)
        audio = path_ref(chunk.get("audio_path"), self.config.output_dir)
        if audio:
            suffix = Path(str(audio.get("path") or "")).suffix.lower().lstrip(".")
            audio = dict(audio) | {"format": suffix or "audio"}
        return {
            "artifact_id": None,
            "job_id": chunk["job_id"],
            "chunk_id": chunk["id"],
            "chunk_index": chunk["chunk_index"],
            "start_ms": chunk["start_ms"],
            "end_ms": chunk["end_ms"],
            "duration_ms": chunk["duration_ms"],
            "sound": f"chunk {chunk['chunk_index']}",
            "kind": "source_chunk",
            "stage": chunk["stage"],
            "prompt": chunk.get("job_prompt"),
            "text": None,
            "created_at": chunk["created_at"],
            "updated_at": chunk["updated_at"],
            "audio": audio,
        }

    def sounds(self, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        safe_limit = max(1, min(limit, 500))
        safe_offset = max(0, offset)
        with self.connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            rows = conn.execute(
                """
                SELECT
                    c.*,
                    j.prompt AS job_prompt,
                    j.source_audio_path
                FROM chunks c
                JOIN jobs j ON j.id = c.job_id
                ORDER BY c.created_at DESC, c.chunk_index DESC
                LIMIT ? OFFSET ?
                """,
                (safe_limit, safe_offset),
            ).fetchall()
        return {
            "limit": safe_limit,
            "offset": safe_offset,
            "total": total,
            "rows": [self.sound_row(row) for row in rows],
        }

    @staticmethod
    def local_path_for_ref(ref: dict[str, Any] | None) -> Path | None:
        if not isinstance(ref, dict):
            return None
        raw = ref.get("local_path") or ref.get("path")
        if not raw or str(raw).startswith("s3://"):
            return None
        return Path(str(raw)).expanduser().resolve()

    @staticmethod
    def waveform_peaks(audio_path: Path, *, max_points: int = 600) -> dict[str, Any]:
        audio = AudioSegment.from_file(str(audio_path)).set_channels(1)
        samples = audio.get_array_of_samples()
        if not samples:
            return {"duration_ms": len(audio), "sample_rate": audio.frame_rate, "peaks": []}

        max_possible = float(audio.max_possible_amplitude) or 1.0
        bucket_size = max(1, math.ceil(len(samples) / max_points))
        peaks = []
        for start in range(0, len(samples), bucket_size):
            window = samples[start : start + bucket_size]
            peak = max(abs(sample) for sample in window) / max_possible if window else 0.0
            peaks.append(round(min(1.0, peak), 4))
        return {"duration_ms": len(audio), "sample_rate": audio.frame_rate, "peaks": peaks}

    @staticmethod
    def waveform_lane_order(lane: dict[str, Any]) -> tuple[int, int, str]:
        if lane["kind"] == "source_chunk":
            return (0, 0, lane["id"])
        order = {
            ARTIFACT_MUSIC_TRACK: 10,
            ARTIFACT_MUSIC_TRACK_RAW: 11,
            ARTIFACT_SFX_VOICE_TRACK: 20,
            ARTIFACT_VOICE_TRACK: 30,
            ARTIFACT_VOICE_TRACK_RAW: 31,
            ARTIFACT_SFX_TRACK: 40,
            ARTIFACT_SFX_ISOLATED_TRACK: 50,
            ARTIFACT_SFX_REMAINING_TRACK: 51,
        }.get(lane["kind"], 90)
        metadata = lane.get("metadata") if isinstance(lane.get("metadata"), dict) else {}
        iteration = int(metadata.get("iteration") or 0)
        return (order, iteration, lane["id"])

    def waveform_lane(self, *, lane_id: str, kind: str, label: str, audio: dict[str, Any] | None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        lane = {
            "id": lane_id,
            "kind": kind,
            "label": label,
            "audio": audio,
            "metadata": metadata or {},
            "waveform": None,
            "waveform_unavailable": None,
        }
        local_path = self.local_path_for_ref(audio)
        if local_path is None:
            lane["waveform_unavailable"] = "no local audio path"
            return lane
        if not local_path.is_file():
            lane["waveform_unavailable"] = f"audio file not found: {local_path}"
            return lane
        try:
            lane["waveform"] = self.waveform_peaks(local_path)
        except Exception as exc:
            lane["waveform_unavailable"] = exception_detail(exc)
        return lane

    def chunk_waveforms(self, chunk_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            chunk = conn.execute(
                """
                SELECT
                    c.*,
                    j.source_audio_path,
                    j.prompt AS job_prompt,
                    j.status AS job_status
                FROM chunks c
                JOIN jobs j ON j.id = c.job_id
                WHERE c.id = ?
                """,
                (chunk_id,),
            ).fetchone()
            if chunk is None:
                raise KeyError(chunk_id)
            artifact_rows = [
                self._artifact_row(row)
                for row in conn.execute(
                    f"""
                    SELECT a.*
                    FROM artifacts a
                    WHERE a.chunk_id = ?
                      AND a.kind IN ({",".join("?" for _ in SOUND_AUDIO_ARTIFACT_KINDS)})
                    ORDER BY a.created_at ASC
                    """,
                    (chunk_id, *SOUND_AUDIO_ARTIFACT_KINDS),
                )
            ]

        chunk_data = dict(chunk)
        chunk_audio = path_ref(chunk_data.get("audio_path"), self.config.output_dir)
        lanes = [
            self.waveform_lane(
                lane_id=f"{chunk_id}:source",
                kind="source_chunk",
                label=f"chunk {chunk_data['chunk_index']} source",
                audio=chunk_audio,
                metadata={
                    "start_ms": chunk_data["start_ms"],
                    "end_ms": chunk_data["end_ms"],
                    "duration_ms": chunk_data["duration_ms"],
                },
            )
        ]
        # Only include target/stem tracks, not residuals.
        _RESIDUAL_KINDS = {ARTIFACT_SFX_VOICE_TRACK, ARTIFACT_SFX_TRACK, ARTIFACT_SFX_REMAINING_TRACK}
        for artifact in artifact_rows:
            if artifact.get("kind") in _RESIDUAL_KINDS:
                continue
            audio = self.preferred_artifact_audio_ref(artifact)
            lanes.append(
                self.waveform_lane(
                    lane_id=artifact["id"],
                    kind=artifact["kind"],
                    label=self.sound_label_for_artifact(artifact),
                    audio=audio,
                    metadata={
                        "artifact_id": artifact["id"],
                        "prompt": artifact.get("prompt"),
                        "text": artifact.get("text"),
                        **(artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}),
                    },
                )
            )

        lanes.sort(key=self.waveform_lane_order)
        return {
            "chunk": {
                "id": chunk_data["id"],
                "job_id": chunk_data["job_id"],
                "chunk_index": chunk_data["chunk_index"],
                "audio_path": chunk_data["audio_path"],
                "start_ms": chunk_data["start_ms"],
                "end_ms": chunk_data["end_ms"],
                "duration_ms": chunk_data["duration_ms"],
                "stage": chunk_data["stage"],
                "error": chunk_data["error"],
                "job_prompt": chunk_data["job_prompt"],
                "job_status": chunk_data["job_status"],
            },
            "lanes": lanes,
        }

    def _task_row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["payload"] = json_loads(data.pop("payload_json", None), {})
        data["result"] = json_loads(data.pop("result_json", None), None)
        return data

    def _stem_row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["result"] = json_loads(data.pop("result_json", None), None)
        data["target"] = {
            "wav": path_ref(data.get("target_wav"), self.config.output_dir),
            "mp3": path_ref(data.get("target_mp3"), self.config.output_dir),
        }
        data["residual"] = {
            "wav": path_ref(data.get("residual_wav"), self.config.output_dir),
            "mp3": path_ref(data.get("residual_mp3"), self.config.output_dir),
        }
        data["zip"] = path_ref(data.get("zip_path"), self.config.output_dir)
        return data

    def _artifact_row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["metadata"] = json_loads(data.pop("metadata_json", None), {})
        data["storage_ref"] = data["metadata"].get("storage") if isinstance(data["metadata"], dict) else None
        data["local_path_ref"] = path_ref(data.get("path"), self.config.output_dir)
        data["path_ref"] = data["storage_ref"] or data["local_path_ref"]
        return data

    def _event_row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["data"] = json_loads(data.pop("data_json", None), {})
        return data


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QLabeler Pipeline</title>
  <style>
    :root {
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1d2430;
      --muted: #606a7c;
      --line: #d9dee7;
      --soft-line: #edf0f5;
      --red: #ef4444;
      --blue: #2f7de1;
      --green: #23a455;
      --orange: #f08a00;
      --slate: #657084;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); }
    header { padding: 18px 28px; background: var(--panel); border-bottom: 1px solid var(--line); }
    h1 { margin: 0; font-size: 24px; font-weight: 650; letter-spacing: 0; }
    h2 { margin: 0; font-size: 16px; font-weight: 650; letter-spacing: 0; }
    main { padding: 20px 28px 40px; display: grid; gap: 18px; }
    section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; word-break: break-all; }
    button { padding: 9px 12px; border: 1px solid #1f5eff; background: #1f5eff; color: #fff; border-radius: 6px; font-weight: 650; cursor: pointer; }
    button.secondary { border-color: #c7cfdb; background: #fff; color: var(--ink); }
    input { min-width: min(560px, 100%); flex: 1; padding: 9px 10px; border: 1px solid #c7cfdb; border-radius: 6px; font-size: 13px; }
    input[type="file"] { background: #fff; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 9px 8px; border-bottom: 1px solid var(--soft-line); text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 650; background: #fafbfc; }
    .meta { margin-top: 6px; color: var(--muted); font-size: 13px; }
    .section-head { display: flex; gap: 12px; align-items: center; justify-content: space-between; padding: 14px 16px; border-bottom: 1px solid var(--soft-line); }
    .section-head h2 { white-space: nowrap; }
    .top-nav { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
    .top-nav a { padding: 7px 10px; border: 1px solid var(--line); border-radius: 6px; color: var(--ink); text-decoration: none; font-size: 13px; font-weight: 650; background: #fff; }
    .top-nav a.active { border-color: #1f5eff; color: #1f5eff; background: #eff6ff; }
    .submit-section { padding: 14px 16px; }
    .submit-section form { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    .status-strip { display: flex; gap: 8px; flex-wrap: wrap; }
    .status-pill { display: inline-flex; gap: 7px; align-items: center; padding: 5px 9px; border: 1px solid var(--line); border-radius: 999px; background: #fff; color: var(--muted); font-size: 12px; }
    .status-pill strong { color: var(--ink); font-size: 13px; }
    .flow-board { padding: 0; }
    .flow-wrap { overflow-x: auto; background: linear-gradient(#fff, #fbfcfd); }
    .flow-graph { display: block; width: 100%; min-width: 0; height: auto; max-height: 700px; }
    .flow-title { font-size: 15px; font-weight: 700; fill: var(--ink); }
    .flow-subtitle { font-size: 12px; fill: var(--muted); }
    .flow-small { font-size: 11px; fill: var(--muted); }
    .flow-edge-label { font-size: 11px; font-weight: 650; fill: var(--red); }
    .flow-edge-label.good { fill: var(--green); }
    .arrow { fill: none; stroke: var(--red); stroke-width: 2.1; marker-end: url(#arrow); }
    .arrow.soft { stroke-dasharray: 5 6; opacity: 0.75; }
    .arrow.green { stroke: var(--green); }
    .flow-node { cursor: pointer; outline: none; }
    .flow-node .node-shape { fill: #fff; stroke-width: 2; transition: filter 120ms ease, stroke-width 120ms ease; }
    .flow-node:hover .node-shape,
    .flow-node.selected .node-shape { filter: drop-shadow(0 5px 10px rgba(31, 41, 55, 0.14)); stroke-width: 3; }
    .kind-input .node-shape { stroke: var(--red); }
    .kind-chunk .node-shape { stroke: #303746; }
    .kind-gate .node-shape { stroke: var(--blue); }
    .kind-model .node-shape { stroke: var(--orange); stroke-dasharray: 3 4; }
    .kind-work .node-shape { stroke: var(--green); }
    .kind-db .node-shape { stroke: var(--red); }
    .kind-terminal .node-shape { stroke: var(--slate); }
    .kind-failed .node-shape { stroke: var(--red); }
    .count-badge .badge-bg { fill: #eef2f7; stroke: #cbd3df; stroke-width: 1; }
    .count-badge .badge-text { fill: var(--ink); font-size: 14px; font-weight: 800; text-anchor: middle; dominant-baseline: middle; }
    .count-badge.active .badge-bg { fill: #fff7ed; stroke: var(--orange); }
    .count-badge.running .badge-bg { fill: #eff6ff; stroke: var(--blue); }
    .count-badge.failed .badge-bg { fill: #fff1f2; stroke: var(--red); }
    .node-inspector { display: grid; grid-template-columns: minmax(180px, 0.8fr) minmax(240px, 1.4fr); gap: 14px; padding: 14px 16px; border-top: 1px solid var(--soft-line); }
    .inspector-title { font-weight: 700; font-size: 15px; }
    .inspector-kind { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .inspector-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 8px; }
    .metric { border: 1px solid var(--line); border-radius: 7px; padding: 8px 9px; background: #fbfcfd; min-height: 58px; }
    .metric span { display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; }
    .metric strong { display: block; margin-top: 3px; font-size: 21px; line-height: 1; }
    .tables { display: grid; grid-template-columns: minmax(0, 1fr); gap: 18px; }
    .table-section { padding: 16px; }
    .two { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }
    .sounds-view, .chunk-view { display: none; }
    body[data-view="sounds"] .overview-view, body[data-view="chunk"] .overview-view { display: none; }
    body[data-view="sounds"] .sounds-view, body[data-view="chunk"] .chunk-view { display: block; }
    .clickable-row { cursor: pointer; }
    .clickable-row:hover td { background: #f5f8fc; }
    .view-empty, .view-error, .view-loading { color: var(--muted); padding: 14px 0; font-size: 13px; }
    .view-error { color: #c53232; }
    .chunk-meta { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
    .waveform-list { display: grid; gap: 8px; margin-top: 14px; }
    .waveform-lane { display: grid; grid-template-columns: minmax(110px, 170px) minmax(240px, 1fr); gap: 12px; align-items: stretch; border: 1px solid var(--line); border-radius: 7px; background: #fbfcfd; padding: 10px; }
    .waveform-label { min-width: 0; }
    .waveform-title { font-weight: 700; font-size: 13px; }
    .waveform-subtitle { margin-top: 4px; color: var(--muted); font-size: 12px; word-break: break-word; }
    .waveform-panel { min-width: 0; }
    .waveform-scale { display: flex; justify-content: space-between; color: var(--muted); font-size: 11px; margin-bottom: 4px; }
    .waveform-canvas { width: 100%; height: 68px; display: block; border: 1px solid var(--soft-line); border-radius: 6px; background: #fff; }
    audio { width: 100%; max-width: 520px; height: 32px; margin-top: 6px; }
    .status-failed { color: #c53232; font-weight: 650; }
    .status-complete, .status-completed { color: #18733f; font-weight: 650; }
    .status-running { color: #9a5a00; font-weight: 650; }
    /* DAW multitrack styles */
    .daw-transport { display: flex; align-items: center; gap: 10px; margin-top: 14px; padding: 8px 12px; background: #1a1a2e; border-radius: 6px; }
    .daw-btn { background: #2a2a4a; color: #fff; border: 1px solid #444; border-radius: 4px; padding: 6px 14px; font-size: 13px; cursor: pointer; font-family: inherit; }
    .daw-btn:hover { background: #3a3a5a; }
    .daw-time { color: #6ddf6d; font-family: 'SF Mono', monospace; font-size: 13px; margin-left: 8px; }
    .daw-tracks { display: flex; flex-direction: column; gap: 0; margin-top: 0; border: 1px solid #333; border-radius: 6px; overflow: hidden; }
    .daw-track { display: grid; grid-template-columns: 180px 1fr; border-bottom: 1px solid #2a2a3a; background: #1e1e2e; }
    .daw-track:last-child { border-bottom: none; }
    .daw-track-source { background: #1e2a1e; }
    .daw-track-header { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-right: 1px solid #333; }
    .daw-mute-btn { background: #2a4a2a; color: #6ddf6d; border: 1px solid #3a5a3a; border-radius: 4px; width: 32px; height: 32px; cursor: pointer; font-size: 16px; display: flex; align-items: center; justify-content: center; }
    .daw-mute-btn.muted { background: #4a2a2a; color: #df6d6d; border-color: #5a3a3a; }
    .daw-track-info { min-width: 0; }
    .daw-track-title { font-weight: 700; font-size: 12px; color: #e0e0e0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .daw-track-title:hover { white-space: normal; overflow: visible; word-break: break-word; }
    .daw-track-kind { font-size: 11px; color: #888; }
    .daw-track-prompt { font-size: 11px; color: #aad; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 130px; }
    .daw-track-prompt:hover { white-space: normal; overflow: visible; max-width: none; word-break: break-word; }
    .daw-download-btn { display: flex; align-items: center; justify-content: center; width: 28px; height: 28px; margin-left: auto; background: #2a3a4a; color: #8cf; border-radius: 4px; text-decoration: none; font-size: 14px; border: 1px solid #3a5a6a; }
    .daw-download-btn:hover { background: #3a4a5a; }
    .daw-track-waveform { position: relative; padding: 4px 0; cursor: pointer; }
    .daw-track-waveform .waveform-canvas { width: 100%; height: 48px; display: block; border: none; border-radius: 0; background: transparent; }
    .daw-playhead { position: absolute; top: 0; bottom: 0; width: 2px; background: #ff4444; left: 0; pointer-events: none; transition: left 0.05s linear; }
    .daw-track audio { display: none; }
    @media (max-width: 720px) {
      body { min-width: 360px; }
      main, header { padding-left: 14px; padding-right: 14px; }
      .section-head { align-items: flex-start; flex-direction: column; }
      .node-inspector { grid-template-columns: 1fr; }
      .waveform-lane { grid-template-columns: 1fr; }
      .daw-track { grid-template-columns: 120px 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>QLabeler Pipeline</h1>
    <div class="meta" id="meta">Loading...</div>
    <nav class="top-nav" aria-label="Dashboard views">
      <a href="#overview" data-nav="overview">Overview</a>
      <a href="#sounds" data-nav="sounds">Chunks</a>
    </nav>
  </header>
  <main>
    <section class="submit-section overview-view">
      <form id="job-form">
        <input id="audio-file" type="file" accept="audio/*,.mp3,.wav,.flac,.m4a" required>
        <input id="prompt" placeholder="Optional Audio Flamingo prompt">
        <button type="submit">Queue</button>
      </form>
    </section>

    <section class="flow-board overview-view">
      <div class="section-head">
        <h2>Pipeline Graph</h2>
        <div class="status-strip" id="summary"></div>
      </div>
      <div class="flow-wrap">
        <svg id="flow-graph" class="flow-graph" viewBox="0 0 2300 760" role="img" aria-label="Pipeline status graph">
          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L9,3 z" fill="#ef4444"></path>
            </marker>
          </defs>

          <path class="arrow" d="M100 265 C120 265 130 265 150 265"></path>
          <path class="arrow" d="M340 265 C352 265 360 265 374 265"></path>
          <path class="arrow" d="M506 265 C534 265 556 262 590 262"></path>
          <path class="arrow green" d="M780 238 C812 218 824 192 846 170"></path>
          <path class="arrow green" d="M780 302 C812 324 824 346 846 360"></path>
          <path class="arrow soft" d="M100 236 C140 166 184 118 245 104"></path>
          <path class="arrow soft" d="M425 104 C760 42 1280 56 1648 142"></path>
          <path class="arrow soft" d="M980 170 C1016 156 1030 140 1052 134"></path>
          <path class="arrow green" d="M1198 144 C1210 148 1214 158 1218 168"></path>
          <path class="arrow soft" d="M980 360 C1018 364 1036 368 1058 372"></path>
          <path class="arrow soft" d="M946 242 C995 334 1032 396 1065 438"></path>
          <path class="arrow soft" d="M946 432 C982 454 1018 462 1058 466"></path>
          <path class="arrow soft" d="M446 370 C590 472 874 492 1058 466"></path>
          <path class="arrow soft" d="M1198 372 C1244 392 1284 420 1322 440"></path>
          <text class="flow-edge-label good" x="1190" y="135">music track</text>
          <text class="flow-edge-label" x="620" y="466">empty</text>
          <text class="flow-edge-label" x="1000" y="344">empty</text>
          <text class="flow-edge-label" x="992" y="488">empty</text>
          <path class="arrow green" d="M1198 134 C1214 126 1228 119 1242 112"></path>
          <path class="arrow soft" d="M1438 112 C1518 116 1582 130 1648 154"></path>
          <path class="arrow green" d="M1198 372 C1220 372 1234 372 1256 372"></path>
          <path class="arrow green" d="M1438 342 C1464 318 1480 300 1504 278"></path>
          <path class="arrow green" d="M1438 402 C1464 426 1480 444 1504 466"></path>
          <path class="arrow" d="M1596 278 C1622 270 1640 292 1658 320"></path>
          <path class="arrow soft" d="M1768 320 C1762 280 1746 246 1718 226"></path>
          <path class="arrow soft" d="M1596 466 C1635 456 1672 452 1712 450"></path>
          <path class="arrow soft" d="M1550 348 C1440 454 1298 486 1200 466"></path>
          <path class="arrow soft" d="M1550 536 C1420 560 1280 526 1200 466"></path>
          <text class="flow-edge-label good" x="1450" y="102">music description</text>
          <text class="flow-edge-label good" x="1442" y="318">voice track</text>
          <text class="flow-edge-label good" x="1448" y="452">sfx track</text>
          <text class="flow-edge-label good" x="1620" y="304">voice transcription</text>
          <text class="flow-edge-label" x="1468" y="372">empty</text>
          <text class="flow-edge-label" x="1458" y="554">empty</text>
          <path class="arrow green" d="M1830 450 C1850 446 1872 440 1900 430"></path>
          <path class="arrow green" d="M2090 400 C2102 400 2110 400 2122 400"></path>
          <path class="arrow" d="M2208 458 C2202 478 2198 494 2193 512"></path>
          <path class="arrow green" d="M2128 590 C2040 588 1978 536 1978 462"></path>
          <path class="arrow soft" d="M2128 624 C1780 690 1400 620 1198 466"></path>
          <text class="flow-edge-label good" x="1846" y="430">sfx track</text>
          <text class="flow-edge-label" x="2042" y="384">pick first sfx</text>
          <text class="flow-edge-label" x="2050" y="568">remaining sfx</text>
          <text class="flow-edge-label" x="1880" y="586">repeat</text>
          <text class="flow-edge-label" x="1768" y="672">empty/exhausted</text>

          <g class="flow-node kind-input selected" tabindex="0" data-node="source">
            <rect class="node-shape" x="30" y="210" width="70" height="110" rx="14"></rect>
            <text class="flow-title" x="65" y="255" text-anchor="middle">Audio</text>
            <text class="flow-title" x="65" y="276" text-anchor="middle">track</text>
            <g class="count-badge" data-badge="source" transform="translate(104 205)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-model" tabindex="0" data-node="describe_scene">
            <rect class="node-shape" x="245" y="50" width="180" height="108" rx="12"></rect>
            <text class="flow-title" x="335" y="92" text-anchor="middle">describe whole</text>
            <text class="flow-title" x="335" y="114" text-anchor="middle">scene</text>
            <text class="flow-small" data-node-meta="describe_scene" x="335" y="178" text-anchor="middle">waiting 0 · running 0</text>
            <g class="count-badge" data-badge="describe_scene" transform="translate(428 48)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-chunk" tabindex="0" data-node="chunks">
            <rect class="node-shape" x="150" y="200" width="190" height="130" rx="13"></rect>
            <text class="flow-title" x="245" y="256" text-anchor="middle">30s chunks</text>
            <text class="flow-subtitle" x="245" y="282" text-anchor="middle">5s overlap</text>
            <g class="count-badge" data-badge="chunks" transform="translate(340 196)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-gate" tabindex="0" data-node="sound_gate">
            <polygon class="node-shape" points="440,160 520,265 440,370 360,265"></polygon>
            <text class="flow-title" x="440" y="246" text-anchor="middle">sound</text>
            <text class="flow-title" x="440" y="268" text-anchor="middle">gate</text>
            <text class="flow-subtitle" x="440" y="292" text-anchor="middle">filter</text>
            <text class="flow-small" data-node-meta="sound_gate" x="440" y="396" text-anchor="middle">waiting 0 · running 0</text>
            <g class="count-badge" data-badge="sound_gate" transform="translate(516 176)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-work" tabindex="0" data-node="separate_music">
            <rect class="node-shape" x="590" y="190" width="190" height="145" rx="12"></rect>
            <text class="flow-title" x="685" y="253" text-anchor="middle">separate music</text>
            <text class="flow-subtitle" x="685" y="282" text-anchor="middle">sam_audio</text>
            <text class="flow-small" data-node-meta="separate_music" x="685" y="356" text-anchor="middle">waiting 0 · running 0</text>
            <g class="count-badge" data-badge="separate_music" transform="translate(782 186)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-gate" tabindex="0" data-node="gate_music">
            <polygon class="node-shape" points="910,80 980,170 910,260 840,170"></polygon>
            <text class="flow-title" x="910" y="152" text-anchor="middle">music</text>
            <text class="flow-subtitle" x="910" y="174" text-anchor="middle">sound gate</text>
            <text class="flow-small" data-node-meta="gate_music" x="910" y="282" text-anchor="middle">waiting 0 · running 0</text>
            <g class="count-badge" data-badge="gate_music" transform="translate(976 92)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-gate" tabindex="0" data-node="gate_sfx_voice">
            <polygon class="node-shape" points="910,270 980,360 910,450 840,360"></polygon>
            <text class="flow-title" x="910" y="342" text-anchor="middle">sfx+voice</text>
            <text class="flow-subtitle" x="910" y="364" text-anchor="middle">sound gate</text>
            <text class="flow-small" data-node-meta="gate_sfx_voice" x="910" y="472" text-anchor="middle">waiting 0 · running 0</text>
            <g class="count-badge" data-badge="gate_sfx_voice" transform="translate(976 282)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-db" tabindex="0" data-node="artifacts_db">
            <circle class="node-shape" cx="1720" cy="158" r="72"></circle>
            <text class="flow-title" x="1720" y="124" text-anchor="middle">ARTIFACTS</text>
            <text class="flow-title" x="1720" y="146" text-anchor="middle">DB</text>
            <text class="flow-subtitle" x="1720" y="174" text-anchor="middle">scene + track</text>
            <text class="flow-subtitle" x="1720" y="195" text-anchor="middle">refs</text>
            <g class="count-badge" data-badge="artifacts_db" transform="translate(1788 100)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-terminal" tabindex="0" data-node="music_ready">
            <rect class="node-shape" x="1052" y="92" width="146" height="84" rx="12"></rect>
            <text class="flow-title" x="1125" y="128" text-anchor="middle">music</text>
            <text class="flow-subtitle" x="1125" y="152" text-anchor="middle">ready</text>
            <g class="count-badge" data-badge="music_ready" transform="translate(1198 88)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-model" tabindex="0" data-node="describe_music">
            <rect class="node-shape" x="1240" y="58" width="198" height="108" rx="12"></rect>
            <text class="flow-title" x="1339" y="102" text-anchor="middle">describe music</text>
            <text class="flow-subtitle" x="1339" y="130" text-anchor="middle">audio_flamingo_next</text>
            <text class="flow-small" data-node-meta="describe_music" x="1339" y="186" text-anchor="middle">waiting 0 · running 0</text>
            <g class="count-badge" data-badge="describe_music" transform="translate(1438 54)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-terminal" tabindex="0" data-node="sfx_voice_ready">
            <rect class="node-shape" x="1058" y="330" width="140" height="84" rx="12"></rect>
            <text class="flow-title" x="1128" y="365" text-anchor="middle">sfx+voice</text>
            <text class="flow-subtitle" x="1128" y="390" text-anchor="middle">ready</text>
            <g class="count-badge" data-badge="sfx_voice_ready" transform="translate(1198 326)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-work" tabindex="0" data-node="separate_voices">
            <rect class="node-shape" x="1256" y="302" width="182" height="140" rx="12"></rect>
            <text class="flow-title" x="1347" y="362" text-anchor="middle">separate voices</text>
            <text class="flow-subtitle" x="1347" y="390" text-anchor="middle">sam_audio</text>
            <text class="flow-small" data-node-meta="separate_voices" x="1347" y="462" text-anchor="middle">waiting 0 · running 0</text>
            <g class="count-badge" data-badge="separate_voices" transform="translate(1438 298)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-gate" tabindex="0" data-node="gate_voice">
            <polygon class="node-shape" points="1550,198 1616,278 1550,358 1484,278"></polygon>
            <text class="flow-title" x="1550" y="263" text-anchor="middle">voice</text>
            <text class="flow-subtitle" x="1550" y="286" text-anchor="middle">sound gate</text>
            <text class="flow-small" data-node-meta="gate_voice" x="1550" y="380" text-anchor="middle">waiting 0 · running 0</text>
            <g class="count-badge" data-badge="gate_voice" transform="translate(1616 210)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-gate" tabindex="0" data-node="gate_sfx">
            <polygon class="node-shape" points="1550,386 1616,466 1550,546 1484,466"></polygon>
            <text class="flow-title" x="1550" y="451" text-anchor="middle">sfx</text>
            <text class="flow-subtitle" x="1550" y="474" text-anchor="middle">sound gate</text>
            <text class="flow-small" data-node-meta="gate_sfx" x="1550" y="568" text-anchor="middle">waiting 0 · running 0</text>
            <g class="count-badge" data-badge="gate_sfx" transform="translate(1616 398)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-model" tabindex="0" data-node="transcribe_voice">
            <rect class="node-shape" x="1658" y="286" width="210" height="108" rx="12"></rect>
            <text class="flow-title" x="1763" y="326" text-anchor="middle">transcribe</text>
            <text class="flow-title" x="1763" y="348" text-anchor="middle">with diarization</text>
            <text class="flow-subtitle" x="1763" y="374" text-anchor="middle">audio_flamingo_next</text>
            <text class="flow-small" data-node-meta="transcribe_voice" x="1763" y="414" text-anchor="middle">waiting 0 · running 0</text>
            <g class="count-badge" data-badge="transcribe_voice" transform="translate(1868 282)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-terminal" tabindex="0" data-node="sfx_ready">
            <rect class="node-shape" x="1712" y="408" width="118" height="84" rx="12"></rect>
            <text class="flow-title" x="1771" y="443" text-anchor="middle">sfx</text>
            <text class="flow-subtitle" x="1771" y="468" text-anchor="middle">ready</text>
            <g class="count-badge" data-badge="sfx_ready" transform="translate(1830 404)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-model" tabindex="0" data-node="list_sfx">
            <rect class="node-shape" x="1900" y="342" width="190" height="118" rx="12"></rect>
            <text class="flow-title" x="1995" y="389" text-anchor="middle">list sound</text>
            <text class="flow-title" x="1995" y="411" text-anchor="middle">effects</text>
            <text class="flow-subtitle" x="1995" y="438" text-anchor="middle">audio_flamingo_next</text>
            <text class="flow-small" data-node-meta="list_sfx" x="1995" y="480" text-anchor="middle">waiting 0 · running 0</text>
            <g class="count-badge" data-badge="list_sfx" transform="translate(2090 338)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-work" tabindex="0" data-node="separate_sfx">
            <rect class="node-shape" x="2122" y="342" width="170" height="116" rx="12"></rect>
            <text class="flow-title" x="2207" y="392" text-anchor="middle">separate sfx</text>
            <text class="flow-subtitle" x="2207" y="420" text-anchor="middle">sam_audio</text>
            <text class="flow-small" data-node-meta="separate_sfx" x="2207" y="478" text-anchor="middle">waiting 0 · running 0</text>
            <g class="count-badge" data-badge="separate_sfx" transform="translate(2270 338)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-gate" tabindex="0" data-node="gate_remaining_sfx">
            <polygon class="node-shape" points="2190,512 2255,590 2190,668 2125,590"></polygon>
            <text class="flow-title" x="2190" y="574" text-anchor="middle">remaining</text>
            <text class="flow-title" x="2190" y="596" text-anchor="middle">sfx</text>
            <text class="flow-subtitle" x="2190" y="618" text-anchor="middle">sound gate</text>
            <text class="flow-small" data-node-meta="gate_remaining_sfx" x="2190" y="690" text-anchor="middle">waiting 0 · running 0</text>
            <g class="count-badge" data-badge="gate_remaining_sfx" transform="translate(2255 526)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-terminal" tabindex="0" data-node="skipped">
            <rect class="node-shape" x="1058" y="430" width="140" height="70" rx="12"></rect>
            <text class="flow-title" x="1128" y="458" text-anchor="middle">skipped</text>
            <text class="flow-subtitle" x="1128" y="481" text-anchor="middle">empty audio</text>
            <g class="count-badge" data-badge="skipped" transform="translate(1198 426)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-failed" tabindex="0" data-node="failed">
            <rect class="node-shape" x="1776" y="590" width="86" height="56" rx="12"></rect>
            <text class="flow-title" x="1819" y="614" text-anchor="middle">failed</text>
            <text class="flow-subtitle" x="1819" y="636" text-anchor="middle">retry</text>
            <g class="count-badge" data-badge="failed" transform="translate(1862 586)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>
        </svg>
      </div>
      <div id="node-inspector" class="node-inspector"></div>
    </section>

    <div class="two overview-view">
      <section class="table-section">
        <h2>Stages</h2>
        <table id="stages"></table>
      </section>
      <section class="table-section">
        <h2>Queues</h2>
        <table id="queues"></table>
      </section>
    </div>

    <div class="tables overview-view">
      <section class="table-section">
        <h2>Recent Jobs</h2>
        <table id="jobs"></table>
      </section>
      <section class="table-section">
        <h2>Recent Failures</h2>
        <table id="failures"></table>
      </section>
      <section class="table-section">
        <h2>Recent Outputs</h2>
        <table id="outputs"></table>
      </section>
    </div>

    <section class="table-section sounds-view">
      <div class="section-head">
        <h2>Chunks</h2>
        <button class="secondary" type="button" id="sounds-refresh">Refresh</button>
      </div>
      <div id="sounds-state" class="view-loading">Loading...</div>
      <table id="sounds-table"></table>
    </section>

    <section class="table-section chunk-view">
      <div class="section-head">
        <h2>Chunk Detail</h2>
        <a href="#sounds">Back to chunks</a>
      </div>
      <div id="chunk-detail"></div>
    </section>
  </main>
  <script>
    const FLOW_NODES = {
      source: { title: 'Audio Track', kind: 'Input', badge: 'jobs' },
      describe_scene: { title: 'Describe Whole Scene', kind: 'Queue: audio_flamingo', badge: 'waiting' },
      chunks: { title: 'Chunk Splitter', kind: 'Preprocess', badge: 'chunks' },
      sound_gate: { title: 'Sound Gate Filter', kind: 'Queue: sound_gate', badge: 'waiting' },
      separate_music: { title: 'Separate Music', kind: 'Queue: sam_audio', badge: 'waiting' },
      gate_music: { title: 'Music Sound Gate', kind: 'Queue: sound_gate', badge: 'waiting' },
      gate_sfx_voice: { title: 'SFX+Voice Sound Gate', kind: 'Queue: sound_gate', badge: 'waiting' },
      describe_music: { title: 'Describe Music', kind: 'Queue: audio_flamingo', badge: 'waiting' },
      separate_voices: { title: 'Separate Voices', kind: 'Queue: sam_audio', badge: 'waiting' },
      gate_voice: { title: 'Voice Sound Gate', kind: 'Queue: sound_gate', badge: 'waiting' },
      gate_sfx: { title: 'SFX Sound Gate', kind: 'Queue: sound_gate', badge: 'waiting' },
      transcribe_voice: { title: 'Transcribe With Diarization', kind: 'Queue: audio_flamingo', badge: 'waiting' },
      list_sfx: { title: 'List Sound Effects', kind: 'Queue: audio_flamingo', badge: 'waiting' },
      separate_sfx: { title: 'Separate SFX', kind: 'Queue: sam_audio', badge: 'waiting' },
      gate_remaining_sfx: { title: 'Remaining SFX Sound Gate', kind: 'Queue: sound_gate', badge: 'waiting' },
      music_ready: { title: 'Music Ready', kind: 'Terminal stage', badge: 'chunks' },
      sfx_voice_ready: { title: 'SFX+Voice Ready', kind: 'Terminal stage', badge: 'chunks' },
      sfx_ready: { title: 'SFX Ready', kind: 'Terminal stage', badge: 'chunks' },
      artifacts_db: { title: 'Artifacts DB', kind: 'Output refs', badge: 'artifacts' },
      skipped: { title: 'Skipped Empty Audio', kind: 'Terminal stage', badge: 'chunks' },
      failed: { title: 'Failed', kind: 'Retryable work', badge: 'failures' },
    };
    let selectedNode = 'source';
    let latestData = null;
    let currentView = 'overview';
    let loadedSounds = false;
    let loadedChunkId = null;

    const statusClass = value => `status-${String(value || '').replaceAll('_', '-')}`;
    const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    const link = ref => ref && ref.url ? `<a href="${esc(ref.url)}"><code>${esc(ref.path)}</code></a>` : (ref && ref.path ? `<code>${esc(ref.path)}</code>` : '');
    const queue = (data, name) => data.queues[name] || { pending: 0, running: 0, failed: 0, completed: 0 };
    const purpose = (data, name) => data.purposes[name] || { pending: 0, running: 0, failed: 0, completed: 0 };
    const number = value => Number(value || 0);
    const active = counts => number(counts.pending) + number(counts.running);
    const NODE_PURPOSES = {
      describe_scene: 'describe_scene',
      sound_gate: 'chunk_sound_gate',
      separate_music: 'separate_music',
      gate_music: 'gate_music',
      gate_sfx_voice: 'gate_sfx_voice',
      describe_music: 'describe_music',
      separate_voices: 'separate_voices',
      gate_voice: 'gate_voice',
      gate_sfx: 'gate_sfx',
      transcribe_voice: 'transcribe_voice',
      list_sfx: 'list_sfx',
      separate_sfx: 'separate_sfx',
      gate_remaining_sfx: 'gate_remaining_sfx',
    };
    const perfPurpose = (data, name) => ((data.performance || {}).purposes || {})[name] || {};
    const perfQueue = (data, name) => ((data.performance || {}).queues || {})[name] || {};
    const perfOverall = data => ((data.performance || {}).overall || {});

    function formatAudioPerMinute(metric) {
      const value = number(metric && metric.audio_seconds_per_minute);
      if (!value) return '—';
      if (value >= 60) return `${(value / 60).toFixed(value >= 600 ? 0 : 1)}m/min`;
      return `${value.toFixed(value >= 10 ? 0 : 1)}s/min`;
    }

    function formatSeconds(value) {
      const seconds = number(value);
      if (!seconds) return '—';
      if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
      return `${seconds.toFixed(seconds >= 10 ? 1 : 2)}s`;
    }

    function formatMs(value) {
      const ms = number(value);
      if (!ms) return '0s';
      const seconds = ms / 1000;
      return seconds >= 60 ? `${(seconds / 60).toFixed(1)}m` : `${seconds.toFixed(seconds >= 10 ? 1 : 2)}s`;
    }

    function formatRealtime(metric) {
      const value = number(metric && metric.realtime_factor);
      return value ? `${value.toFixed(value >= 10 ? 1 : 2)}x` : '—';
    }

    function performanceForNode(data, nodeName) {
      if (nodeName === 'source' || nodeName === 'chunks') return perfOverall(data);
      const purposeName = NODE_PURPOSES[nodeName];
      return purposeName ? perfPurpose(data, purposeName) : {};
    }

    function performanceMetrics(metric) {
      return [
        ['Audio/min', formatAudioPerMinute(metric)],
        ['Avg task', formatSeconds(metric.avg_task_seconds)],
        ['Realtime', formatRealtime(metric)],
        ['Samples', metric.completed_tasks || 0],
      ];
    }

    function rows(headers, items, cells) {
      return `<thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>${items.map(item => `<tr>${cells(item).join('')}</tr>`).join('') || `<tr><td colspan="${headers.length}">No rows</td></tr>`}</tbody>`;
    }

    function flowStats(data) {
      const jobs = data.jobs || { queued: 0, running: 0, complete: 0, failed: 0 };
      const activeJobs = number(jobs.queued) + number(jobs.running);
      const scene = purpose(data, 'describe_scene');
      const chunkGate = purpose(data, 'chunk_sound_gate');
      const musicSplit = purpose(data, 'separate_music');
      const musicGate = purpose(data, 'gate_music');
      const sfxVoiceGate = purpose(data, 'gate_sfx_voice');
      const describeMusic = purpose(data, 'describe_music');
      const voiceSplit = purpose(data, 'separate_voices');
      const voiceGate = purpose(data, 'gate_voice');
      const sfxGate = purpose(data, 'gate_sfx');
      const voiceTranscribe = purpose(data, 'transcribe_voice');
      const listSfx = purpose(data, 'list_sfx');
      const sfxSplit = purpose(data, 'separate_sfx');
      const remainingSfxGate = purpose(data, 'gate_remaining_sfx');
      const terminalChunks =
        number(data.stages.complete) + number(data.stages.skipped_silent) +
        number(data.stages.music_ready) + number(data.stages.music_described) +
        number(data.stages.sfx_voice_ready) + number(data.stages.voice_transcribed) +
        number(data.stages.sfx_ready) + number(data.stages.sfx_exhausted) +
        number(data.stages.sfx_iteration_limit) + number(data.stages.sfx_loop_failed) +
        number(data.stages.skipped_music) +
        number(data.stages.skipped_sfx_voice) + number(data.stages.skipped_voice) +
        number(data.stages.skipped_sfx) +
        number(data.stages.failed);
      const activeChunks = Math.max(0, number(data.totals.chunks) - terminalChunks);
      return {
        source: {
          badge: activeJobs,
          running: number(jobs.running),
          failed: number(jobs.failed),
          metrics: [['Ongoing jobs', activeJobs], ['Queued jobs', jobs.queued], ['Running jobs', jobs.running], ['Failed jobs', jobs.failed]],
        },
        describe_scene: {
          badge: active(scene),
          waiting: number(scene.pending),
          running: number(scene.running),
          failed: number(scene.failed),
          done: number(scene.completed),
          metrics: [['Ongoing', active(scene)], ['Waiting', scene.pending], ['Running', scene.running], ['Completed', scene.completed]],
        },
        chunks: {
          badge: activeChunks,
          failed: number(data.stages.failed),
          metrics: [['Ongoing chunks', activeChunks], ['Waiting gate', data.stages.sound_gate], ['Separating music', data.stages.separate_music], ['Ready', data.stages.sfx_voice_ready]],
        },
        sound_gate: {
          badge: active(chunkGate),
          waiting: number(chunkGate.pending),
          running: number(chunkGate.running),
          failed: number(chunkGate.failed),
          done: number(chunkGate.completed),
          metrics: [['Ongoing', active(chunkGate)], ['Waiting', chunkGate.pending], ['Running', chunkGate.running], ['Failed', chunkGate.failed]],
        },
        separate_music: {
          badge: active(musicSplit),
          waiting: number(musicSplit.pending),
          running: number(musicSplit.running),
          failed: number(musicSplit.failed),
          done: number(musicSplit.completed),
          metrics: [['Ongoing', active(musicSplit)], ['Waiting', musicSplit.pending], ['Running', musicSplit.running], ['Failed', musicSplit.failed]],
        },
        gate_music: {
          badge: active(musicGate),
          waiting: number(musicGate.pending),
          running: number(musicGate.running),
          failed: number(musicGate.failed),
          done: number(musicGate.completed),
          metrics: [['Ongoing', active(musicGate)], ['Waiting', musicGate.pending], ['Running', musicGate.running], ['Failed', musicGate.failed]],
        },
        gate_sfx_voice: {
          badge: active(sfxVoiceGate),
          waiting: number(sfxVoiceGate.pending),
          running: number(sfxVoiceGate.running),
          failed: number(sfxVoiceGate.failed),
          done: number(sfxVoiceGate.completed),
          metrics: [['Ongoing', active(sfxVoiceGate)], ['Waiting', sfxVoiceGate.pending], ['Running', sfxVoiceGate.running], ['Failed', sfxVoiceGate.failed]],
        },
        describe_music: {
          badge: active(describeMusic),
          waiting: number(describeMusic.pending),
          running: number(describeMusic.running),
          failed: number(describeMusic.failed),
          done: number(describeMusic.completed),
          metrics: [['Ongoing', active(describeMusic)], ['Waiting', describeMusic.pending], ['Running', describeMusic.running], ['Completed', describeMusic.completed]],
        },
        separate_voices: {
          badge: active(voiceSplit),
          waiting: number(voiceSplit.pending),
          running: number(voiceSplit.running),
          failed: number(voiceSplit.failed),
          done: number(voiceSplit.completed),
          metrics: [['Ongoing', active(voiceSplit)], ['Waiting', voiceSplit.pending], ['Running', voiceSplit.running], ['Failed', voiceSplit.failed]],
        },
        gate_voice: {
          badge: active(voiceGate),
          waiting: number(voiceGate.pending),
          running: number(voiceGate.running),
          failed: number(voiceGate.failed),
          done: number(voiceGate.completed),
          metrics: [['Ongoing', active(voiceGate)], ['Waiting', voiceGate.pending], ['Running', voiceGate.running], ['Failed', voiceGate.failed]],
        },
        gate_sfx: {
          badge: active(sfxGate),
          waiting: number(sfxGate.pending),
          running: number(sfxGate.running),
          failed: number(sfxGate.failed),
          done: number(sfxGate.completed),
          metrics: [['Ongoing', active(sfxGate)], ['Waiting', sfxGate.pending], ['Running', sfxGate.running], ['Failed', sfxGate.failed]],
        },
        transcribe_voice: {
          badge: active(voiceTranscribe),
          waiting: number(voiceTranscribe.pending),
          running: number(voiceTranscribe.running),
          failed: number(voiceTranscribe.failed),
          done: number(voiceTranscribe.completed),
          metrics: [['Ongoing', active(voiceTranscribe)], ['Waiting', voiceTranscribe.pending], ['Running', voiceTranscribe.running], ['Completed', voiceTranscribe.completed]],
        },
        list_sfx: {
          badge: active(listSfx),
          waiting: number(listSfx.pending),
          running: number(listSfx.running),
          failed: number(listSfx.failed),
          done: number(listSfx.completed),
          metrics: [['Ongoing', active(listSfx)], ['Waiting', listSfx.pending], ['Running', listSfx.running], ['Completed', listSfx.completed]],
        },
        separate_sfx: {
          badge: active(sfxSplit),
          waiting: number(sfxSplit.pending),
          running: number(sfxSplit.running),
          failed: number(sfxSplit.failed),
          done: number(sfxSplit.completed),
          metrics: [['Ongoing', active(sfxSplit)], ['Waiting', sfxSplit.pending], ['Running', sfxSplit.running], ['Failed', sfxSplit.failed]],
        },
        gate_remaining_sfx: {
          badge: active(remainingSfxGate),
          waiting: number(remainingSfxGate.pending),
          running: number(remainingSfxGate.running),
          failed: number(remainingSfxGate.failed),
          done: number(remainingSfxGate.completed),
          metrics: [['Ongoing', active(remainingSfxGate)], ['Waiting', remainingSfxGate.pending], ['Running', remainingSfxGate.running], ['Failed', remainingSfxGate.failed]],
        },
        artifacts_db: {
          badge: 0,
          metrics: [['Ongoing writes', 0], ['Artifacts total', data.totals.artifacts], ['Recent outputs', data.recent_outputs.length]],
        },
        music_ready: {
          badge: 0,
          metrics: [['Ongoing', 0], ['Ready chunks', data.stages.music_ready], ['Described chunks', data.stages.music_described], ['Skipped music', data.stages.skipped_music]],
        },
        sfx_voice_ready: {
          badge: 0,
          metrics: [['Ongoing', 0], ['Ready chunks', data.stages.sfx_voice_ready], ['Skipped sfx+voice', data.stages.skipped_sfx_voice]],
        },
        sfx_ready: {
          badge: 0,
          metrics: [
            ['Ongoing', 0],
            ['Ready chunks', data.stages.sfx_ready],
            ['Loop exhausted', data.stages.sfx_exhausted],
            ['Iteration limit', data.stages.sfx_iteration_limit],
            ['Loop failed', data.stages.sfx_loop_failed],
          ],
        },
        skipped: {
          badge: 0,
          metrics: [
            ['Ongoing skips', 0],
            ['Skipped total', number(data.stages.skipped_silent) + number(data.stages.skipped_music) + number(data.stages.skipped_sfx_voice) + number(data.stages.skipped_voice) + number(data.stages.skipped_sfx) + number(data.stages.sfx_exhausted)],
            ['Chunk gate empty', data.stages.skipped_silent],
            ['Music empty', data.stages.skipped_music],
            ['SFX+voice empty', data.stages.skipped_sfx_voice],
            ['Voice empty', data.stages.skipped_voice],
            ['SFX empty', data.stages.skipped_sfx],
            ['Loop exhausted', data.stages.sfx_exhausted],
          ],
        },
        failed: {
          badge: number(data.tasks.failed),
          failed: number(data.tasks.failed),
          metrics: [['Failed tasks', data.tasks.failed], ['Failed chunks', data.stages.failed], ['Recent failures', data.recent_failures.length]],
        },
      };
    }

    function setBadge(name, value, variant) {
      const badge = document.querySelector(`[data-badge="${name}"]`);
      if (!badge) return;
      const text = badge.querySelector('text');
      const rect = badge.querySelector('rect');
      const display = String(value || 0);
      const width = Math.max(36, display.length * 10 + 22);
      text.textContent = display;
      rect.setAttribute('x', String(-width / 2));
      rect.setAttribute('width', String(width));
      badge.classList.toggle('active', number(value) > 0);
      badge.classList.toggle('running', variant === 'running');
      badge.classList.toggle('failed', variant === 'failed');
    }

    function renderGraph(data) {
      const stats = flowStats(data);
      Object.entries(stats).forEach(([name, values]) => {
        const variant = values.failed ? 'failed' : values.running ? 'running' : '';
        setBadge(name, values.badge, variant);
        const meta = document.querySelector(`[data-node-meta="${name}"]`);
        if (meta) {
          meta.textContent = `waiting ${values.waiting || 0} · running ${values.running || 0} · ${formatAudioPerMinute(performanceForNode(data, name))}`;
        }
      });
      document.querySelectorAll('.flow-node').forEach(node => {
        node.classList.toggle('selected', node.dataset.node === selectedNode);
      });
      renderInspector(data);
    }

    function renderInspector(data) {
      const stats = flowStats(data);
      const node = FLOW_NODES[selectedNode] || FLOW_NODES.source;
      const values = stats[selectedNode] || stats.source;
      const metrics = (values.metrics || []).concat(performanceMetrics(performanceForNode(data, selectedNode)));
      document.getElementById('node-inspector').innerHTML = `
        <div>
          <div class="inspector-title">${esc(node.title)}</div>
          <div class="inspector-kind">${esc(node.kind)}</div>
        </div>
        <div class="inspector-grid">
          ${metrics.map(([label, value]) => `<div class="metric"><span>${esc(label)}</span><strong>${value || 0}</strong></div>`).join('')}
        </div>`;
    }

    function renderTables(data) {
      document.getElementById('stages').innerHTML = rows(['Stage', 'Chunks'], Object.entries(data.stages), ([stage, count]) => [`<td><code>${esc(stage)}</code></td>`, `<td>${count}</td>`]);
      document.getElementById('queues').innerHTML = rows(['Queue', 'Pending', 'Running', 'Audio/min', 'Avg task', 'Failed', 'Completed'], Object.entries(data.queues), ([q, c]) => {
        const metric = perfQueue(data, q);
        return [`<td><code>${esc(q)}</code></td>`, `<td>${c.pending || 0}</td>`, `<td>${c.running || 0}</td>`, `<td>${esc(formatAudioPerMinute(metric))}</td>`, `<td>${esc(formatSeconds(metric.avg_task_seconds))}</td>`, `<td>${c.failed || 0}</td>`, `<td>${c.completed || 0}</td>`];
      });
      document.getElementById('jobs').innerHTML = rows(['Job', 'Status', 'Chunks', 'Source', 'Updated'], data.recent_jobs, j => [`<td><code>${esc(j.id)}</code></td>`, `<td class="${statusClass(j.status)}">${esc(j.status)}</td>`, `<td>${j.complete_chunks || 0}/${j.chunk_count || 0}</td>`, `<td><code>${esc(j.source_audio_path)}</code></td>`, `<td>${esc(j.updated_at)}</td>`]);
      document.getElementById('failures').innerHTML = rows(['Task', 'Queue', 'Chunk', 'Error', 'Retry'], data.recent_failures, f => [`<td><code>${esc(f.id)}</code></td>`, `<td><code>${esc(f.queue)}</code></td>`, `<td>${esc(f.chunk_index)}</td>`, `<td>${esc(f.error)}</td>`, `<td><button class="secondary" onclick="retryTask('${esc(f.id)}')">Retry</button></td>`]);
      document.getElementById('outputs').innerHTML = rows(['Kind', 'Chunk', 'Prompt/Text', 'File', 'Created'], data.recent_outputs, o => [`<td><code>${esc(o.kind)}</code></td>`, `<td>${esc(o.chunk_index || '')}</td>`, `<td>${esc(o.text || o.prompt || '')}</td>`, `<td>${link(o.path_ref)}</td>`, `<td>${esc(o.created_at)}</td>`]);
    }

    function renderSounds(data) {
      const state = document.getElementById('sounds-state');
      const table = document.getElementById('sounds-table');
      const rowsData = data.rows || [];
      state.textContent = rowsData.length ? `${rowsData.length} of ${data.total || rowsData.length} chunks` : '';
      state.className = rowsData.length ? 'meta' : 'view-empty';
      if (!rowsData.length) {
        table.innerHTML = '';
        state.textContent = 'No chunks yet.';
        return;
      }
      table.innerHTML = `
        <thead><tr><th>Chunk</th><th>Time</th><th>Prompt</th><th>Whole Audio</th><th>Stage</th><th>Created</th></tr></thead>
        <tbody>
          ${rowsData.map(row => `
            <tr class="clickable-row" data-chunk-id="${esc(row.chunk_id)}">
              <td><code>${esc(row.sound)}</code></td>
              <td>${formatMs(row.start_ms)}-${formatMs(row.end_ms)}</td>
              <td>${esc(row.prompt || row.text || '')}</td>
              <td>${link(row.audio)}${row.audio && row.audio.format ? `<div class="meta">${esc(row.audio.format)}</div>` : ''}</td>
              <td><code>${esc(row.stage || row.kind)}</code></td>
              <td>${esc(row.created_at)}</td>
            </tr>`).join('')}
        </tbody>`;
      table.querySelectorAll('[data-chunk-id]').forEach(row => {
        row.addEventListener('click', event => {
          const target = event.target;
          if (target && target.closest && target.closest('a')) return;
          location.hash = `#chunk/${row.dataset.chunkId}`;
        });
      });
    }

    async function loadSounds(force = false) {
      if (loadedSounds && !force) return;
      const state = document.getElementById('sounds-state');
      state.textContent = 'Loading...';
      state.className = 'view-loading';
      document.getElementById('sounds-table').innerHTML = '';
      try {
        const data = await fetch('/api/sounds?limit=200').then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        });
        renderSounds(data);
        loadedSounds = true;
      } catch (error) {
        state.textContent = `Could not load sounds: ${error.message || error}`;
        state.className = 'view-error';
      }
    }

    function laneAudio(lane) {
      const ref = lane.audio || {};
      const href = ref.url || '';
      return href;
    }

    function renderChunkDetail(data) {
      const chunk = data.chunk || {};
      const lanes = data.lanes || [];
      const detail = document.getElementById('chunk-detail');
      detail.innerHTML = `
        <div>
          <h2>Chunk ${esc(chunk.chunk_index || '')}</h2>
          <div class="chunk-meta">
            <span class="status-pill">Job <strong>${esc(chunk.job_id || '')}</strong></span>
            <span class="status-pill">Stage <strong>${esc(chunk.stage || '')}</strong></span>
            <span class="status-pill">Start <strong>${formatMs(chunk.start_ms)}</strong></span>
            <span class="status-pill">End <strong>${formatMs(chunk.end_ms)}</strong></span>
            <span class="status-pill">Duration <strong>${formatMs(chunk.duration_ms)}</strong></span>
          </div>
        </div>
        <div class="daw-transport">
          <button id="daw-play" class="daw-btn">&#9654; Play</button>
          <button id="daw-stop" class="daw-btn">&#9632; Stop</button>
          <span id="daw-time" class="daw-time">0:00.0 / ${formatMs(chunk.duration_ms)}</span>
        </div>
        <div class="daw-tracks" id="daw-tracks">
          ${lanes.map((lane, index) => {
            const href = laneAudio(lane);
            const waveform = lane.waveform || {};
            const isSource = lane.kind === 'source_chunk';
            return `
              <div class="daw-track ${isSource ? 'daw-track-source' : ''}" data-lane-index="${index}">
                <div class="daw-track-header">
                  <button class="daw-mute-btn ${isSource ? '' : 'muted'}" data-lane-index="${index}" title="Toggle mute">
                    ${isSource ? '&#128264;' : '&#128263;'}
                  </button>
                  <div class="daw-track-info">
                    <div class="daw-track-title">${esc(lane.label)}</div>
                    <div class="daw-track-kind"><code>${esc(lane.kind)}</code></div>
                    ${(lane.metadata || {}).prompt ? `<div class="daw-track-prompt">${esc(lane.metadata.prompt)}</div>` : ''}
                  </div>
                  ${href ? `<a class="daw-download-btn" href="${esc(href)}" download title="Download">&#11015;</a>` : ''}
                </div>
                <div class="daw-track-waveform">
                  <canvas class="waveform-canvas" data-lane-index="${index}" width="900" height="48"></canvas>
                  <div class="daw-playhead" data-lane-index="${index}"></div>
                </div>
                <audio preload="auto" src="${esc(href)}" data-lane-index="${index}" ${isSource ? '' : 'muted'}></audio>
              </div>`;
          }).join('') || '<div class="view-empty">No tracks.</div>'}
        </div>`;
      drawWaveforms(lanes);
      initDAW(lanes, chunk.duration_ms || 0);
    }

    function initDAW(lanes, durationMs) {
      const audios = Array.from(document.querySelectorAll('#daw-tracks audio'));
      const playBtn = document.getElementById('daw-play');
      const stopBtn = document.getElementById('daw-stop');
      const timeEl = document.getElementById('daw-time');
      const playheads = document.querySelectorAll('.daw-playhead');
      let playing = false;
      let rafId = null;

      function updateTime() {
        const master = audios[0];
        if (!master) return;
        const current = master.currentTime * 1000;
        timeEl.textContent = `${formatMs(Math.floor(current))} / ${formatMs(durationMs)}`;
        const pct = durationMs > 0 ? (current / durationMs) * 100 : 0;
        playheads.forEach(ph => { ph.style.left = `${Math.min(100, pct)}%`; });
        if (playing) rafId = requestAnimationFrame(updateTime);
      }

      function playAll() {
        audios.forEach(a => { a.currentTime = audios[0] ? audios[0].currentTime : 0; a.play(); });
        playing = true;
        playBtn.innerHTML = '&#10074;&#10074; Pause';
        updateTime();
      }

      function pauseAll() {
        audios.forEach(a => a.pause());
        playing = false;
        playBtn.innerHTML = '&#9654; Play';
        if (rafId) cancelAnimationFrame(rafId);
      }

      function stopAll() {
        pauseAll();
        audios.forEach(a => { a.currentTime = 0; });
        updateTime();
      }

      playBtn.addEventListener('click', () => { playing ? pauseAll() : playAll(); });
      stopBtn.addEventListener('click', stopAll);

      // Master ended
      if (audios[0]) {
        audios[0].addEventListener('ended', () => { pauseAll(); updateTime(); });
      }

      // Mute/unmute buttons
      document.querySelectorAll('.daw-mute-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const idx = parseInt(btn.dataset.laneIndex, 10);
          const audio = audios[idx];
          if (!audio) return;
          audio.muted = !audio.muted;
          btn.classList.toggle('muted', audio.muted);
          btn.innerHTML = audio.muted ? '&#128263;' : '&#128264;';
        });
      });

      // Click on waveform to seek
      document.querySelectorAll('.daw-track-waveform').forEach(wf => {
        wf.addEventListener('click', (e) => {
          const rect = wf.getBoundingClientRect();
          const pct = (e.clientX - rect.left) / rect.width;
          const seekTime = (pct * durationMs) / 1000;
          audios.forEach(a => { a.currentTime = seekTime; });
          updateTime();
        });
      });
    }

    function drawWaveforms(lanes) {
      document.querySelectorAll('.waveform-canvas').forEach(canvas => {
        const lane = lanes[number(canvas.dataset.laneIndex)] || {};
        const peaks = (lane.waveform || {}).peaks || [];
        const ratio = window.devicePixelRatio || 1;
        const cssWidth = Math.max(1, canvas.clientWidth || 900);
        const cssHeight = Math.max(1, canvas.clientHeight || 48);
        canvas.width = Math.floor(cssWidth * ratio);
        canvas.height = Math.floor(cssHeight * ratio);
        const ctx = canvas.getContext('2d');
        ctx.scale(ratio, ratio);
        ctx.clearRect(0, 0, cssWidth, cssHeight);
        // Dark background for DAW tracks
        const isDaw = canvas.closest('.daw-track');
        ctx.fillStyle = isDaw ? 'transparent' : '#ffffff';
        ctx.fillRect(0, 0, cssWidth, cssHeight);
        ctx.strokeStyle = isDaw ? '#333' : '#d9dee7';
        ctx.beginPath();
        ctx.moveTo(0, cssHeight / 2);
        ctx.lineTo(cssWidth, cssHeight / 2);
        ctx.stroke();
        if (!peaks.length) return;
        const barWidth = cssWidth / peaks.length;
        const isSource = lane.kind === 'source_chunk';
        ctx.fillStyle = isDaw ? (isSource ? '#4ddf4d' : '#5599ee') : '#2f7de1';
        peaks.forEach((peak, index) => {
          const height = Math.max(1, peak * (cssHeight - 8));
          const x = index * barWidth;
          const y = (cssHeight - height) / 2;
          ctx.fillRect(x, y, Math.max(1, barWidth - 1), height);
        });
      });
    }

    async function loadChunk(chunkId) {
      if (loadedChunkId === chunkId) return;
      const detail = document.getElementById('chunk-detail');
      loadedChunkId = chunkId;
      detail.innerHTML = '<div class="view-loading">Loading...</div>';
      try {
        const data = await fetch(`/api/chunks/${encodeURIComponent(chunkId)}/waveforms`).then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        });
        renderChunkDetail(data);
      } catch (error) {
        detail.innerHTML = `<div class="view-error">Could not load chunk: ${esc(error.message || error)}</div>`;
      }
    }

    function renderSummary(data) {
      const jobs = data.jobs || { queued: 0, running: 0, complete: 0, failed: 0 };
      const activeJobs = number(jobs.queued) + number(jobs.running);
      const terminalChunks =
        number(data.stages.complete) + number(data.stages.skipped_silent) +
        number(data.stages.music_ready) + number(data.stages.music_described) +
        number(data.stages.sfx_voice_ready) + number(data.stages.voice_transcribed) +
        number(data.stages.sfx_ready) + number(data.stages.sfx_exhausted) +
        number(data.stages.sfx_iteration_limit) + number(data.stages.sfx_loop_failed) +
        number(data.stages.skipped_music) +
        number(data.stages.skipped_sfx_voice) + number(data.stages.skipped_voice) +
        number(data.stages.skipped_sfx) +
        number(data.stages.failed);
      const activeChunks = Math.max(0, number(data.totals.chunks) - terminalChunks);
      const pills = [
        ['Total jobs', data.totals.jobs],
        ['Ongoing jobs', activeJobs],
        ['Ongoing chunks', activeChunks],
        ['Throughput', formatAudioPerMinute(perfOverall(data))],
        ['Pending', data.tasks.pending],
        ['Running', data.tasks.running],
        ['Failed', data.tasks.failed],
      ];
      document.getElementById('summary').innerHTML = pills.map(([label, value]) => `<span class="status-pill">${esc(label)} <strong>${value || 0}</strong></span>`).join('');
    }

    async function refresh() {
      if (currentView !== 'overview') return;
      const data = await fetch('/api/dashboard').then(r => r.json());
      latestData = data;
      document.getElementById('meta').textContent = `backend=${data.backend} storage=${data.storage_backend || 'local'} db=${data.db_path}`;
      renderSummary(data);
      renderGraph(data);
      renderTables(data);
    }

    async function retryTask(id) {
      await fetch(`/api/tasks/${id}/retry`, { method: 'POST' });
      refresh();
    }

    function activateRoute() {
      const hash = location.hash || '#overview';
      if (hash.startsWith('#chunk/')) {
        currentView = 'chunk';
        document.body.dataset.view = 'chunk';
        document.querySelectorAll('[data-nav]').forEach(item => item.classList.remove('active'));
        loadChunk(decodeURIComponent(hash.slice('#chunk/'.length)));
        return;
      }
      if (hash === '#sounds') {
        currentView = 'sounds';
        document.body.dataset.view = 'sounds';
        document.querySelectorAll('[data-nav]').forEach(item => item.classList.toggle('active', item.dataset.nav === 'sounds'));
        loadSounds();
        return;
      }
      currentView = 'overview';
      document.body.dataset.view = 'overview';
      document.querySelectorAll('[data-nav]').forEach(item => item.classList.toggle('active', item.dataset.nav === 'overview'));
      refresh();
    }

    document.querySelectorAll('.flow-node').forEach(node => {
      const activate = () => {
        selectedNode = node.dataset.node || 'source';
        if (latestData) renderGraph(latestData);
      };
      node.addEventListener('click', activate);
      node.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          activate();
        }
      });
    });

    document.getElementById('job-form').addEventListener('submit', async event => {
      event.preventDefault();
      const audioFile = document.getElementById('audio-file').files[0];
      const prompt = document.getElementById('prompt').value;
      if (!audioFile) return;
      const body = new FormData();
      body.append('audio_file', audioFile);
      if (prompt) body.append('prompt', prompt);
      const response = await fetch('/api/jobs/upload', { method: 'POST', body });
      if (!response.ok) alert(await response.text());
      document.getElementById('audio-file').value = '';
      loadedSounds = false;
      refresh();
    });
    document.getElementById('sounds-refresh').addEventListener('click', () => loadSounds(true));
    window.addEventListener('hashchange', activateRoute);
    activateRoute();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


def create_app(config: PipelineConfig | None = None, storage: ArtifactStorage | None = None) -> FastAPI:
    runtime = PipelineRuntime(config or PipelineConfig.from_env(), storage=storage)
    runtime.config.output_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.init_db()
        runtime.start_worker()
        try:
            yield
        finally:
            runtime.stop_worker()

    app = FastAPI(title="QLabeler Pipeline", version="1.0.0", lifespan=lifespan)
    app.state.pipeline_runtime = runtime
    app.mount("/files", StaticFiles(directory=str(runtime.config.output_dir)), name="files")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "backend": runtime.config.backend, "storage_backend": runtime.storage.backend}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        return {
            "ready": True,
            "backend": runtime.config.backend,
            "storage_backend": runtime.storage.backend,
            "db_path": str(runtime.config.db_path),
            "worker_enabled": runtime.config.worker_enabled,
        }

    @app.get("/api/dashboard")
    def api_dashboard() -> dict[str, Any]:
        return runtime.dashboard_summary()

    @app.get("/api/sounds")
    def api_sounds(
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        return runtime.sounds(limit=limit, offset=offset)

    @app.post("/api/jobs")
    def create_job(request: JobCreateRequest) -> dict[str, Any]:
        try:
            return runtime.create_job(request.audio_path, request.prompt)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=exception_detail(exc)) from exc

    @app.post("/api/jobs/upload")
    def upload_job(
        audio_file: UploadFile = File(...),
        prompt: str | None = Form(default=None),
    ) -> dict[str, Any]:
        try:
            uploaded_path = runtime.save_upload(audio_file)
            return runtime.create_job(str(uploaded_path), prompt)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=exception_detail(exc)) from exc

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str) -> dict[str, Any]:
        try:
            return runtime.job_detail(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc

    @app.get("/api/chunks/{chunk_id}/waveforms")
    def chunk_waveforms(chunk_id: str) -> dict[str, Any]:
        try:
            return runtime.chunk_waveforms(chunk_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Chunk not found: {chunk_id}") from exc

    @app.post("/api/tasks/{task_id}/retry", response_model=RetryResponse)
    def retry_task(task_id: str) -> RetryResponse:
        try:
            return RetryResponse(**runtime.retry_task(task_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/mock/sound-gate")
    def mock_sound_gate(request: MockAudioRequest) -> dict[str, Any]:
        try:
            audio_path = resolve_local_path(request.audio_ref(), base_dir=runtime.config.workspace_dir)
            return runtime.mock_sound_gate(audio_path)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/mock/audio-flamingo/ask")
    def mock_audio_flamingo(request: MockAudioRequest) -> dict[str, Any]:
        try:
            audio_path = resolve_local_path(request.audio_ref(), base_dir=runtime.config.workspace_dir)
            return runtime.mock_audio_flamingo(audio_path, request.prompt_text())
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/mock/sam-audio/separate")
    def mock_sam_audio(request: MockAudioRequest) -> dict[str, Any]:
        try:
            audio_path = resolve_local_path(request.audio_ref(), base_dir=runtime.config.workspace_dir)
            return runtime.mock_sam_audio(audio_path, request.prompt_text(), clean_prefix(request.output_prefix or "mock_sam_audio"))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
