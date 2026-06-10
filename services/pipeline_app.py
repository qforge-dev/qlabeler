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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydub import AudioSegment

from services.common import OUTPUT_DIR, WORKSPACE_DIR, exception_detail, parse_bool, resolve_local_path


TASK_SOUND_GATE = "sound_gate"
TASK_AUDIO_FLAMINGO = "audio_flamingo"
TASK_SAM_AUDIO = "sam_audio"
TASK_QUEUES = (TASK_SOUND_GATE, TASK_AUDIO_FLAMINGO, TASK_SAM_AUDIO)

STAGE_SOUND_GATE = "sound_gate"
STAGE_DESCRIBE_SFX = "describe_sfx"
STAGE_SEPARATE_SFX = "separate_sfx"
STAGE_COMPLETE = "complete"
STAGE_SKIPPED_SILENT = "skipped_silent"
STAGE_FAILED = "failed"
STAGES = (
    STAGE_SOUND_GATE,
    STAGE_DESCRIBE_SFX,
    STAGE_SEPARATE_SFX,
    STAGE_COMPLETE,
    STAGE_SKIPPED_SILENT,
    STAGE_FAILED,
)

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    if not path:
        return None
    resolved = Path(path).resolve()
    ref = {"path": str(resolved)}
    try:
        relative = resolved.relative_to(output_dir.resolve())
    except ValueError:
        return ref
    ref["url"] = f"/files/{relative.as_posix()}"
    return ref


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
    sam_audio_endpoint: str = "http://127.0.0.1:8002/v1/sam-audio/separate"
    worker_enabled: bool = True
    worker_interval_seconds: float = 1.0
    chunk_ms: int = 30_000
    overlap_ms: int = 5_000
    sound_gate_min_dbfs: float = -50.0
    sound_gate_min_peak_dbfs: float = -55.0
    sound_gate_window_ms: int = 100
    sound_gate_min_active_ms: int = 250
    sound_gate_min_active_ratio: float = 0.01
    request_timeout_seconds: float = 600.0

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        workspace_dir = Path(os.environ.get("WORKSPACE_DIR", str(WORKSPACE_DIR))).expanduser().resolve()
        output_dir = Path(os.environ.get("OUTPUT_DIR", str(OUTPUT_DIR))).expanduser().resolve()
        db_path = Path(os.environ.get("PIPELINE_DB_PATH", str(workspace_dir / "pipeline.sqlite3"))).expanduser().resolve()
        return cls(
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            db_path=db_path,
            backend=os.environ.get("PIPELINE_BACKEND", "mock").strip().lower() or "mock",
            afnext_endpoint=os.environ.get("AFNEXT_ENDPOINT", "http://127.0.0.1:8001/v1/audio-flamingo/ask"),
            sam_audio_endpoint=os.environ.get("SAM_AUDIO_ENDPOINT", "http://127.0.0.1:8002/v1/sam-audio/separate"),
            worker_enabled=parse_bool(os.environ.get("PIPELINE_WORKER_ENABLED"), default=True),
            worker_interval_seconds=float(os.environ.get("PIPELINE_WORKER_INTERVAL_SECONDS", "1.0")),
            chunk_ms=int(os.environ.get("PIPELINE_CHUNK_SECONDS", "30")) * 1000,
            overlap_ms=int(os.environ.get("PIPELINE_OVERLAP_SECONDS", "5")) * 1000,
            sound_gate_min_dbfs=float(os.environ.get("PIPELINE_SOUND_GATE_MIN_DBFS", "-50")),
            sound_gate_min_peak_dbfs=float(os.environ.get("PIPELINE_SOUND_GATE_MIN_PEAK_DBFS", "-55")),
            sound_gate_window_ms=int(os.environ.get("PIPELINE_SOUND_GATE_WINDOW_MS", "100")),
            sound_gate_min_active_ms=int(os.environ.get("PIPELINE_SOUND_GATE_MIN_ACTIVE_MS", "250")),
            sound_gate_min_active_ratio=float(os.environ.get("PIPELINE_SOUND_GATE_MIN_ACTIVE_RATIO", "0.01")),
            request_timeout_seconds=float(os.environ.get("PIPELINE_REQUEST_TIMEOUT_SECONDS", "600")),
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
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._worker_thread: threading.Thread | None = None
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
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
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
                CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id, created_at);
                """
            )

    def start_worker(self) -> None:
        if not self.config.worker_enabled:
            return
        with self._worker_lock:
            if self._worker_thread and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self.worker_loop, name="pipeline-worker", daemon=True)
            self._worker_thread.start()

    def stop_worker(self) -> None:
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)

    def worker_loop(self) -> None:
        while not self._stop_event.is_set():
            did_work = self.process_pending_once()
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
            chunk_path = chunk_dir / f"chunk_{chunk_index:04d}_{start_ms:08d}-{end_ms:08d}.wav"
            segment[start_ms:end_ms].export(str(chunk_path), format="wav")
            chunks.append(
                {
                    "id": new_id(),
                    "chunk_index": chunk_index,
                    "audio_path": str(chunk_path),
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": end_ms - start_ms,
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
                    payload={"audio_path": chunk["audio_path"]},
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
        chunk_id: str,
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

    def claim_next_task(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM tasks
                WHERE status = ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (STATUS_PENDING,),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None

            timestamp = now_iso()
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
            conn.execute("COMMIT")
            return dict(row) | {"status": STATUS_RUNNING, "attempts": int(row["attempts"]) + 1}

    def process_pending_once(self) -> bool:
        task = self.claim_next_task()
        if task is None:
            return False

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
        return True

    def process_sound_gate(self, task: dict[str, Any]) -> None:
        payload = json_loads(task["payload_json"], {})
        audio_path = Path(payload["audio_path"])
        result = self.sound_gate(audio_path)
        timestamp = now_iso()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if result["has_sound"]:
                conn.execute(
                    "UPDATE chunks SET stage = ?, error = NULL, updated_at = ? WHERE id = ?",
                    (STAGE_DESCRIBE_SFX, timestamp, task["chunk_id"]),
                )
                self._insert_task(
                    conn,
                    job_id=task["job_id"],
                    chunk_id=task["chunk_id"],
                    queue=TASK_AUDIO_FLAMINGO,
                    payload={
                        "audio_path": str(audio_path),
                        "prompt": self.default_audio_flamingo_prompt(task["job_id"]),
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

    def process_audio_flamingo(self, task: dict[str, Any]) -> None:
        payload = json_loads(task["payload_json"], {})
        audio_path = Path(payload["audio_path"])
        prompt = payload.get("prompt") or self.default_audio_flamingo_prompt(task["job_id"])
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

        text = str(response.get("text", "")).strip()
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

    def process_sam_audio(self, task: dict[str, Any]) -> None:
        payload = json_loads(task["payload_json"], {})
        audio_path = Path(payload["audio_path"])
        prompt = payload["prompt"]
        chunk = self.chunk(task["chunk_id"])
        output_prefix = clean_prefix(f"job_{task['job_id']}_chunk_{chunk['chunk_index']:04d}_{prompt}")

        if self.config.backend == "mock":
            response = self.mock_sam_audio(audio_path, prompt, output_prefix, task["job_id"])
        else:
            response = post_json(
                self.config.sam_audio_endpoint,
                {
                    "audio_path": str(audio_path),
                    "input": prompt,
                    "output_prefix": output_prefix,
                    "max_audio_seconds": max(35, int((chunk["duration_ms"] + 999) / 1000)),
                    "predict_spans": False,
                    "reranking_candidates": 1,
                },
                timeout=self.config.request_timeout_seconds,
            )

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

            stage = {
                TASK_SOUND_GATE: STAGE_SOUND_GATE,
                TASK_AUDIO_FLAMINGO: STAGE_DESCRIBE_SFX,
                TASK_SAM_AUDIO: STAGE_SEPARATE_SFX,
            }.get(task["queue"], STAGE_FAILED)
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

    def default_audio_flamingo_prompt(self, job_id: str) -> str:
        with self.connect() as conn:
            row = conn.execute("SELECT prompt FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row and row["prompt"]:
            return row["prompt"]
        return "Identify audible sources and return exactly two lines: SOUNDS: <sources>; SAM_PROMPT: <one target sound only>."

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
            or active_ratio >= self.config.sound_gate_min_active_ratio
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
        return {
            "model_id": "mock/audio-flamingo-next",
            "audio_path": str(audio_path),
            "prompt": prompt,
            "text": "SOUNDS: horse hooves, cinematic strings\nSAM_PROMPT: horse hooves",
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

    def extract_sam_prompt(self, text: str) -> str:
        match = re.search(r"SAM_PROMPT\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            raise ValueError(f"Audio Flamingo response missing SAM_PROMPT: {text}")
        prompt = match.group(1).strip().splitlines()[0].strip().strip("\"'` .;")
        if not prompt:
            raise ValueError(f"Audio Flamingo response had an empty SAM_PROMPT: {text}")
        return prompt[:180]

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
            events = [
                self._event_row(row)
                for row in conn.execute("SELECT * FROM events WHERE job_id = ? ORDER BY created_at DESC, id DESC LIMIT 100", (job_id,))
            ]
        return {"job": dict(job), "chunks": chunks, "tasks": tasks, "stems": stems, "events": events}

    def dashboard_summary(self) -> dict[str, Any]:
        with self.connect() as conn:
            totals = {
                "jobs": conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0],
                "chunks": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
                "stems": conn.execute("SELECT COUNT(*) FROM stems").fetchone()[0],
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

            recent_jobs = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT
                        j.*,
                        SUM(CASE WHEN c.stage = 'complete' THEN 1 ELSE 0 END) AS complete_chunks,
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
                    JOIN chunks c ON c.id = t.chunk_id
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
        return {
            "backend": self.config.backend,
            "db_path": str(self.config.db_path),
            "output_dir": str(self.config.output_dir),
            "totals": totals,
            "jobs": job_counts,
            "tasks": task_counts,
            "stages": stage_counts,
            "queues": queue_counts,
            "recent_jobs": recent_jobs,
            "recent_failures": failures,
            "recent_outputs": outputs,
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
    .submit-section { padding: 14px 16px; }
    .submit-section form { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    .status-strip { display: flex; gap: 8px; flex-wrap: wrap; }
    .status-pill { display: inline-flex; gap: 7px; align-items: center; padding: 5px 9px; border: 1px solid var(--line); border-radius: 999px; background: #fff; color: var(--muted); font-size: 12px; }
    .status-pill strong { color: var(--ink); font-size: 13px; }
    .flow-board { padding: 0; }
    .flow-wrap { overflow-x: auto; background: linear-gradient(#fff, #fbfcfd); }
    .flow-graph { display: block; width: 100%; min-width: 0; height: auto; max-height: 520px; }
    .flow-title { font-size: 15px; font-weight: 700; fill: var(--ink); }
    .flow-subtitle { font-size: 12px; fill: var(--muted); }
    .flow-small { font-size: 11px; fill: var(--muted); }
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
    .status-failed { color: #c53232; font-weight: 650; }
    .status-complete, .status-completed { color: #18733f; font-weight: 650; }
    .status-running { color: #9a5a00; font-weight: 650; }
    @media (max-width: 720px) {
      body { min-width: 360px; }
      main, header { padding-left: 14px; padding-right: 14px; }
      .section-head { align-items: flex-start; flex-direction: column; }
      .node-inspector { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>QLabeler Pipeline</h1>
    <div class="meta" id="meta">Loading...</div>
  </header>
  <main>
    <section class="submit-section">
      <form id="job-form">
        <input id="audio-file" type="file" accept="audio/*,.mp3,.wav,.flac,.m4a" required>
        <input id="prompt" placeholder="Optional Audio Flamingo prompt">
        <button type="submit">Queue</button>
      </form>
    </section>

    <section class="flow-board">
      <div class="section-head">
        <h2>Pipeline Graph</h2>
        <div class="status-strip" id="summary"></div>
      </div>
      <div class="flow-wrap">
        <svg id="flow-graph" class="flow-graph" viewBox="0 0 1052 404" role="img" aria-label="Pipeline status graph">
          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L9,3 z" fill="#ef4444"></path>
            </marker>
          </defs>

          <g transform="scale(0.72)">
          <path class="arrow" d="M105 276 C132 276 142 276 165 276"></path>
          <path class="arrow" d="M315 276 C336 276 350 276 371 276"></path>
          <path class="arrow" d="M496 276 C525 276 540 246 578 246"></path>
          <path class="arrow soft" d="M444 356 C520 432 822 444 1042 444"></path>
          <path class="arrow" d="M744 246 C783 246 804 276 838 276"></path>
          <path class="arrow green" d="M1004 276 C1042 256 1062 220 1096 188"></path>
          <path class="arrow soft" d="M1004 305 C1098 350 1166 404 1252 426"></path>
          <path class="arrow soft" d="M693 310 C814 414 1042 482 1252 466"></path>
          <path class="arrow soft" d="M432 196 C518 72 890 80 1098 134"></path>

          <g class="flow-node kind-input selected" tabindex="0" data-node="source">
            <rect class="node-shape" x="34" y="220" width="72" height="112" rx="14"></rect>
            <text class="flow-title" x="70" y="265" text-anchor="middle">Audio</text>
            <text class="flow-title" x="70" y="286" text-anchor="middle">track</text>
            <g class="count-badge" data-badge="source" transform="translate(108 216)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-chunk" tabindex="0" data-node="chunks">
            <rect class="node-shape" x="165" y="216" width="150" height="120" rx="13"></rect>
            <text class="flow-title" x="240" y="268" text-anchor="middle">30s chunks</text>
            <text class="flow-subtitle" x="240" y="292" text-anchor="middle">5s overlap</text>
            <g class="count-badge" data-badge="chunks" transform="translate(316 212)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-gate" tabindex="0" data-node="sound_gate">
            <polygon class="node-shape" points="432,184 496,276 432,368 368,276"></polygon>
            <text class="flow-title" x="432" y="258" text-anchor="middle">sound</text>
            <text class="flow-title" x="432" y="278" text-anchor="middle">gate</text>
            <text class="flow-subtitle" x="432" y="299" text-anchor="middle">filter</text>
            <text class="flow-small" data-node-meta="sound_gate" x="432" y="392" text-anchor="middle">waiting 0</text>
            <g class="count-badge" data-badge="sound_gate" transform="translate(493 204)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-model" tabindex="0" data-node="describe_sfx">
            <rect class="node-shape" x="578" y="194" width="166" height="104" rx="12"></rect>
            <text class="flow-title" x="661" y="234" text-anchor="middle">describe SFX</text>
            <text class="flow-subtitle" x="661" y="262" text-anchor="middle">audio_flamingo</text>
            <text class="flow-small" data-node-meta="describe_sfx" x="661" y="316" text-anchor="middle">waiting 0</text>
            <g class="count-badge" data-badge="describe_sfx" transform="translate(745 190)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-work" tabindex="0" data-node="separate_sfx">
            <rect class="node-shape" x="838" y="216" width="166" height="120" rx="12"></rect>
            <text class="flow-title" x="921" y="268" text-anchor="middle">separate SFX</text>
            <text class="flow-subtitle" x="921" y="294" text-anchor="middle">sam_audio</text>
            <text class="flow-small" data-node-meta="separate_sfx" x="921" y="358" text-anchor="middle">waiting 0</text>
            <g class="count-badge" data-badge="separate_sfx" transform="translate(1005 212)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-db" tabindex="0" data-node="stems_db">
            <circle class="node-shape" cx="1166" cy="158" r="78"></circle>
            <text class="flow-title" x="1166" y="128" text-anchor="middle">STEMS</text>
            <text class="flow-title" x="1166" y="150" text-anchor="middle">DB</text>
            <text class="flow-subtitle" x="1166" y="178" text-anchor="middle">target + residual</text>
            <text class="flow-subtitle" x="1166" y="199" text-anchor="middle">refs</text>
            <g class="count-badge" data-badge="stems_db" transform="translate(1238 100)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-terminal" tabindex="0" data-node="skipped_silent">
            <rect class="node-shape" x="1042" y="402" width="152" height="84" rx="12"></rect>
            <text class="flow-title" x="1118" y="438" text-anchor="middle">skipped</text>
            <text class="flow-subtitle" x="1118" y="462" text-anchor="middle">silent chunk</text>
            <g class="count-badge" data-badge="skipped_silent" transform="translate(1195 398)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>

          <g class="flow-node kind-failed" tabindex="0" data-node="failed">
            <rect class="node-shape" x="1252" y="410" width="132" height="82" rx="12"></rect>
            <text class="flow-title" x="1318" y="447" text-anchor="middle">failed</text>
            <text class="flow-subtitle" x="1318" y="470" text-anchor="middle">retryable</text>
            <g class="count-badge" data-badge="failed" transform="translate(1385 406)"><rect class="badge-bg" x="-18" y="-15" width="36" height="30" rx="15"></rect><text class="badge-text">0</text></g>
          </g>
          </g>
        </svg>
      </div>
      <div id="node-inspector" class="node-inspector"></div>
    </section>

    <div class="two">
      <section class="table-section">
        <h2>Stages</h2>
        <table id="stages"></table>
      </section>
      <section class="table-section">
        <h2>Queues</h2>
        <table id="queues"></table>
      </section>
    </div>

    <div class="tables">
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
  </main>
  <script>
    const FLOW_NODES = {
      source: { title: 'Audio Track', kind: 'Input', badge: 'jobs' },
      chunks: { title: 'Chunk Splitter', kind: 'Preprocess', badge: 'chunks' },
      sound_gate: { title: 'Sound Gate Filter', kind: 'Queue: sound_gate', badge: 'waiting' },
      describe_sfx: { title: 'Describe SFX', kind: 'Queue: audio_flamingo', badge: 'waiting' },
      separate_sfx: { title: 'Separate SFX', kind: 'Queue: sam_audio', badge: 'waiting' },
      stems_db: { title: 'Stems DB', kind: 'Output refs', badge: 'stems' },
      skipped_silent: { title: 'Skipped Silent', kind: 'Terminal stage', badge: 'chunks' },
      failed: { title: 'Failed', kind: 'Retryable work', badge: 'failures' },
    };
    let selectedNode = 'source';
    let latestData = null;

    const statusClass = value => `status-${String(value || '').replaceAll('_', '-')}`;
    const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    const link = ref => ref && ref.url ? `<a href="${esc(ref.url)}"><code>${esc(ref.path)}</code></a>` : (ref && ref.path ? `<code>${esc(ref.path)}</code>` : '');
    const queue = (data, name) => data.queues[name] || { pending: 0, running: 0, failed: 0, completed: 0 };
    const number = value => Number(value || 0);

    function rows(headers, items, cells) {
      return `<thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>${items.map(item => `<tr>${cells(item).join('')}</tr>`).join('') || `<tr><td colspan="${headers.length}">No rows</td></tr>`}</tbody>`;
    }

    function flowStats(data) {
      const jobs = data.jobs || { queued: 0, running: 0, complete: 0, failed: 0 };
      const soundGate = queue(data, 'sound_gate');
      const audioFlamingo = queue(data, 'audio_flamingo');
      const samAudio = queue(data, 'sam_audio');
      const activeJobs = number(jobs.queued) + number(jobs.running);
      const activeChunks = number(data.stages.sound_gate) + number(data.stages.describe_sfx) + number(data.stages.separate_sfx);
      const soundGateActive = number(soundGate.pending) + number(soundGate.running);
      const audioFlamingoActive = number(audioFlamingo.pending) + number(audioFlamingo.running);
      const samAudioActive = number(samAudio.pending) + number(samAudio.running);
      return {
        source: {
          badge: activeJobs,
          running: number(jobs.running),
          failed: number(jobs.failed),
          metrics: [['Ongoing jobs', activeJobs], ['Queued jobs', jobs.queued], ['Running jobs', jobs.running], ['Failed jobs', jobs.failed]],
        },
        chunks: {
          badge: activeChunks,
          failed: number(data.stages.failed),
          metrics: [['Ongoing chunks', activeChunks], ['Waiting gate', data.stages.sound_gate], ['Describing', data.stages.describe_sfx], ['Separating', data.stages.separate_sfx]],
        },
        sound_gate: {
          badge: soundGateActive,
          waiting: number(soundGate.pending),
          running: number(soundGate.running),
          failed: number(soundGate.failed),
          done: number(soundGate.completed),
          metrics: [['Ongoing', soundGateActive], ['Waiting', soundGate.pending], ['Running', soundGate.running], ['Failed', soundGate.failed]],
        },
        describe_sfx: {
          badge: audioFlamingoActive,
          waiting: number(audioFlamingo.pending),
          running: number(audioFlamingo.running),
          failed: number(audioFlamingo.failed),
          done: number(audioFlamingo.completed),
          metrics: [['Ongoing', audioFlamingoActive], ['Waiting', audioFlamingo.pending], ['Running', audioFlamingo.running], ['Failed', audioFlamingo.failed]],
        },
        separate_sfx: {
          badge: samAudioActive,
          waiting: number(samAudio.pending),
          running: number(samAudio.running),
          failed: number(samAudio.failed),
          done: number(samAudio.completed),
          metrics: [['Ongoing', samAudioActive], ['Waiting', samAudio.pending], ['Running', samAudio.running], ['Failed', samAudio.failed]],
        },
        stems_db: {
          badge: 0,
          metrics: [['Ongoing writes', 0], ['Stem rows total', data.totals.stems], ['Recent outputs', data.recent_outputs.length]],
        },
        skipped_silent: {
          badge: 0,
          metrics: [['Ongoing skips', 0], ['Silent chunks total', data.stages.skipped_silent], ['All chunks', data.totals.chunks]],
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
          meta.textContent = `waiting ${values.waiting || 0} · running ${values.running || 0}`;
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
      const metrics = values.metrics || [];
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
      document.getElementById('queues').innerHTML = rows(['Queue', 'Pending', 'Running', 'Failed', 'Completed'], Object.entries(data.queues), ([q, c]) => [`<td><code>${esc(q)}</code></td>`, `<td>${c.pending || 0}</td>`, `<td>${c.running || 0}</td>`, `<td>${c.failed || 0}</td>`, `<td>${c.completed || 0}</td>`]);
      document.getElementById('jobs').innerHTML = rows(['Job', 'Status', 'Chunks', 'Source', 'Updated'], data.recent_jobs, j => [`<td><code>${esc(j.id)}</code></td>`, `<td class="${statusClass(j.status)}">${esc(j.status)}</td>`, `<td>${j.complete_chunks || 0}/${j.chunk_count || 0}</td>`, `<td><code>${esc(j.source_audio_path)}</code></td>`, `<td>${esc(j.updated_at)}</td>`]);
      document.getElementById('failures').innerHTML = rows(['Task', 'Queue', 'Chunk', 'Error', 'Retry'], data.recent_failures, f => [`<td><code>${esc(f.id)}</code></td>`, `<td><code>${esc(f.queue)}</code></td>`, `<td>${esc(f.chunk_index)}</td>`, `<td>${esc(f.error)}</td>`, `<td><button class="secondary" onclick="retryTask('${esc(f.id)}')">Retry</button></td>`]);
      document.getElementById('outputs').innerHTML = rows(['Chunk', 'Prompt', 'Target', 'Residual', 'Zip'], data.recent_outputs, o => [`<td>${esc(o.chunk_index)}</td>`, `<td>${esc(o.prompt)}</td>`, `<td>${link(o.target.wav || o.target.mp3)}</td>`, `<td>${link(o.residual.wav || o.residual.mp3)}</td>`, `<td>${link(o.zip)}</td>`]);
    }

    function renderSummary(data) {
      const jobs = data.jobs || { queued: 0, running: 0, complete: 0, failed: 0 };
      const activeJobs = number(jobs.queued) + number(jobs.running);
      const activeChunks = number(data.stages.sound_gate) + number(data.stages.describe_sfx) + number(data.stages.separate_sfx);
      const pills = [
        ['Total jobs', data.totals.jobs],
        ['Ongoing jobs', activeJobs],
        ['Ongoing chunks', activeChunks],
        ['Pending', data.tasks.pending],
        ['Running', data.tasks.running],
        ['Failed', data.tasks.failed],
      ];
      document.getElementById('summary').innerHTML = pills.map(([label, value]) => `<span class="status-pill">${esc(label)} <strong>${value || 0}</strong></span>`).join('');
    }

    async function refresh() {
      const data = await fetch('/api/dashboard').then(r => r.json());
      latestData = data;
      document.getElementById('meta').textContent = `backend=${data.backend} db=${data.db_path}`;
      renderSummary(data);
      renderGraph(data);
      renderTables(data);
    }

    async function retryTask(id) {
      await fetch(`/api/tasks/${id}/retry`, { method: 'POST' });
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
      refresh();
    });
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


def create_app(config: PipelineConfig | None = None) -> FastAPI:
    runtime = PipelineRuntime(config or PipelineConfig.from_env())
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
        return {"status": "ok", "backend": runtime.config.backend}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        return {
            "ready": True,
            "backend": runtime.config.backend,
            "db_path": str(runtime.config.db_path),
            "worker_enabled": runtime.config.worker_enabled,
        }

    @app.get("/api/dashboard")
    def api_dashboard() -> dict[str, Any]:
        return runtime.dashboard_summary()

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
