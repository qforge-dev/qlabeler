from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


def clean_storage_part(value: str | None, default: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "_", (value or "").strip()).strip("._-")
    return cleaned[:128] or default


def local_file_ref(path: str | Path | None, output_dir: Path) -> dict[str, str] | None:
    if not path:
        return None
    resolved = Path(path).expanduser().resolve()
    ref = {"backend": "local", "path": str(resolved)}
    try:
        relative = resolved.relative_to(output_dir.expanduser().resolve())
    except ValueError:
        return ref
    ref["url"] = f"/files/{relative.as_posix()}"
    return ref


class ArtifactStorage(Protocol):
    backend: str

    def store_artifact_file(
        self,
        local_path: Path,
        *,
        artifact_id: str,
        job_id: str,
        chunk_id: str | None,
        kind: str,
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class LocalStorage:
    output_dir: Path
    backend: str = "local"

    def store_artifact_file(
        self,
        local_path: Path,
        *,
        artifact_id: str,
        job_id: str,
        chunk_id: str | None,
        kind: str,
    ) -> dict[str, Any]:
        return local_file_ref(local_path, self.output_dir) or {"backend": self.backend, "path": str(local_path.resolve())}


@dataclass(frozen=True)
class S3Storage:
    bucket: str
    prefix: str = "qlabeler"
    region: str | None = None
    endpoint_url: str | None = None
    public_base_url: str | None = None
    presign_seconds: int = 0
    client: Any | None = None
    backend: str = "s3"

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("S3 storage requires boto3. Install boto3 in the pipeline environment.") from exc

        kwargs: dict[str, Any] = {}
        if self.region:
            kwargs["region_name"] = self.region
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        return boto3.client("s3", **kwargs)

    def _key_for(self, local_path: Path, *, artifact_id: str, job_id: str, chunk_id: str | None, kind: str) -> str:
        prefix = self.prefix.strip("/")
        parts = [
            clean_storage_part(job_id, "job"),
            clean_storage_part(chunk_id, "job"),
            clean_storage_part(kind, "artifact"),
            f"{clean_storage_part(artifact_id, 'artifact')}_{clean_storage_part(local_path.name, 'file')}",
        ]
        key = "/".join(part for part in parts if part)
        return f"{prefix}/{key}" if prefix else key

    def store_artifact_file(
        self,
        local_path: Path,
        *,
        artifact_id: str,
        job_id: str,
        chunk_id: str | None,
        kind: str,
    ) -> dict[str, Any]:
        resolved = local_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Artifact file does not exist: {resolved}")

        key = self._key_for(resolved, artifact_id=artifact_id, job_id=job_id, chunk_id=chunk_id, kind=kind)
        content_type, _ = mimetypes.guess_type(resolved.name)
        extra_args = {"ContentType": content_type} if content_type else None

        client = self._client()
        if extra_args:
            client.upload_file(str(resolved), self.bucket, key, ExtraArgs=extra_args)
        else:
            client.upload_file(str(resolved), self.bucket, key)

        uri = f"s3://{self.bucket}/{key}"
        ref: dict[str, Any] = {
            "backend": self.backend,
            "bucket": self.bucket,
            "key": key,
            "uri": uri,
            "path": uri,
            "local_path": str(resolved),
        }
        if self.public_base_url:
            ref["url"] = f"{self.public_base_url.rstrip('/')}/{key}"
        elif self.presign_seconds > 0 and hasattr(client, "generate_presigned_url"):
            ref["url"] = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=self.presign_seconds,
            )
        return ref


def create_storage_adapter(
    *,
    backend: str,
    output_dir: Path,
    s3_bucket: str | None = None,
    s3_prefix: str = "qlabeler",
    s3_region: str | None = None,
    s3_endpoint_url: str | None = None,
    s3_public_base_url: str | None = None,
    s3_presign_seconds: int = 0,
) -> ArtifactStorage:
    storage_backend = backend.strip().lower()
    if storage_backend == "local":
        return LocalStorage(output_dir=output_dir)
    if storage_backend == "s3":
        if not s3_bucket:
            raise ValueError("PIPELINE_STORAGE_BACKEND=s3 requires S3_BUCKET.")
        return S3Storage(
            bucket=s3_bucket,
            prefix=s3_prefix,
            region=s3_region,
            endpoint_url=s3_endpoint_url,
            public_base_url=s3_public_base_url,
            presign_seconds=s3_presign_seconds,
        )
    raise ValueError(f"Unsupported pipeline storage backend: {backend}")
