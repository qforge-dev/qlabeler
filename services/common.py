from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlparse


WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", Path.cwd())).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", WORKSPACE_DIR / "outputs")).expanduser().resolve()


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def exception_detail(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def resolve_local_path(value: str, *, base_dir: Path = WORKSPACE_DIR, must_exist: bool = True) -> Path:
    """Resolve a request-supplied local path or file:// URL."""
    raw = (value or "").strip()
    if not raw:
        raise ValueError("A local audio path is required.")

    parsed = urlparse(raw)
    if parsed.scheme and parsed.scheme != "file":
        raise ValueError("Only local filesystem paths and file:// URLs are supported.")

    path_text = unquote(parsed.path) if parsed.scheme == "file" else raw
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()

    if must_exist and not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    return path


def resolve_output_dir(value: str | None, *, default_subdir: str) -> Path:
    raw = (value or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = WORKSPACE_DIR / path
        path = path.resolve()
    else:
        path = (OUTPUT_DIR / default_subdir).resolve()

    path.mkdir(parents=True, exist_ok=True)
    return path


def public_file_ref(path: Path) -> dict[str, str]:
    """Return a local file reference and, when possible, a mounted API URL."""
    resolved = path.resolve()
    ref = {"path": str(resolved)}
    try:
        relative = resolved.relative_to(OUTPUT_DIR)
    except ValueError:
        return ref
    ref["url"] = f"/files/{relative.as_posix()}"
    return ref
