import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import zipfile
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def append(path, rows):
    with Path(path).open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")


def rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def command(args):
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
        return dict(returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(error=str(error))


def identity(model=None):
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p)
        for base in ("benchmark", "nanoinfer") for p in sorted((ROOT/base).rglob("*"))
        if p.is_file() and p.suffix in (".py", ".md", ".json", ".txt")
        and not any(x.startswith(".") or x in ("datasets", "__pycache__") for x in p.relative_to(ROOT).parts)}
    packages = {}
    for name in ("torch", "triton", "transformers", "flash-attn", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = dict(protocol_version=2, at=now(), python=sys.version, executable=sys.executable,
        platform=platform.platform(), packages=packages, source_sha256=sources,
        git_head=command(["git", "rev-parse", "HEAD"]), git_status=command(["git", "status", "--short"]),
        gpu=command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,power.limit", "--format=csv"]))
    if model:
        result["model_path"] = str(Path(model).resolve())
        result["model_sha256"] = {p.name: digest(p) for p in sorted(Path(model).iterdir()) if p.is_file()}
    return result


def snapshot(output, manifest):
    with zipfile.ZipFile(Path(output)/"source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, expected in manifest["source_sha256"].items():
            if digest(ROOT/name) != expected:
                raise RuntimeError(f"Source changed during snapshot: {name}")
            archive.write(ROOT/name, name)
        for name in ("pyproject.toml", "requirements.txt"):
            archive.write(ROOT/name, name)


def checksums(directory):
    directory = Path(directory)
    lines = [f"{digest(p)}  {p.relative_to(directory).as_posix()}" for p in sorted(directory.rglob("*"))
             if p.is_file() and p.name != "checksums.sha256"]
    (directory/"checksums.sha256").write_text("\n".join(lines)+"\n", encoding="utf-8")
