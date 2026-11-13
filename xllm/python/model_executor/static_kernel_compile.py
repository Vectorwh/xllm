#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://github.com/xLLM-AI/xllm/blob/main/LICENSE
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compile collected CANN operator arguments into a private static kernel cache.

This module needs neither torch nor TorchAir. It runs only during startup.
Completed installations are immutable: a lock and an atomic manifest publish
them to other ranks, while failed attempts retain logs without becoming hits.
"""

import fcntl
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from scripts.logger import logger

_CACHE_SCHEMA = 3


def _cache_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = path.resolve(strict=True)
    metadata = path.stat()
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o022:
        raise ValueError(f"Static kernel cache must be owned by the current user and not writable by others: {path}")
    return path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as output:
        temporary = Path(output.name)
        json.dump(value, output, sort_keys=True)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _blacklisted_ops() -> list[str]:
    value = os.environ.get("XLLM_STATIC_KERNEL_BLACKLIST", "")
    return sorted({op.strip() for op in value.split(",") if op.strip()})


def _operator_inputs(dump_dirs: list[Path]) -> dict[str, Path]:
    # CANN emits compiler inputs separately from debug/PID metadata.
    directories: list[Path] = []
    for dump_dir in sorted(set(path.resolve(strict=True) for path in dump_dirs)):
        selected = sorted(path for path in dump_dir.rglob("*_opcompile") if path.is_dir())
        if not selected:
            selected = sorted(path for path in dump_dir.rglob("*_debug") if path.is_dir())
        directories.extend(selected)
    blacklist = set(_blacklisted_ops())
    excluded: set[str] = set()
    operators: dict[str, Path] = {}
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            description = json.loads(path.read_text(encoding="utf-8"))
            if description.get("op_type") in blacklist:
                excluded.add(description["op_type"])
                continue
            signature = hashlib.sha256(_canonical_json(description)).hexdigest()
            operators.setdefault(signature, path)
    if excluded:
        logger.info("Excluded static kernel OpTypes: %s", sorted(excluded))
    return operators


def _toolchain_fingerprint(compiler: Path) -> dict[str, Any]:
    cann_root = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not cann_root:
        raise ValueError("Source CANN set_env.sh before enabling static kernels")
    root = Path(cann_root).resolve()
    fingerprints: dict[str, str] = {str(compiler): _digest(compiler)}
    version_files = set(root.glob("*.info")) | set(root.glob("*/version.info"))
    version_files |= set(root.glob("*/ascend_*_install.info"))
    for path in sorted(version_files):
        if path.is_file():
            fingerprints[str(path)] = _digest(path)
    for name in ("libopapi.so", "libnnopbase.so", "libascend_dump.so"):
        path = root / "lib64" / name
        if path.is_file():
            fingerprints[str(path.resolve())] = _digest(path)
    opp_root = Path(os.environ.get("ASCEND_OPP_PATH", str(root / "opp"))).resolve()
    vendor_roots = {Path(path).resolve() for path in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if path}
    # CANN also loads vendors registered in the OPP installation when the
    # custom-path environment variable is empty.
    installed_vendors = opp_root / "vendors"
    if installed_vendors.is_dir():
        vendor_roots.update(path.resolve() for path in installed_vendors.iterdir() if path.is_dir())
    vendor_payloads: dict[str, tuple[int, str]] = {}
    for vendor in vendor_roots:
        files = set(vendor.rglob("*.info")) | set(vendor.rglob("*.so"))
        for path in sorted(files):
            if path.is_file():
                fingerprints[str(path)] = _digest(path)
        for path in vendor.rglob("*"):
            if path.is_file() and path.suffix in (".o", ".bin", ".json", ".py", ".cpp", ".h"):
                vendor_payloads[str(path)] = (path.stat().st_size, _digest(path))
    return {
        "cann_root": str(root),
        "opp_root": str(opp_root),
        "vendor_roots": sorted(str(path) for path in vendor_roots),
        "vendor_payloads": vendor_payloads,
        "files": fingerprints,
    }


def _installed_files(install_path: Path) -> dict[str, str]:
    root = install_path / "static_kernel/ai_core"
    configurations = sorted(root.glob("config/*/binary_info_config.json"))
    if not configurations:
        raise ValueError(f"Static installation has no SoC kernel index: {install_path}")
    for configuration in configurations:
        if not json.loads(configuration.read_text(encoding="utf-8")):
            raise ValueError(f"Empty static kernel index: {configuration}")
    binaries = [path for path in root.rglob("*") if path.suffix in (".o", ".bin")]
    if not binaries:
        raise ValueError(f"Static installation has no kernel binaries: {install_path}")
    return {
        str(path.relative_to(install_path)): _digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix != ".lock"
    }


def _valid_cached_result(manifest: Path, cache_key: str) -> dict[str, Any] | None:
    if not manifest.is_file():
        return None
    # Cache corruption is an expected disk/cache condition, checked before any
    # model runtime state changes. Rebuild into a fresh immutable directory.
    try:
        cached = json.loads(manifest.read_text(encoding="utf-8"))
        if cached["cache_key"] != cache_key or cached["schema"] != _CACHE_SCHEMA:
            return None
        if cached.get("fragment") and cached.get("empty"):
            if cached["install_path"] is not None or cached["files"]:
                return None
            return {**cached, "cache_hit": True, "new_operator_count": 0, "compiled_operator_count": 0}
        files = _installed_files(Path(cached["install_path"]))
        if files != cached["files"]:
            return None
        return {**cached, "cache_hit": True, "new_operator_count": 0, "compiled_operator_count": 0}
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("Ignoring incomplete or corrupt static kernel cache: %s", manifest)
        return None


def _run(arguments: list[str], log: Path) -> None:
    logger.info("Running static kernel tool; log: %s", log)
    with (
        log.open("wb") as output,
        subprocess.Popen(arguments, stdout=output, stderr=subprocess.STDOUT, start_new_session=True) as process,
    ):
        try:
            status = process.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        if status != 0:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise subprocess.CalledProcessError(status, arguments)


def _compilation_environment(soc: str, deterministic: bool, compiler: Path) -> dict[str, Any]:
    return {
        "soc": soc,
        "deterministic": deterministic,
        "blacklisted_ops": _blacklisted_ops(),
        "toolchain": _toolchain_fingerprint(compiler),
        "helper": _digest(Path(__file__)),
        "options": {
            name: os.environ.get(name)
            for name in (
                "TASK_QUEUE_ENABLE",
                "HCCL_DETERMINISTIC",
                "CLOSE_MATMUL_K_SHIFT",
                "ASCEND_COMPUTE_UNIT",
                "ACL_OP_SELECT_IMPL_MODE",
                "ACL_OPTYPELIST_FOR_IMPLMODE",
                "ALLOW_MATMUL_HF32",
                "ALLOW_CONV_HF32",
                "NPU_FUZZY_COMPILE_BLACKLIST",
            )
        },
    }


def _compiler_path(compiler: Path | None = None) -> Path:
    if compiler is not None:
        return compiler
    executable = shutil.which("op_compiler")
    if executable is None:
        raise ValueError("op_compiler is not on PATH; source CANN set_env.sh")
    return Path(executable).resolve()


def _extend_installation(previous: Path, installation: Path) -> None:
    # Installers may replace an OpType index. Restore missing files and merge
    # its static candidates, without ever modifying the previous installation.
    for source in previous.rglob("*"):
        if not source.is_file():
            continue
        destination = installation / source.relative_to(previous)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        elif source.suffix in (".o", ".bin") and _digest(source) != _digest(destination):
            raise ValueError(f"Incremental static kernel binary collision: {destination}")
    for source in previous.glob("static_kernel/ai_core/config/*/binary_info_config.json"):
        target = installation / source.relative_to(previous)
        old = json.loads(source.read_text(encoding="utf-8"))
        current = json.loads(target.read_text(encoding="utf-8"))
        for op_type, group in old.items():
            if op_type not in current:
                current[op_type] = group
                continue
            if isinstance(group, dict) and isinstance(current[op_type], dict):
                merged = {**group, **current[op_type]}
                candidates = {_canonical_json(item): item for item in group.get("staticList", [])}
                candidates.update({_canonical_json(item): item for item in current[op_type].get("staticList", [])})
                merged["staticList"] = list(candidates.values())
                current[op_type] = merged
            elif isinstance(group, list) and isinstance(current[op_type], list):
                candidates = {_canonical_json(item): item for item in group + current[op_type]}
                current[op_type] = list(candidates.values())
            else:
                raise ValueError(f"Invalid static kernel index for {op_type}: {target}")
        _atomic_json(target, current)


def _compile_fragment(
    operators: dict[str, Path],
    cache_dir: Path,
    environment_key: str,
    compiler: Path,
    soc: str,
    jobs: int,
) -> dict[str, Any]:
    # Cache the exact newly collected descriptions, independently of earlier
    # rounds. A changing description must not invalidate unrelated later shapes.
    cache_key = hashlib.sha256(_canonical_json([_CACHE_SCHEMA, environment_key, sorted(operators)])).hexdigest()
    fragments = cache_dir / "fragments"
    fragments.mkdir(exist_ok=True)
    manifest = fragments / f"{cache_key}.json"
    with (fragments / f"{cache_key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = _valid_cached_result(manifest, cache_key)
        if cached is not None:
            logger.info("Static kernel fragment cache hit: %s", cache_key)
            return cached
        log_dir = cache_dir / "logs" / cache_key
        log_dir.mkdir(parents=True, exist_ok=True)
        installations = cache_dir / "installations"
        installations.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"build-{cache_key[:12]}-", dir=cache_dir) as temporary:
            attempt = Path(temporary)
            inputs = attempt / "inputs"
            inputs.mkdir()
            for signature, source in operators.items():
                destination = inputs / source.name
                if destination.exists():
                    destination = inputs / f"{source.stem}_{signature}{source.suffix}"
                shutil.copyfile(source, destination)
            output = attempt / "packages"
            output.mkdir()
            _run(
                [str(compiler), "-p", str(inputs), "-v", soc, "-j", str(jobs), "-f", "true", "-o", str(output)],
                log_dir / "compile.log",
            )
            packages = sorted(output.rglob("*.run"))
            install = Path(tempfile.mkdtemp(prefix=f"{cache_key[:12]}-", dir=installations)) if packages else None
            try:
                for index, package in enumerate(packages):
                    _run(
                        ["bash", str(package), f"--install-path={install}", "--quiet"], log_dir / f"install-{index}.log"
                    )
                result = {
                    "schema": _CACHE_SCHEMA,
                    "fragment": True,
                    "cache_key": cache_key,
                    "cache_hit": False,
                    "empty": install is None,
                    "install_path": str(install) if install is not None else None,
                    "files": _installed_files(install) if install is not None else {},
                    "operator_signatures": sorted(operators),
                    "compilation_environment": environment_key,
                    "log_dir": str(log_dir),
                }
                _atomic_json(manifest, result)
            except Exception:
                if install is not None:
                    shutil.rmtree(install)
                raise
        return result


def compile_static_kernels(
    dump_dir: Path,
    cache_dir: Path,
    soc: str,
    deterministic: bool = False,
    jobs: int = 1,
    compiler: Path | None = None,
    dump_dirs: list[Path] | None = None,
    previous_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge participating processes' inputs, deduplicate, compile and install."""
    compiler = _compiler_path(compiler)
    operators = _operator_inputs([dump_dir] if dump_dirs is None else dump_dirs)
    environment = _compilation_environment(soc, deterministic, compiler)
    environment_key = hashlib.sha256(_canonical_json(environment)).hexdigest()
    context = {
        "schema": _CACHE_SCHEMA,
        "environment": environment,
        "operators": sorted(operators),
    }
    cache_key = hashlib.sha256(_canonical_json(context)).hexdigest()
    cache_dir = _cache_directory(cache_dir)
    if not operators:
        return {
            "empty": True,
            "operator_count": 0,
            "cache_key": cache_key,
            "compilation_environment": environment_key,
            "install_path": None,
        }
    manifest = cache_dir / f"{cache_key}.json"
    with (cache_dir / f"{cache_key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = _valid_cached_result(manifest, cache_key)
        if cached is not None:
            logger.info("Static kernel cache hit: %s", cache_key)
            return cached
        previous = None
        if previous_result and isinstance(previous_result.get("cache_key"), str):
            key = previous_result["cache_key"]
            if len(key) == 64 and all(c in "0123456789abcdef" for c in key):
                previous = _valid_cached_result(cache_dir / f"{key}.json", key)
        if previous and (
            previous.get("compilation_environment") != environment_key
            or not set(previous.get("operator_signatures", [])) <= operators.keys()
        ):
            previous = None
        covered = set(previous["operator_signatures"]) if previous else set()
        pending = {key: path for key, path in operators.items() if key not in covered}
        # Keep one log set per key, independently of disposable inputs/packages.
        log_dir = cache_dir / "logs" / cache_key
        log_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(log_dir / "context.json", context)
        installations = cache_dir / "installations"
        installations.mkdir(exist_ok=True)
        fragment = _compile_fragment(pending, cache_dir, environment_key, compiler, soc, jobs)
        if fragment["empty"] and previous is None:
            raise ValueError(f"op_compiler produced no installation packages; inspect {fragment['log_dir']}")
        install_path = Path(tempfile.mkdtemp(prefix=f"{cache_key[:12]}-", dir=installations))
        try:
            if not fragment["empty"]:
                shutil.copytree(fragment["install_path"], install_path, dirs_exist_ok=True)
            if previous is not None:
                _extend_installation(Path(previous["install_path"]), install_path)
            result = {
                "schema": _CACHE_SCHEMA,
                "cache_key": cache_key,
                "cache_hit": False,
                "fragment_cache_hit": fragment["cache_hit"],
                "operator_count": len(operators),
                "new_operator_count": len(pending),
                "compiled_operator_count": 0 if fragment["cache_hit"] else len(pending),
                "operator_signatures": sorted(operators),
                "compilation_environment": environment_key,
                "install_path": str(install_path),
                "log_dir": fragment["log_dir"],
                "files": _installed_files(install_path),
            }
            _atomic_json(manifest, result)
        except Exception:
            # A failed merge must not leave a partial immutable entry.
            shutil.rmtree(install_path)
            raise
        logger.info("Static kernel installation ready: %s (%d operator descriptions)", install_path, len(operators))
        return result
