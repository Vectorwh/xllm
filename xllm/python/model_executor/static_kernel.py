# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://github.com/xLLM-AI/xllm/blob/main/LICENSE
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Worker-local static kernel preparation before ACL graph capture."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import torch
import torch.distributed as dist

from scripts.logger import logger
from xllm.python.model_executor.static_kernel_compile import (
    _cache_directory,
    _canonical_json,
    _compilation_environment,
    _compiler_path,
    compile_static_kernels,
)

if TYPE_CHECKING:
    from xllm.python.model_executor.runner import Runner

_TIMEOUT = timedelta(hours=1)
_Output = TypeVar("_Output")


class _Rendezvous:
    def __init__(self, store: dist.Store, model: str, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
        prefix = "xllm/static_kernel/" + model
        generation = store.add(f"{prefix}/initializations/{rank}", 1)
        self.identity = f"{prefix}/{generation}"
        self.store = dist.PrefixStore(self.identity, store)

    def _key(self, round_id: int, phase: str, rank: int) -> str:
        return f"rounds/{round_id}/{phase}/{rank}"

    def publish(self, round_id: int, phase: str, value: Any) -> None:
        self.store.set(self._key(round_id, phase, self.rank), json.dumps(value))

    def receive(self, round_id: int, phase: str, ranks: Sequence[int]) -> list[Any]:
        keys = [self._key(round_id, phase, rank) for rank in ranks]
        self.store.wait(keys, _TIMEOUT)
        return [json.loads(self.store.get(key)) for key in keys]

    def exchange(self, round_id: int, phase: str, value: Any) -> list[Any]:
        self.publish(round_id, phase, value)
        return self.receive(round_id, phase, range(self.world_size))

    def finish(self, round_id: int) -> None:
        self.publish(round_id, "finished", True)
        if self.rank != 0:
            return
        self.receive(round_id, "finished", range(self.world_size))
        for rank in range(self.world_size):
            for phase in ("collected", "package", "selected", "finished"):
                self.store.delete_key(self._key(round_id, phase, rank))


class _ProcessState:
    """CANN argument dumping is process-wide, including local worker threads."""

    def __init__(self, cache: Path, device: torch.device) -> None:
        import torch_npu

        self.api = torch_npu._C
        required = ("_aclop_start_dump", "_aclop_stop_dump", "_aclnn_reselect_static_kernel_with_path")
        missing = [name for name in required if not callable(getattr(self.api, name, None))]
        if missing:
            raise RuntimeError(f"Static Kernel requires TorchNPU interfaces: {', '.join(missing)}")
        self.cache = _cache_directory(cache)
        sessions = self.cache / "sessions"
        sessions.mkdir(exist_ok=True)
        self.session = Path(tempfile.mkdtemp(prefix="worker-", dir=sessions))
        self.soc = torch.npu.get_device_name(device)
        environment = _compilation_environment(self.soc, torch.are_deterministic_algorithms_enabled(), _compiler_path())
        environment_key = hashlib.sha256(_canonical_json(environment)).hexdigest()
        metadata = self.cache.stat()
        self.group = (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            str(self.cache),
            metadata.st_dev,
            metadata.st_ino,
            environment_key,
        )
        self.condition = threading.Condition()
        self.members: dict[str, set[int]] = {}
        self.completed: dict[tuple[str, int], set[int]] = {}
        self.readers: dict[tuple[str, int], int] = {}
        self.active = 0
        self.directory: Path | None = None
        self.directories: list[Path] = []
        self.previous: dict[str, Any] | None = None

    @contextmanager
    def collect(self, device: torch.device) -> Iterator[None]:
        torch.npu.synchronize(device)
        with self.condition:
            if self.active == 0:
                self.directory = Path(tempfile.mkdtemp(prefix="forward-", dir=self.session))
                self.api._aclop_start_dump(str(self.directory))
            self.active += 1
        try:
            yield
        finally:
            torch.npu.synchronize(device)
            with self.condition:
                self.active -= 1
                if self.active == 0:
                    self.api._aclop_stop_dump()
                    assert self.directory is not None
                    self.directories.append(self.directory)
                    self.condition.notify_all()

    def seal(self, identity: str, round_id: int, rank: int) -> list[str]:
        key = (identity, round_id)
        with self.condition:
            completed = self.completed.setdefault(key, set())
            if rank in completed:
                raise RuntimeError(f"Repeated static kernel collection: {key}, rank={rank}")
            completed.add(rank)
            expected = len(self.members[identity])
            self.condition.notify_all()
            if not self.condition.wait_for(
                lambda: len(completed) == expected and self.active == 0, timeout=_TIMEOUT.total_seconds()
            ):
                raise TimeoutError(f"Static kernel local collection timeout: {key}, completed={completed}")
            directories = [str(path) for path in self.directories]
            self.readers[key] = self.readers.get(key, 0) + 1
            if self.readers[key] == expected:
                del self.completed[key], self.readers[key]
            return directories


def _unregister(state: _ProcessState, identity: str, rank: int) -> None:
    with state.condition:
        members = state.members[identity]
        members.remove(rank)
        if not members:
            del state.members[identity]


_state_lock = threading.Lock()
_process_state: _ProcessState | None = None


class StaticKernelWorker:
    def __init__(self, config: dict[str, Any], device: torch.device, *, is_spec_draft: bool) -> None:
        from xllm.python.distributed.collectives import _shared_store

        rank, world_size = int(config["rank"]), int(config["world_size"])
        store = _shared_store(
            config["python_rendezvous_host"], int(config["python_rendezvous_port"]), rank, world_size
        )
        model = json.dumps(
            {"model": config["model_path"], "model_type": config["model_type"], "draft": is_spec_draft}, sort_keys=True
        )
        self.rendezvous = _Rendezvous(store, model, rank, world_size)
        self.device = device
        self.round_id = 0
        self.graph_inputs = False
        configured = config.get("static_kernel_cache_dir")
        cache = (
            Path(configured)
            if configured
            else Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "xllm/static_kernel"
        )
        global _process_state
        with _state_lock:
            if _process_state is None:
                _process_state = _ProcessState(cache, device)
            elif _process_state.cache != cache.resolve():
                raise ValueError("Local Static Kernel workers must use the same cache directory")
            self.state = _process_state
            with self.state.condition:
                self.state.members.setdefault(self.rendezvous.identity, set()).add(rank)
        self._finalizer = weakref.finalize(self, _unregister, self.state, self.rendezvous.identity, rank)

    @contextmanager
    def collect_forward(self) -> Iterator[None]:
        self.graph_inputs = True
        with self.state.collect(self.device):
            yield

    def prepare(self, forward: Callable[[], _Output], runners: Sequence[Runner]) -> tuple[_Output, bool]:
        rendezvous, state = self.rendezvous, self.state
        round_id = self.round_id
        self.round_id += 1
        self.graph_inputs = False
        logger.info(
            "Static kernel warmup collection begin: rank=%d, model=%s, round=%d",
            rendezvous.rank,
            rendezvous.identity,
            round_id,
        )
        if any(runner.static_kernel is not None for runner in runners):
            raise RuntimeError("Static kernel collection must not reenter model execution")
        for runner in runners:
            runner.static_kernel = self
        try:
            output = forward()
        finally:
            for runner in runners:
                runner.static_kernel = None
        directories = state.seal(rendezvous.identity, round_id, rendezvous.rank)
        reports = rendezvous.exchange(
            round_id, "collected", {"rank": rendezvous.rank, "group": state.group, "directories": directories}
        )
        local = [report for report in reports if report["group"] == list(state.group)]
        compiler_rank = min(report["rank"] for report in local)
        if rendezvous.rank == compiler_rank:
            inputs = sorted({directory for report in local for directory in report["directories"]})
            logger.info("Static kernel compilation: round=%d, directories=%d", round_id, len(inputs))
            # Compilation is a startup/tool boundary. Publish failures so peers
            # fail with the compiler error instead of waiting for a missing key.
            try:
                package = compile_static_kernels(
                    state.session,
                    state.cache,
                    state.soc,
                    torch.are_deterministic_algorithms_enabled(),
                    jobs=max(1, min(8, (os.cpu_count() or 1) // 4)),
                    dump_dirs=[Path(path) for path in inputs],
                    previous_result=state.previous,
                )
                package = {key: value for key, value in package.items() if key not in ("files", "operator_signatures")}
                package["ok"] = True
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                package = {"ok": False, "error": str(error)}
            rendezvous.publish(round_id, "package", package)
        package = rendezvous.receive(round_id, "package", [compiler_rank])[0]
        if not package["ok"]:
            raise RuntimeError(f"Static kernel compilation failed: {package}")
        if not package.get("empty", False):
            with torch.npu.device(self.device):
                torch.npu.synchronize()
                # TorchNPU drains its host queue and selects on the execution
                # thread chosen by TASK_QUEUE_ENABLE before returning.
                state.api._aclnn_reselect_static_kernel_with_path(package["install_path"])
                torch.npu.synchronize()
        rendezvous.exchange(round_id, "selected", True)
        rendezvous.finish(round_id)
        state.previous = package
        logger.info(
            "Static kernel warmup selected: rank=%d, round=%d, graph_inputs=%d, package=%s",
            rendezvous.rank,
            round_id,
            self.graph_inputs,
            json.dumps(package),
        )
        return output, self.graph_inputs
