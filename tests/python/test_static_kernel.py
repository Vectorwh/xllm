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

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch.distributed as dist

from xllm.python.model_executor.static_kernel import StaticKernelWorker, _Rendezvous


def test_store_rounds_propagate_failure_cleanup_and_isolate_reinitialization() -> None:
    store = dist.TCPStore("127.0.0.1", 0, 2, True, wait_for_workers=False)
    peer = dist.TCPStore("127.0.0.1", store.port, 2, False, wait_for_workers=False)
    previous = [_Rendezvous(connection, "model", rank, 2) for rank, connection in enumerate((store, peer))]
    previous[0].publish(0, "package", {"ok": False})
    current = [_Rendezvous(connection, "model", rank, 2) for rank, connection in enumerate((store, peer))]
    assert current[0].identity == current[1].identity != previous[0].identity

    def run(rendezvous: _Rendezvous) -> None:
        for round_id, success in enumerate((True, False, True)):
            reports = rendezvous.exchange(round_id, "collected", {"rank": rendezvous.rank, "directories": []})
            assert [report["rank"] for report in reports] == [0, 1]
            if rendezvous.rank == 0:
                rendezvous.publish(round_id, "package", {"ok": success})
            package = rendezvous.receive(round_id, "package", [0])[0]
            assert package["ok"] == success
            assert rendezvous.exchange(round_id, "selected", package) == [package, package]
            rendezvous.finish(round_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, current))
    assert not store.check([f"{current[0].identity}/rounds/2/collected/1"])


def test_idle_rank_joins_preparation_without_dump_and_forward_failure_resets_runner() -> None:
    worker = StaticKernelWorker.__new__(StaticKernelWorker)
    worker.round_id = 0
    worker.state = SimpleNamespace(seal=MagicMock(return_value=[]), group=("local",), api=MagicMock())
    worker.rendezvous = SimpleNamespace(
        rank=1,
        identity="model",
        exchange=MagicMock(
            side_effect=[
                [
                    {"rank": 0, "group": ["local"], "directories": []},
                    {"rank": 1, "group": ["local"], "directories": []},
                ],
                [True, True],
            ]
        ),
        receive=MagicMock(return_value=[{"ok": True, "empty": True}]),
        finish=MagicMock(),
    )
    runner = SimpleNamespace(static_kernel=None)
    assert worker.prepare(lambda: "idle", [runner]) == ("idle", False)
    worker.rendezvous.finish.assert_called_once_with(0)
    worker.state.api._aclnn_reselect_static_kernel_with_path.assert_not_called()
    assert runner.static_kernel is None

    def fail() -> None:
        assert runner.static_kernel is worker
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        worker.prepare(fail, [runner])
    assert runner.static_kernel is None
