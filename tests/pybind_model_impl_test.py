# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

_ROOT = Path(__file__).resolve().parents[1]


class PublicModelImplTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._model = Path(self._directory.name)
        (self._model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))

        package = types.ModuleType("xllm")
        package.__path__ = [str(_ROOT / "xllm")]
        bindings = types.ModuleType("xllm.pybind")
        bindings.__path__ = [str(_ROOT / "xllm" / "pybind")]
        self._export = types.ModuleType("xllm_export")
        self._export.Options = types.SimpleNamespace
        self._export.RequestParams = types.SimpleNamespace
        self._export.RequestOutput = types.SimpleNamespace
        self._export.LLMMaster = Mock()
        self._export.VLMMaster = Mock()
        self._export.get_model_backend = Mock(return_value="llm")
        self._export.init_config_from_option = Mock(return_value=None)
        modules = {
            "xllm": package,
            "xllm.pybind": bindings,
            "xllm_export": self._export,
            "psutil": types.ModuleType("psutil"),
        }
        self._modules = patch.dict(sys.modules, modules)
        self._modules.start()
        self.addCleanup(self._modules.stop)
        for name in tuple(sys.modules):
            if name.startswith("xllm.pybind."):
                del sys.modules[name]
        self._signals = patch("signal.signal")
        self._signals.start()
        self.addCleanup(self._signals.stop)
        utils = importlib.import_module("xllm.pybind.utils")
        ports = patch.object(utils, "get_free_port", return_value=29500)
        ports.start()
        self.addCleanup(ports.stop)
        # These cases exercise the public API surface; bringing the real
        # accelerator up would need an NPU and would initialise torch_npu
        # inside the test process.
        runtime = patch.object(utils, "_configure_npu_runtime")
        self._npu_runtime = runtime.start()
        self.addCleanup(runtime.stop)

    def _construct(self, module_name: str, class_name: str, **kwargs: Any) -> Any:
        model_type = "qwen3_vl" if module_name == "vlm" else "qwen3"
        (self._model / "config.json").write_text(json.dumps({"model_type": model_type}))
        module = importlib.import_module(f"xllm.pybind.{module_name}")
        return getattr(module, class_name)(str(self._model), **kwargs)

    def test_constructors_forward_model_implementation_to_master(self) -> None:
        for module_name, class_name, master_name in (
            ("llm", "LLM", "LLMMaster"),
            ("vlm", "VLM", "VLMMaster"),
            ("embedding", "Embedding", "LLMMaster"),
        ):
            for value, expected in (("native", "native"), ("python", "python"), ("py", "python"), ("", "native")):
                with self.subTest(api=class_name, model_impl=value):
                    self._construct(module_name, class_name, model_impl=value)
                    options = getattr(self._export, master_name).call_args.args[0]
                    self.assertEqual(options.model_impl, expected)

    def test_vision_chunk_budget_reaches_offline_vlm_master(self) -> None:
        parser = importlib.import_module("xllm.pybind.args").ArgumentParser().parser
        (self._model / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
        self._export.get_model_backend.return_value = "vlm"
        for module_name, class_name in (("vlm", "VLM"), ("llm", "LLM")):
            for budget in (0, 2):
                with self.subTest(api=class_name, budget=budget):
                    arguments = vars(
                        parser.parse_args(
                            ["--model", str(self._model), "--max_vit_attention_units_per_chunk", str(budget)]
                        )
                    )
                    module = importlib.import_module(f"xllm.pybind.{module_name}")
                    if class_name == "LLM":
                        module.LLM(**arguments)
                    else:
                        module.VLM(
                            str(self._model),
                            max_vit_attention_units_per_chunk=arguments["max_vit_attention_units_per_chunk"],
                        )
                    options = self._export.VLMMaster.call_args.args[0]
                    self.assertEqual(options.max_vit_attention_units_per_chunk, budget)
                    self._export.init_config_from_option.assert_called_with(options, "qwen3_vl")

    def test_constructors_preserve_native_default(self) -> None:
        for module_name, class_name, master_name in (
            ("llm", "LLM", "LLMMaster"),
            ("vlm", "VLM", "VLMMaster"),
            ("embedding", "Embedding", "LLMMaster"),
        ):
            with self.subTest(api=class_name):
                self._construct(module_name, class_name)
                self.assertEqual(getattr(self._export, master_name).call_args.args[0].model_impl, "native")

    def test_invalid_implementation_does_not_create_master(self) -> None:
        for module_name, class_name in (("llm", "LLM"), ("vlm", "VLM"), ("embedding", "Embedding")):
            with self.subTest(api=class_name), self.assertRaisesRegex(ValueError, "unsupported model_impl"):
                self._construct(module_name, class_name, model_impl="unknown")
        self._export.LLMMaster.assert_not_called()
        self._export.VLMMaster.assert_not_called()

    def test_python_implementation_bootstraps_the_process_runtime(self) -> None:
        utilities = importlib.import_module("xllm.pybind.utils")
        with patch.object(utilities, "_configure_python_model_path") as model_path:
            self._construct("llm", "LLM", model_impl="python")
        model_path.assert_called_once_with()
        self._npu_runtime.assert_called_once_with(0)

    def test_native_implementation_leaves_the_process_runtime_alone(self) -> None:
        self._construct("llm", "LLM")
        self._npu_runtime.assert_not_called()

    def test_cli_selects_implementation_and_rejects_unknown_value(self) -> None:
        parser = importlib.import_module("xllm.pybind.args").ArgumentParser().parser
        self.assertEqual(parser.parse_args([]).model_impl, "native")
        self.assertEqual(parser.parse_args(["--model_impl", "python"]).model_impl, "python")
        self.assertEqual(parser.parse_args(["--model_impl", "py"]).model_impl, "py")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--model_impl", "unknown"])

    def test_offline_entries_reject_multi_node_before_master(self) -> None:
        # nnodes > 1 is not constructible offline: reject it with a Python
        # exception in the entry points instead of the C++ CHECK aborting the
        # process after the Options object has been built.
        for module_name, class_name in (("llm", "LLM"), ("vlm", "VLM"), ("embedding", "Embedding")):
            with self.subTest(api=class_name), self.assertRaisesRegex(ValueError, "nnodes"):
                self._construct(module_name, class_name, nnodes=2)
        self._export.LLMMaster.assert_not_called()
        self._export.VLMMaster.assert_not_called()

    def test_offline_shutdown_releases_owners_before_npu_and_preserves_exit_status(self) -> None:
        script = textwrap.dedent("""
            import atexit
            import gc
            import importlib
            import json
            import os
            import signal
            import sys
            import types
            import weakref
            from pathlib import Path

            root, model, module_name, class_name, release_owner, result_path = sys.argv[1:]
            package = types.ModuleType("xllm")
            package.__path__ = [str(Path(root) / "xllm")]
            bindings = types.ModuleType("xllm.pybind")
            bindings.__path__ = [str(Path(root) / "xllm" / "pybind")]
            export = types.ModuleType("xllm_export")
            export.Options = types.SimpleNamespace
            export.RequestParams = types.SimpleNamespace
            export.RequestOutput = types.SimpleNamespace
            export.init_config_from_option = lambda *args: None
            export.get_model_backend = lambda *args: "llm"
            sys.modules.update({"xllm": package, "xllm.pybind": bindings,
                                "xllm_export": export, "psutil": types.ModuleType("psutil")})
            events = []
            live_masters = 0
            comm_alive = True

            class Master:
                def __init__(self, options: object) -> None:
                    global live_masters
                    live_masters += 1
                    self.stopped = False

                def __del__(self) -> None:
                    self._shutdown()

                def _shutdown(self) -> None:
                    global live_masters
                    if self.stopped:
                        return
                    if release_owner == "error":
                        raise RuntimeError("injected native Master shutdown failure")
                    if release_owner == "foreign_error":
                        raise MemoryError("injected non-RuntimeError shutdown failure")
                    if release_owner == "signal":
                        os.kill(os.getpid(), signal.SIGTERM)
                    self.stopped = True
                    live_masters -= 1
                    events.append("master-stopped")

            def shutdown_comms() -> None:
                global comm_alive
                if live_masters:
                    os._exit(89)
                comm_alive = False
                events.append("comms-released")

            def shutdown_npu() -> None:
                events.append("npu-shutdown")
                Path(result_path).write_text(json.dumps({"events": events,
                    "live_masters": live_masters, "comm_alive": comm_alive}))
                if live_masters or comm_alive:
                    os._exit(88)

            export.LLMMaster = Master
            export.VLMMaster = Master
            export._arm_shutdown_deadline = lambda: events.append("deadline")
            export._shutdown_python_host = shutdown_comms
            # Model construction happens after the host imports torch_npu.
            atexit.register(shutdown_npu)
            module = importlib.import_module("xllm.pybind." + module_name)
            owner = getattr(module, class_name)(model)
            if release_owner == "alias":
                retained_master = owner.master
            if release_owner in ("release", "alias"):
                reference = weakref.ref(owner)
                del owner
                gc.collect()
                if reference() is not None:
                    os._exit(87)
            sys.exit(7)
        """)
        for module_name, class_name in (("llm", "LLM"), ("vlm", "VLM"), ("embedding", "Embedding")):
            ownership_modes = ("retain", "release", "alias")
            if module_name == "llm":
                ownership_modes += ("error", "foreign_error", "signal")
            for release_owner in ownership_modes:
                with self.subTest(api=class_name, owner=release_owner):
                    result_path = self._model / f"shutdown-{module_name}-{release_owner}.json"
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            script,
                            str(_ROOT),
                            str(self._model),
                            module_name,
                            class_name,
                            release_owner,
                            str(result_path),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=15,
                        check=False,
                    )
                    if release_owner == "signal":
                        # A signal during terminal cleanup must not raise
                        # SystemExit inside atexit; the process exits 1 before
                        # later NPU callbacks run.
                        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                        self.assertNotIn("Exception ignored in atexit callback", result.stderr)
                        self.assertFalse(result_path.exists(), "NPU shutdown ran after a terminal cleanup signal")
                        continue
                    if release_owner in ("error", "foreign_error"):
                        message = {
                            "error": "injected native Master shutdown failure",
                            "foreign_error": "injected non-RuntimeError shutdown failure",
                        }[release_owner]
                        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                        self.assertIn(message, result.stderr)
                        self.assertFalse(result_path.exists(), "NPU shutdown ran after failed Master teardown")
                        continue
                    self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
                    state = json.loads(result_path.read_text())
                    self.assertEqual(state["live_masters"], 0)
                    self.assertFalse(state["comm_alive"])
                    self.assertLess(state["events"].index("master-stopped"), state["events"].index("comms-released"))
                    self.assertLess(state["events"].index("comms-released"), state["events"].index("npu-shutdown"))


class PythonModelPathTest(unittest.TestCase):
    """XLLM_PYTHON_MODEL_PATH must name the directory that holds the ``xllm``
    package for both shapes the package ships in: a wheel install (the package
    lives in site-packages) and an in-tree run (the package lives in the
    repository). Spawned workers create their own interpreter and only prepend
    this one directory, so a wrong answer breaks the Python backend in exactly
    one of the two shapes."""

    _VARIABLE = "XLLM_PYTHON_MODEL_PATH"

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._root = Path(self._directory.name)
        modules = {
            "psutil": types.ModuleType("psutil"),
            "xllm_export": types.SimpleNamespace(),
        }
        self._modules = patch.dict(sys.modules, modules)
        self._modules.start()
        self.addCleanup(self._modules.stop)
        environment = patch.dict(os.environ, {}, clear=False)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop(self._VARIABLE, None)
        self.addCleanup(os.environ.pop, self._VARIABLE, None)

    def _load_utils(self, package_root: Path) -> types.ModuleType:
        """Import ``utils.py`` from ``<package_root>/xllm/pybind/utils.py``."""
        source = _ROOT / "xllm" / "pybind" / "utils.py"
        target = package_root / "xllm" / "pybind" / "utils.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        spec = importlib.util.spec_from_file_location(f"_model_path_probe_{abs(hash(str(package_root)))}", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_site_packages_layout_points_at_the_installation_prefix(self) -> None:
        site_packages = self._root / "lib" / "python3.12" / "site-packages"
        (site_packages / "xllm" / "python").mkdir(parents=True)
        utils = self._load_utils(site_packages)

        utils._configure_python_model_path()

        self.assertEqual(os.environ[self._VARIABLE], str(site_packages.resolve()))
        self.assertTrue((Path(os.environ[self._VARIABLE]) / "xllm" / "python").is_dir())

    def test_source_tree_layout_points_at_the_repository_root(self) -> None:
        utils = self._load_utils(self._root)

        utils._configure_python_model_path()

        self.assertEqual(os.environ[self._VARIABLE], str(self._root.resolve()))
        self.assertTrue((Path(os.environ[self._VARIABLE]) / "xllm" / "pybind" / "utils.py").is_file())

    def test_preset_value_is_not_overwritten(self) -> None:
        site_packages = self._root / "lib" / "python3.12" / "site-packages"
        (site_packages / "xllm" / "python").mkdir(parents=True)
        os.environ[self._VARIABLE] = str(site_packages)
        utils = self._load_utils(self._root / "build")

        utils._configure_python_model_path()

        self.assertEqual(os.environ[self._VARIABLE], str(site_packages))

    def test_invalid_preset_value_fails_instead_of_deferring(self) -> None:
        os.environ[self._VARIABLE] = "/opt/custom/does-not-hold-the-package"
        utils = self._load_utils(self._root / "build")

        with self.assertRaisesRegex(RuntimeError, "/opt/custom"):
            utils._configure_python_model_path()
        # The rejected preset is left untouched: this process did not publish
        # it, so it has no business cleaning it up either.
        self.assertEqual(os.environ[self._VARIABLE], "/opt/custom/does-not-hold-the-package")

    def test_relocated_utils_fails_instead_of_guessing(self) -> None:
        # Drop utils.py two directories below the temporary root without any
        # 'xllm' package around it, so the three-level backtracking lands
        # inside the temporary root and finds no package there: it must fail
        # loudly instead of publishing the directory to spawned workers.
        source = _ROOT / "xllm" / "pybind" / "utils.py"
        target = self._root / "any" / "nested" / "utils.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        spec = importlib.util.spec_from_file_location("_model_path_probe_stray", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with self.assertRaisesRegex(RuntimeError, "XLLM_PYTHON_MODEL_PATH"):
            module._configure_python_model_path()
        self.assertNotIn(self._VARIABLE, os.environ)


if __name__ == "__main__":
    unittest.main()
