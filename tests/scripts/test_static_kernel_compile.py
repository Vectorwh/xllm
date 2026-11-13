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

"""Essential CPU coverage for static-kernel cache selection and integrity."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "static_kernel_compile", _ROOT / "xllm/python/model_executor/static_kernel_compile.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_TOOLCHAIN_FINGERPRINT = _MODULE._toolchain_fingerprint


class StaticKernelCompileTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        toolchain = mock.patch.object(_MODULE, "_toolchain_fingerprint", return_value={"version": "test"})
        toolchain.start()
        self.addCleanup(toolchain.stop)
        environment = mock.patch.dict(_MODULE.os.environ, {"XLLM_STATIC_KERNEL_BLACKLIST": ""})
        environment.start()
        self.addCleanup(environment.stop)

    def _install(self) -> tuple[str, Path]:
        install = self.root / "install"
        core = install / "static_kernel/ai_core"
        index = core / "config/test/binary_info_config.json"
        index.parent.mkdir(parents=True)
        index.write_text(json.dumps({"Add": [{"binPath": "kernel/Add.o"}]}), encoding="utf-8")
        binary = core / "kernel/Add.o"
        binary.parent.mkdir()
        binary.write_bytes(b"kernel")
        key = "a" * 64
        manifest = {
            "schema": _MODULE._CACHE_SCHEMA,
            "cache_key": key,
            "install_path": str(install),
            "files": _MODULE._installed_files(install),
        }
        (self.cache / f"{key}.json").write_text(json.dumps(manifest), encoding="utf-8")
        return key, binary

    def test_collection_deduplicates_exact_descriptions_without_dropping_attributes(self) -> None:
        folder = self.root / "dump/rank_opcompile"
        folder.mkdir(parents=True)
        descriptions = [
            {"op_type": "Add", "shape": [1, 8], "attr": 1},
            {"attr": 1, "shape": [1, 8], "op_type": "Add"},
            {"op_type": "Add", "shape": [1, 8], "attr": 2},
            {"op_type": "RepeatInterleave", "shape": [1, 8]},
        ]
        for i, value in enumerate(descriptions):
            (folder / f"op{i}.json").write_text(json.dumps(value), encoding="utf-8")
        idle = self.root / "idle_worker"
        idle.mkdir()
        result = _MODULE._operator_inputs([idle, self.root / "dump"])
        observed = [json.loads(path.read_text(encoding="utf-8")) for path in result.values()]
        self.assertEqual(len(observed), 3)
        self.assertEqual({value["attr"] for value in observed if value["op_type"] == "Add"}, {1, 2})
        self.assertIn("RepeatInterleave", {value["op_type"] for value in observed})
        (folder / "mul.json").write_text(json.dumps({"op_type": "Mul"}), encoding="utf-8")
        with mock.patch.dict(_MODULE.os.environ, {"XLLM_STATIC_KERNEL_BLACKLIST": " Mul, RepeatInterleave, Mul, , "}):
            self.assertEqual(_MODULE._blacklisted_ops(), ["Mul", "RepeatInterleave"])
            self.assertEqual(len(_MODULE._operator_inputs([self.root / "dump"])), 2)
            _MODULE.os.environ["XLLM_STATIC_KERNEL_BLACKLIST"] = " Add, Add, , "
            self.assertEqual(_MODULE._blacklisted_ops(), ["Add"])
            selected = _MODULE._operator_inputs([self.root / "dump"])
            self.assertEqual(
                {json.loads(path.read_text(encoding="utf-8"))["op_type"] for path in selected.values()},
                {"Mul", "RepeatInterleave"},
            )
            _MODULE.os.environ["XLLM_STATIC_KERNEL_BLACKLIST"] = ""
            self.assertEqual(_MODULE._blacklisted_ops(), [])
            self.assertEqual(len(_MODULE._operator_inputs([self.root / "dump"])), 4)
            del _MODULE.os.environ["XLLM_STATIC_KERNEL_BLACKLIST"]
            self.assertEqual(_MODULE._blacklisted_ops(), [])

    def test_vendor_fingerprint_tracks_content_instead_of_timestamps(self) -> None:
        compiler = self.root / "op_compiler"
        compiler.write_bytes(b"compiler")
        payload = self.root / "opp/vendors/test/kernel.o"
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"old")
        metadata = payload.stat()
        with mock.patch.dict(
            _MODULE.os.environ,
            {
                "ASCEND_HOME_PATH": str(self.root),
                "ASCEND_OPP_PATH": str(self.root / "opp"),
                "ASCEND_CUSTOM_OPP_PATH": "",
            },
        ):
            baseline = _TOOLCHAIN_FINGERPRINT(compiler)
            os.utime(payload, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000))
            self.assertEqual(baseline, _TOOLCHAIN_FINGERPRINT(compiler))
            payload.write_bytes(b"new")
            os.utime(payload, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
            self.assertNotEqual(baseline, _TOOLCHAIN_FINGERPRINT(compiler))

    def test_corrupt_kernel_is_never_a_cache_hit(self) -> None:
        key, binary = self._install()
        manifest = self.cache / f"{key}.json"
        self.assertTrue(_MODULE._valid_cached_result(manifest, key)["cache_hit"])
        binary.write_bytes(b"corrupted")
        self.assertIsNone(_MODULE._valid_cached_result(manifest, key))
        manifest.write_text("truncated JSON", encoding="utf-8")
        self.assertIsNone(_MODULE._valid_cached_result(manifest, key))

    def test_compile_cleans_temporary_files_and_preserves_installations(self) -> None:
        dump = self.root / "dump/rank_opcompile"
        dump.mkdir(parents=True)
        (dump / "add.json").write_text(json.dumps({"op_type": "Add", "shape": [1, 8]}), encoding="utf-8")
        compiler = self.root / "op_compiler"
        for failure in (None, "compile", "install"):
            with self.subTest(failure=failure):
                cache = self.root / str(failure)

                def run(arguments: list[str], log: Path, stage_failure: str | None = failure) -> None:
                    log.write_text("tool output", encoding="utf-8")
                    stage = "compile" if arguments[0] == str(compiler) else "install"
                    if stage == stage_failure:
                        raise subprocess.CalledProcessError(1, arguments)
                    if stage == "compile":
                        output = Path(arguments[arguments.index("-o") + 1])
                        (output / "test.run").write_text("fake installer", encoding="utf-8")
                    else:
                        install = Path(
                            next(arg.split("=", 1)[1] for arg in arguments if arg.startswith("--install-path="))
                        )
                        core = install / "static_kernel/ai_core"
                        index = core / "config/test/binary_info_config.json"
                        index.parent.mkdir(parents=True)
                        index.write_text(json.dumps({"Add": [{"binPath": "kernel/Add.o"}]}), encoding="utf-8")
                        binary = core / "kernel/Add.o"
                        binary.parent.mkdir()
                        binary.write_bytes(b"kernel")

                with mock.patch.object(_MODULE, "_run", side_effect=run) as runner:
                    if failure:
                        with self.assertRaises(subprocess.CalledProcessError):
                            _MODULE.compile_static_kernels(self.root / "dump", cache, "test", compiler=compiler)
                        self.assertEqual(list((cache / "installations").iterdir()), [])
                        self.assertEqual(list(cache.glob("*.json")), [])
                    else:
                        result = _MODULE.compile_static_kernels(self.root / "dump", cache, "test", compiler=compiler)
                        install = Path(result["install_path"])
                        self.assertTrue(install.is_relative_to(cache / "installations"))
                        self.assertTrue((install / "static_kernel/ai_core/kernel/Add.o").is_file())
                        calls = runner.call_count
                        self.assertTrue(
                            _MODULE.compile_static_kernels(self.root / "dump", cache, "test", compiler=compiler)[
                                "cache_hit"
                            ]
                        )
                        self.assertEqual(runner.call_count, calls)
                self.assertEqual(list(cache.glob("build-*")), [])
                self.assertEqual(len(list((cache / "logs").rglob("compile.log"))), 1)

    def test_incremental_shapes_preserve_old_kernels_and_skip_covered_inputs(self) -> None:
        dump = self.root / "round32/rank_opcompile"
        dump.mkdir(parents=True)
        (dump / "matmul.json").write_text(json.dumps({"op_type": "MatMulV2", "rows": 32}), encoding="utf-8")
        compiler = self.root / "op_compiler"
        compiled_rows: list[list[int]] = []

        def run(arguments: list[str], log: Path) -> None:
            log.write_text("tool output", encoding="utf-8")
            if arguments[0] == str(compiler):
                inputs = Path(arguments[arguments.index("-p") + 1])
                rows = [json.loads(path.read_text())["rows"] for path in inputs.glob("*.json")]
                compiled_rows.append(rows)
                output = Path(arguments[arguments.index("-o") + 1])
                (output / "test.run").write_text(json.dumps(rows), encoding="utf-8")
            else:
                rows = json.loads(Path(arguments[1]).read_text())
                install = Path(next(arg.split("=", 1)[1] for arg in arguments if arg.startswith("--install-path=")))
                core = install / "static_kernel/ai_core"
                index = core / "config/test/binary_info_config.json"
                index.parent.mkdir(parents=True, exist_ok=True)
                binaries = core / "kernel"
                binaries.mkdir(exist_ok=True)
                entries = []
                for row in rows:
                    (binaries / f"MatMul{row}.o").write_bytes(f"kernel{row}".encode())
                    entries.append({"binPath": f"kernel/MatMul{row}.o", "simplifiedKey": str(row)})
                index.write_text(json.dumps({"MatMulV2": {"staticList": entries}}), encoding="utf-8")

        with mock.patch.object(_MODULE, "_run", side_effect=run) as runner:
            first = _MODULE.compile_static_kernels(dump.parent, self.cache, "test", compiler=compiler)
            smaller = self.root / "round16/rank_opcompile"
            smaller.mkdir(parents=True)
            (smaller / "matmul.json").write_text(json.dumps({"op_type": "MatMulV2", "rows": 16}), encoding="utf-8")
            second = _MODULE.compile_static_kernels(
                smaller.parent,
                self.cache,
                "test",
                compiler=compiler,
                dump_dirs=[dump.parent, smaller.parent],
                previous_result=first,
            )
            repeated = _MODULE.compile_static_kernels(
                dump.parent,
                self.cache,
                "test",
                compiler=compiler,
                dump_dirs=[dump.parent, smaller.parent],
                previous_result=second,
            )
            self.assertEqual(compiled_rows, [[32], [16]])
            self.assertEqual(runner.call_count, 4)
            self.assertTrue(repeated["cache_hit"])
            self.assertEqual(repeated["new_operator_count"], 0)
            self.assertEqual(second["operator_count"], 2)
            self.assertEqual(second["new_operator_count"], 1)
            first_index = Path(first["install_path"]) / "static_kernel/ai_core/config/test/binary_info_config.json"
            second_index = Path(second["install_path"]) / "static_kernel/ai_core/config/test/binary_info_config.json"
            self.assertEqual(len(json.loads(first_index.read_text())["MatMulV2"]["staticList"]), 1)
            self.assertEqual(len(json.loads(second_index.read_text())["MatMulV2"]["staticList"]), 2)
            for row in [32, 16]:
                self.assertEqual(
                    (Path(second["install_path"]) / f"static_kernel/ai_core/kernel/MatMul{row}.o").read_bytes(),
                    f"kernel{row}".encode(),
                )
            # A process-dependent attribute changes the first collection, but
            # must not make the unchanged later shape compile again.
            (dump / "matmul.json").write_text(
                json.dumps({"op_type": "MatMulV2", "rows": 32, "runtime_attribute": 1}), encoding="utf-8"
            )
            restarted = _MODULE.compile_static_kernels(dump.parent, self.cache, "test", compiler=compiler)
            resumed = _MODULE.compile_static_kernels(
                smaller.parent,
                self.cache,
                "test",
                compiler=compiler,
                dump_dirs=[dump.parent, smaller.parent],
                previous_result=restarted,
            )
            self.assertEqual(compiled_rows, [[32], [16], [32]])
            self.assertEqual(runner.call_count, 6)
            self.assertFalse(resumed["cache_hit"])
            self.assertTrue(resumed["fragment_cache_hit"])
            self.assertEqual(resumed["new_operator_count"], 1)
            self.assertEqual(resumed["compiled_operator_count"], 0)

    def test_compilation_environment_and_empty_participant(self) -> None:
        compiler = self.root / "op_compiler"
        baseline = _MODULE._compilation_environment("test", False, compiler)
        self.assertNotEqual(baseline, _MODULE._compilation_environment("test", True, compiler))
        with mock.patch.dict(_MODULE.os.environ, {"XLLM_STATIC_KERNEL_BLACKLIST": "Cast"}):
            self.assertNotEqual(baseline, _MODULE._compilation_environment("test", False, compiler))
        with mock.patch.dict(_MODULE.os.environ, {"CLOSE_MATMUL_K_SHIFT": "1"}):
            changed = _MODULE._compilation_environment("test", False, compiler)
            with mock.patch.dict(_MODULE.os.environ, {"CLOSE_MATMUL_K_SHIFT": "0"}):
                self.assertNotEqual(changed, _MODULE._compilation_environment("test", False, compiler))
        with mock.patch.object(_MODULE, "_run") as runner:
            result = _MODULE.compile_static_kernels(self.root, self.cache, "test", compiler=compiler, dump_dirs=[])
            self.assertTrue(result["empty"])
            runner.assert_not_called()
        self.cache.chmod(0o777)
        with self.assertRaisesRegex(ValueError, "not writable by others"):
            _MODULE.compile_static_kernels(self.root, self.cache, "test", compiler=compiler, dump_dirs=[])
        with (
            mock.patch.object(_MODULE.shutil, "which", return_value=None),
            self.assertRaisesRegex(ValueError, "op_compiler is not on PATH"),
        ):
            _MODULE.compile_static_kernels(self.root, self.cache, "test", dump_dirs=[])


if __name__ == "__main__":
    unittest.main()
