import hashlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from tools import lamina_release as release
from tools import quickstart
from tools.package_windows import validate_build, validate_gpu_vision


def engine_fixture(directory):
    directory.mkdir(parents=True, exist_ok=True)
    files = {"lamina-infer.exe": b"fake engine", "cublas64_13.dll": b"fake cublas",
             "cublasLt64_13.dll": b"fake cublasLt", "cudart64_13.dll": b"fake cudart"}
    for name, data in files.items():
        (directory / name).write_bytes(data)
    (directory / "release.json").write_text(json.dumps({
        "version": release.VERSION, "cuda_architectures": list(release.ARCHITECTURES),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}))


class ReleaseTest(unittest.TestCase):
    def test_gpu_vision_release_requires_portable_cuda_and_all_architectures(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            values = {"STRATA_PORTABLE": "ON", "STRATA_VISION_CUDA": "ON",
                      "CMAKE_BUILD_TYPE": "Release", "CMAKE_CUDA_ARCHITECTURES": "86;89;120"}
            for field, invalid in (("STRATA_PORTABLE", "OFF"), ("STRATA_VISION_CUDA", "OFF"),
                                   ("CMAKE_BUILD_TYPE", "Debug"), ("CMAKE_CUDA_ARCHITECTURES", "89")):
                with self.subTest(field=field):
                    (directory / "CMakeCache.txt").write_text("\n".join(
                        f"{key}:STRING={invalid if key == field else value}" for key, value in values.items()))
                    with self.assertRaises(ValueError): validate_gpu_vision(directory)
            (directory / "CMakeCache.txt").write_text("\n".join(f"{k}:STRING={v}" for k, v in values.items()))
            self.assertEqual(validate_gpu_vision(directory)["CMAKE_CUDA_ARCHITECTURES"], "86;89;120")

    def test_download_installs_only_verified_archive(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture"
            engine_fixture(fixture)
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                for path in fixture.iterdir():
                    archive.write(path, path.name)
            payload = buffer.getvalue()
            checksum = hashlib.sha256(payload).hexdigest().encode()
            for good_checksum in (False, True):
                with self.subTest(good_checksum=good_checksum):
                    data = root / str(good_checksum)
                    responses = [BytesIO(checksum if good_checksum else b"0" * 64), BytesIO(payload)]
                    with patch.object(release.urllib.request, "urlopen", side_effect=responses):
                        if good_checksum:
                            path = release.install_engine(data, "89")
                            self.assertEqual(path.read_bytes(), b"fake engine")
                        else:
                            with self.assertRaisesRegex(ValueError, "archive checksum mismatch"):
                                release.install_engine(data, "89")
                            self.assertFalse((data / "engine" / release.VERSION).exists())

    def test_cached_engine_verified_without_network(self):
        with TemporaryDirectory() as temporary:
            data = Path(temporary)
            directory = data / "engine" / release.VERSION
            engine_fixture(directory)
            with patch.object(release.urllib.request, "urlopen", side_effect=AssertionError("network")):
                self.assertEqual(release.install_engine(data, "86"), directory / "lamina-infer.exe")
                (directory / "lamina-infer.exe").write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    release.install_engine(data, "86")

    def test_wrong_version_and_architecture_rejected(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            engine_fixture(directory)
            with self.assertRaisesRegex(ValueError, "does not support"):
                release.validate_runtime(directory, "75")
            manifest = json.loads((directory / "release.json").read_text())
            manifest["version"] = "v0.0.0"
            (directory / "release.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "does not support"):
                release.validate_runtime(directory, "89")

    def test_zip_traversal_rejected_before_any_write(self):
        for path in ("../escaped", "/absolute", "C:/escaped", "dir/../../escaped", "..\\escaped", "engine.exe:stream"):
            with self.subTest(path=path), TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "test.zip"
                with zipfile.ZipFile(archive, "w") as out:
                    out.writestr("valid", b"ok")
                    out.writestr(path, b"bad")
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    release.safe_extract(archive, root / "out")
                self.assertFalse((root / "out/valid").exists())

    def test_no_release_for_unsupported_gpu(self):
        with patch.object(release.urllib.request, "urlopen", side_effect=AssertionError("network")):
            with self.assertRaisesRegex(ValueError, "--build-source"):
                release.install_engine(Path("unused"), "75")

    def test_packager_refuses_single_gpu_and_nonportable_build(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / "CMakeCache.txt"
            cache.write_text("STRATA_PORTABLE:BOOL=OFF\n")
            with self.assertRaisesRegex(ValueError, "portable"):
                validate_build(directory)
            cache.write_text("\n".join(f"{key}:BOOL=ON" for key in
                ("STRATA_PORTABLE", "LAMINA_ENABLE_CUDA", "LAMINA_PREFILL_BLAS", "LAMINA_CPU_QUANT")) +
                "\nCMAKE_CUDA_ARCHITECTURES:STRING=89\nCMAKE_BUILD_TYPE:STRING=Release\n")
            with self.assertRaisesRegex(ValueError, "all GPU architectures"):
                validate_build(directory)

    def test_required_requirements_adds_build_toolchain_for_local_compiles(self):
        runtime = quickstart.required_requirements({"vision": True, "vision_device": "cpu"})
        self.assertEqual(runtime, quickstart.RUNTIME_REQUIREMENTS)
        self.assertEqual(quickstart.required_requirements({"vision": False}), quickstart.RUNTIME_REQUIREMENTS)
        self.assertEqual(quickstart.required_requirements({"vision": True, "vision_device": "gpu"}), quickstart.REQUIREMENTS)
        self.assertEqual(quickstart.required_requirements({}, build_source=True), quickstart.REQUIREMENTS)

    def test_source_gpu_vision_installs_build_toolchain_before_compiling(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            for name in quickstart.REQUIREMENTS:
                (root / name).write_text(name)
            from tools.lamina_model import FILENAME
            (data / "models").mkdir()
            (data / "tokenizer").mkdir()
            (data / "models" / FILENAME).touch()
            (data / "tokenizer/tokenizer.json").touch()
            engine = root / "build-cuda/lamina-infer.exe"
            engine.parent.mkdir(parents=True)
            engine.touch()
            (engine.parent / "CMakeCache.txt").write_text("CMAKE_CUDA_ARCHITECTURES:STRING=89\n")
            encoder = root / "build-vision-gpu/bin/strata-vision.exe"
            commands = []
            selected = iter([None, encoder, encoder])
            with patch.object(quickstart, "ROOT", root), patch.object(quickstart, "DATA", data), \
                    patch.object(quickstart, "CONFIG", data / "quickstart.json"), \
                    patch.object(quickstart, "ENGINE", engine), \
                    patch.object(quickstart, "portable", return_value=False), \
                    patch.object(quickstart, "selected_vision", side_effect=lambda config: next(selected)), \
                    patch.object(quickstart, "gpu", return_value=("RTX 4060", "89", 8192)), \
                    patch.object(quickstart, "total_ram_gib", return_value=64), \
                    patch.object(quickstart, "run", side_effect=lambda args: commands.append(args)):
                quickstart.setup({"model": "qwen3.6", "vision": True, "vision_device": "gpu", "mtp": False})
            install = commands[0]
            self.assertIn("install", install)
            self.assertIn("requirements-build.txt", install)
            bootstrap = next(i for i, command in enumerate(commands) if "tools.bootstrap_cuda" in command)
            build = next(i for i, command in enumerate(commands)
                         if "tools.build_windows" in command and "--vision-gpu" in command)
            self.assertLess(bootstrap, build)

    def test_source_gpu_vision_reuses_installed_cuda_toolchain(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            for name in quickstart.REQUIREMENTS:
                (root / name).write_text(name)
            from tools.lamina_model import FILENAME
            (data / "models").mkdir()
            (data / "tokenizer").mkdir()
            (data / "models" / FILENAME).touch()
            (data / "tokenizer/tokenizer.json").touch()
            nvcc = data / "toolchains/cuda/bin/nvcc.exe"
            nvcc.parent.mkdir(parents=True)
            nvcc.touch()
            engine = root / "build-cuda/lamina-infer.exe"
            engine.parent.mkdir(parents=True)
            engine.touch()
            (engine.parent / "CMakeCache.txt").write_text("CMAKE_CUDA_ARCHITECTURES:STRING=89\n")
            encoder = root / "build-vision-gpu/bin/strata-vision.exe"
            commands = []
            selected = iter([None, encoder, encoder])
            with patch.object(quickstart, "ROOT", root), patch.object(quickstart, "DATA", data), \
                    patch.object(quickstart, "CONFIG", data / "quickstart.json"), \
                    patch.object(quickstart, "ENGINE", engine), \
                    patch.object(quickstart, "portable", return_value=False), \
                    patch.object(quickstart, "selected_vision", side_effect=lambda config: next(selected)), \
                    patch.object(quickstart, "gpu", return_value=("RTX 4060", "89", 8192)), \
                    patch.object(quickstart, "total_ram_gib", return_value=64), \
                    patch.object(quickstart, "run", side_effect=lambda args: commands.append(args)):
                quickstart.setup({"model": "qwen3.6", "vision": True, "vision_device": "gpu", "mtp": False})
            self.assertFalse(any("tools.bootstrap_cuda" in command for command in commands))
            self.assertTrue(any("tools.build_windows" in command and "--vision-gpu" in command for command in commands))

    def test_portable_setup_does_not_install_or_compile(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            (data / "models").mkdir(parents=True)
            (data / "tokenizer").mkdir()
            from tools.lamina_model import FILENAME
            (data / "models" / FILENAME).touch()
            (data / "tokenizer/tokenizer.json").touch()
            engine_fixture(root / "build-cuda")
            commands = []
            with patch.object(quickstart, "ROOT", root), patch.object(quickstart, "DATA", data), patch.object(quickstart, "CONFIG", data / "quickstart.json"), \
                    patch.object(quickstart, "ENGINE", root / "build-cuda/lamina-infer.exe"), \
                    patch.object(quickstart, "portable", return_value=True), \
                    patch.object(quickstart, "gpu", return_value=("RTX 4060", "89", 8192)), \
                    patch.object(quickstart, "total_ram_gib", return_value=64), \
                    patch.object(quickstart, "run", side_effect=lambda args: commands.append(args)):
                quickstart.setup({"mtp": False})
            self.assertEqual(len(commands), 1)
            self.assertIn("tools.lamina_assets", commands[0])
            self.assertIn("--text-only", commands[0])

    def test_portable_ornith_prepares_own_vision_without_compiler(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            engine_fixture(root / "engine")
            (root / "engine/strata-vision.exe").touch()
            with patch.object(quickstart, "ROOT", root), patch.object(quickstart, "DATA", data), \
                    patch.object(quickstart, "CONFIG", data / "quickstart.json"), \
                    patch.object(quickstart, "ENGINE", root / "engine/lamina-infer.exe"), \
                    patch.object(quickstart, "portable", return_value=True), \
                    patch.object(quickstart, "gpu", return_value=("RTX 4060", "89", 8192)), \
                    patch.object(quickstart, "total_ram_gib", return_value=64), \
                    patch.object(quickstart, "run", side_effect=AssertionError("compiler or package installer")), \
                    patch("tools.lamina_ornith.setup") as prepare_model, \
                    patch("tools.lamina_assets.download_assets") as prepare_vision:
                config = {"model": "ornith", "mtp": True}
                quickstart.setup(config)
                prepare_model.assert_called_once_with(data)
                prepare_vision.assert_called_once_with(data, model="ornith")
                options = quickstart.engine_options(config, 131072)
                self.assertEqual(Path(options[options.index("--vision-projector") + 1]),
                                 data / "vision/ornith-mmproj-BF16.gguf")
                self.assertIn("--vision", options)

    def test_portable_gpu_vision_selects_bundled_encoder_without_compiling(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); data = root / "data"; data.mkdir()
            engine_fixture(root / "engine")
            (root / "engine/strata-vision-gpu.exe").touch()
            with patch.object(quickstart, "ROOT", root), patch.object(quickstart, "DATA", data), \
                    patch.object(quickstart, "CONFIG", data / "quickstart.json"), \
                    patch.object(quickstart, "ENGINE", root / "engine/lamina-infer.exe"), \
                    patch.object(quickstart, "portable", return_value=True), \
                    patch.object(quickstart, "gpu", return_value=("RTX 4060", "89", 8192)), \
                    patch.object(quickstart, "total_ram_gib", return_value=64), \
                    patch.object(quickstart, "run", side_effect=AssertionError("compiler or package installer")), \
                    patch("tools.lamina_ornith.setup"), patch("tools.lamina_assets.download_assets"):
                config = {"model": "ornith", "mtp": False, "vision": True, "vision_device": "gpu"}
                quickstart.setup(config)
                options = quickstart.engine_options(config, 131072)
                self.assertIn("--vision-gpu", options)
                self.assertEqual(Path(options[options.index("--vision-engine") + 1]), root / "engine/strata-vision-gpu.exe")


if __name__ == "__main__":
    unittest.main()
