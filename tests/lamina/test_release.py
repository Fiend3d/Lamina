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
from tools.package_windows import validate_build


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
            with patch.object(quickstart, "DATA", data), patch.object(quickstart, "CONFIG", data / "quickstart.json"), \
                    patch.object(quickstart, "ENGINE", root / "build-cuda/lamina-infer.exe"), \
                    patch.object(quickstart, "portable", return_value=True), \
                    patch.object(quickstart, "gpu", return_value=("RTX 4060", "89", 8192)), \
                    patch.object(quickstart, "total_ram_gib", return_value=64), \
                    patch.object(quickstart, "run", side_effect=lambda args: commands.append(args)):
                quickstart.setup({"mtp": False})
            self.assertEqual(len(commands), 1)
            self.assertIn("tools.lamina_assets", commands[0])
            self.assertIn("--text-only", commands[0])


if __name__ == "__main__":
    unittest.main()
