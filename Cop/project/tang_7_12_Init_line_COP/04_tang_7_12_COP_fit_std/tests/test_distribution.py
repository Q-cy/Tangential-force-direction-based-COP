"""验证 Tangential SDK 的正式 wheel、静态资源和公共 CLI 契约。"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
SOURCE_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "tangential"
COMPILED_MODULES = {
    "tangential/acquisition/buffer",
    "tangential/processing/calibration",
    "tangential/processing/cop",
    "tangential/processing/slip",
    "tangential/runtime/sensor",
    "tangential/runtime/session",
    "tangential/runtime/synchronization",
    "tangential/sensors/force",
    "tangential/sensors/pressure",
    "tangential/storage/csv",
}
LEGACY_ROOT_FILES = {
    "data.py",
    "table.py",
    "realtime.py",
    "tangential_package.py",
    "tangential_other_package.py",
    "main.py",
    "example.py",
    "fit.py",
    "plot_static.py",
}


def _subprocess_env(*, python_path: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    if python_path is not None:
        env["PYTHONPATH"] = str(python_path)
    return env


def _expected_package_modules() -> set[str]:
    return {
        "tangential/" + path.relative_to(PACKAGE_ROOT).as_posix()
        for path in PACKAGE_ROOT.rglob("*.py")
        if ("tangential/" + path.relative_to(PACKAGE_ROOT).with_suffix("").as_posix())
        not in COMPILED_MODULES
    }


def _isolated_import_command() -> str:
    return (
        "import sys; import tangential; import tangential.api; import tangential.runtime; "
        "from dataclasses import fields; "
        "from tangential import FitCalibrationModel, TangentialFrame, TangentialSensorAPI; "
        "assert [item.name for item in fields(TangentialFrame)] == ['raw', 'adc_sum', 'cop_x', 'cop_y', 'angle', 'dx', 'dy', 'motion_state']; "
        "assert not hasattr(TangentialFrame, 'total'); "
        "assert not hasattr(tangential, 'TangentialSample'); "
        "assert not hasattr(tangential.api, 'TangentialSample'); "
        "assert not hasattr(tangential.runtime, 'TangentialSample'); "
        "legacy_projection_name = 'to_' + 'tangential_' + 'frame'; "
        "assert not hasattr(tangential, legacy_projection_name); "
        "assert not hasattr(tangential.api, legacy_projection_name); "
        "assert not hasattr(tangential.runtime, legacy_projection_name); "
        "assert TangentialSensorAPI is not None; "
        "assert not hasattr(tangential, 'TangentialSensor'); "
        "assert not hasattr(tangential.api, 'TangentialSensor'); "
        "assert not hasattr(tangential.runtime, 'TangentialSensor'); "
        "model = FitCalibrationModel.from_default(); "
        "assert model.available; "
        "assert abs(model.predict(0.1, 0.1, 100000)[0] - 1.4477653909084447) < 1e-12; "
        "assert 'pyqtgraph' not in sys.modules; "
        "assert 'matplotlib' not in sys.modules"
    )


class DistributionConfigurationTests(unittest.TestCase):
    def test_legacy_root_files_and_imports_are_absent(self):
        for filename in LEGACY_ROOT_FILES:
            self.assertFalse((PROJECT_ROOT / filename).exists(), f"旧根文件仍存在: {filename}")

        forbidden_modules = {
            "data",
            "table",
            "realtime",
            "main",
            "example",
            "fit",
            "plot_static",
            "tangential_package",
            "tangential_other_package",
        }
        for path in PROJECT_ROOT.rglob("*.py"):
            relative = path.relative_to(PROJECT_ROOT)
            if any(part in {"build", "dist", "__pycache__"} for part in relative.parts):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported = [node.module or ""]
                else:
                    continue
                for module in imported:
                    self.assertFalse(
                        module == "src.tangential" or module.startswith("src.tangential."),
                        f"{path} 使用了 src.tangential 导入",
                    )
                    self.assertFalse(
                        any(module == legacy or module.startswith(legacy + ".") for legacy in forbidden_modules),
                        f"{path} 使用了旧模块导入: {module}",
                    )

    def test_pyproject_declares_public_distribution_contract(self):
        with PYPROJECT.open("rb") as stream:
            config = tomllib.load(stream)

        project = config["project"]
        self.assertEqual(project["name"], "tangential-sensor")
        self.assertEqual(project["version"], "0.5.0")
        self.assertEqual(project["requires-python"], ">=3.11")
        self.assertEqual(project["scripts"], {"tangential": "tangential.cli:main"})
        self.assertEqual(set(project["dependencies"]), {"numpy", "scipy", "pyserial"})
        self.assertEqual(
            set(project["optional-dependencies"]["full"]),
            {"pyqtgraph", "matplotlib", "PyQt5"},
        )

        setuptools = config["tool"]["setuptools"]
        self.assertEqual(setuptools["package-dir"], {"": "src"})
        self.assertEqual(setuptools["packages"]["find"]["where"], ["src"])
        self.assertEqual(
            setuptools["package-data"],
            {
                "tangential": ["py.typed", "**/*.pyi"],
                "tangential.resources": ["fit_coefs.bin"],
            },
        )
        self.assertNotIn("data-files", setuptools)
        self.assertIn("Cython>=3.1,<4", config["build-system"]["requires"])

    def test_version_is_consistent_across_package_and_cli(self):
        with PYPROJECT.open("rb") as stream:
            project_version = tomllib.load(stream)["project"]["version"]
        from tangential import __version__
        from tangential.cli import VERSION

        self.assertEqual(project_version, "0.5.0")
        self.assertEqual(__version__, project_version)
        self.assertEqual(VERSION, project_version)

    def test_manifest_declares_exact_model_resource(self):
        lines = {
            line.strip()
            for line in (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("recursive-include src/tangential/resources *.bin", lines)
        self.assertIn("recursive-include src/tangential *.pyi", lines)
        self.assertIn("include src/tangential/py.typed", lines)


class PublicImportTests(unittest.TestCase):
    def test_public_import_isolated_from_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", _isolated_import_command()],
                cwd=directory,
                env=_subprocess_env(python_path=SOURCE_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)


class WheelDistributionTests(unittest.TestCase):
    def test_wheel_build_and_isolated_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source"
            source_dir.mkdir()
            for filename in ("pyproject.toml", "setup.py", "MANIFEST.in", "readme.md"):
                shutil.copy2(PROJECT_ROOT / filename, source_dir / filename)
            shutil.copytree(SOURCE_ROOT, source_dir / "src")

            wheel_dir = root / "wheel"
            wheel_dir.mkdir()
            build = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    ".",
                    "--no-deps",
                    "--no-build-isolation",
                    "--no-index",
                    "--wheel-dir",
                    str(wheel_dir),
                ],
                cwd=source_dir,
                env=_subprocess_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(build.returncode, 0, msg=build.stderr or build.stdout)

            wheels = sorted(wheel_dir.glob("tangential_sensor-0.5.0-*.whl"))
            self.assertEqual(len(wheels), 1, msg=build.stdout)
            wheel_path = wheels[0]
            with zipfile.ZipFile(wheel_path) as archive:
                names = set(archive.namelist())
                expected_modules = _expected_package_modules()
                self.assertTrue(expected_modules <= names)
                for module in COMPILED_MODULES:
                    self.assertFalse(f"{module}.py" in names)
                    self.assertTrue(
                        any(name.startswith(module + ".cpython-311-") and name.endswith(".so")
                            for name in names),
                        module,
                    )
                    self.assertIn(f"{module}.pyi", names)
                self.assertIn("tangential/py.typed", names)
                self.assertFalse(any(name.endswith((".pyx", ".c", ".cpp")) for name in names))
                self.assertIn("tangential/resources/fit_coefs.bin", names)
                sensor_stub = archive.read(
                    "tangential/runtime/sensor.pyi"
                ).decode("utf-8")
                self.assertNotIn("TangentialSample", sensor_stub)
                self.assertNotIn("TangentialSampleProcessor", sensor_stub)
                self.assertNotIn("sample_processor", sensor_stub)
                self.assertNotIn("_process_sample", sensor_stub)
                self.assertTrue(
                    any(name.endswith(".dist-info/entry_points.txt") for name in names)
                )
                entry_point_files = [
                    name for name in names if name.endswith(".dist-info/entry_points.txt")
                ]
                self.assertEqual(len(entry_point_files), 1)
                entry_points = archive.read(entry_point_files[0]).decode("utf-8")
                self.assertIn("[console_scripts]", entry_points)
                self.assertIn("tangential = tangential.cli:main", entry_points)
                self.assertFalse(any(name.startswith("share/") for name in names))
                self.assertFalse(any("/share/" in name for name in names))
                self.assertFalse(any(name in LEGACY_ROOT_FILES for name in names))
                self.assertNotIn("tangential/examples/dual_pressure.py", names)
                self.assertIn("tangential/examples/dual_sensor.py", names)
                self.assertNotIn(
                    "tangential_sensor-0.5.0.data/data/share/tangential/fit_coefs.bin",
                    names,
                )

            install_dir = root / "install"
            install_dir.mkdir()
            install = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-index",
                    "--target",
                    str(install_dir),
                    str(wheel_path),
                ],
                cwd=root,
                env=_subprocess_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(install.returncode, 0, msg=install.stderr or install.stdout)

            imported = subprocess.run(
                [sys.executable, "-c", _isolated_import_command()],
                cwd=root,
                env=_subprocess_env(python_path=install_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(imported.returncode, 0, msg=imported.stderr or imported.stdout)

            version = subprocess.run(
                [sys.executable, "-m", "tangential.cli", "--version"],
                cwd=root,
                env=_subprocess_env(python_path=install_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(version.returncode, 0, msg=version.stderr)
            self.assertEqual(version.stdout.strip(), "0.5.0")

            for command in ("example", "app", "dual", "plot", "fit"):
                help_result = subprocess.run(
                    [sys.executable, "-m", "tangential.cli", command, "--help"],
                    cwd=root,
                    env=_subprocess_env(python_path=install_dir),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    help_result.returncode,
                    0,
                    msg=f"{command}: {help_result.stderr or help_result.stdout}",
                )
                self.assertIn(command, help_result.stdout)


if __name__ == "__main__":
    unittest.main()
