"""打包契约测试。

这些测试只验证分发侧约定，不导入项目根目录下的旧脚本。源码包由主代理
创建后，``src/tangential`` 相关测试会自动启用；在此之前仅验证配置本身。
构建测试使用本地 setuptools/wheel 和 ``pip --no-index``，不访问网络。
"""

from __future__ import annotations

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


def _subprocess_env(*, python_path: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    if python_path is not None:
        env["PYTHONPATH"] = str(python_path)
    return env


class DistributionConfigurationTests(unittest.TestCase):
    def test_pyproject_declares_public_distribution_contract(self):
        with PYPROJECT.open("rb") as stream:
            config = tomllib.load(stream)

        project = config["project"]
        self.assertEqual(project["name"], "tangential-sensor")
        self.assertEqual(project["version"], "0.2.0")
        self.assertEqual(project["requires-python"], ">=3.11")
        self.assertEqual(
            set(project["dependencies"]),
            {"numpy", "scipy", "pyserial"},
        )
        self.assertEqual(
            set(project["optional-dependencies"]["full"]),
            {"pyqtgraph", "matplotlib", "PyQt5"},
        )

        setuptools = config["tool"]["setuptools"]
        self.assertEqual(setuptools["package-dir"], {"": "src"})
        self.assertEqual(
            config["tool"]["setuptools"]["packages"]["find"]["where"],
            ["src"],
        )
        self.assertEqual(
            config["tool"]["setuptools"]["package-data"],
            {"tangential.resources": ["fit_coefs.bin"]},
        )
        self.assertNotIn("data-files", config["tool"]["setuptools"])

    def test_manifest_keeps_model_in_source_distribution(self):
        manifest = PROJECT_ROOT / "MANIFEST.in"
        self.assertIn(
            "recursive-include src/tangential/resources *.bin",
            manifest.read_text(),
        )


@unittest.skipUnless(
    (PACKAGE_ROOT / "__init__.py").is_file(),
    "主代理尚未创建 src/tangential，暂不运行公开 API 分发测试",
)
class PublicImportTests(unittest.TestCase):
    def test_public_import_isolated_from_project_root(self):
        command = (
            "import sys; "
            "from tangential import FitCalibrationModel, TangentialSample, TangentialSensor; "
            "assert TangentialSample is not None; "
            "assert TangentialSensor is not None; "
            "model = FitCalibrationModel.from_default(); "
            "assert model.available; "
            "assert abs(model.predict(0.1, 0.1, 100000)[0] - 1.4477653909084447) < 1e-12; "
            "assert 'pyqtgraph' not in sys.modules"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", command],
                cwd=directory,
                env=_subprocess_env(python_path=SOURCE_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(
            result.returncode,
            0,
            msg=result.stderr or result.stdout,
        )


@unittest.skipUnless(
    (PACKAGE_ROOT / "__init__.py").is_file(),
    "主代理尚未创建 src/tangential，暂不构建 wheel",
)
class WheelDistributionTests(unittest.TestCase):
    def test_wheel_contains_package_and_model_data_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory) / "source"
            source_dir.mkdir()
            for filename in ("pyproject.toml", "MANIFEST.in", "readme.md"):
                shutil.copy2(PROJECT_ROOT / filename, source_dir / filename)
            shutil.copytree(SOURCE_ROOT, source_dir / "src")
            wheel_dir = Path(directory) / "wheel"
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
            self.assertEqual(build.returncode, 0, msg=build.stderr)

            wheels = sorted(wheel_dir.glob("tangential_sensor-*.whl"))
            self.assertEqual(len(wheels), 1, msg=build.stdout)
            wheel_path = wheels[0]
            with zipfile.ZipFile(wheel_path) as archive:
                names = set(archive.namelist())
                self.assertTrue(any(name.startswith("tangential/") for name in names))
                self.assertIn("tangential/resources/fit_coefs.bin", names)
                self.assertNotIn(
                    "tangential_sensor-0.2.0.data/data/share/tangential/fit_coefs.bin",
                    names,
                )

            install_dir = Path(directory) / "install"
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
                cwd=directory,
                env=_subprocess_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(install.returncode, 0, msg=install.stderr)

            command = (
                "import sys; "
                "from tangential import FitCalibrationModel, TangentialSample, TangentialSensor; "
                "assert TangentialSample is not None; "
                "assert TangentialSensor is not None; "
                "model = FitCalibrationModel.from_default(); "
                "assert model.available; "
                "assert abs(model.predict(0.1, 0.1, 100000)[0] - 1.4477653909084447) < 1e-12; "
                "assert 'pyqtgraph' not in sys.modules"
            )
            imported = subprocess.run(
                [sys.executable, "-c", command],
                cwd=directory,
                env=_subprocess_env(python_path=install_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                imported.returncode,
                0,
                msg=imported.stderr or imported.stdout,
            )


if __name__ == "__main__":
    unittest.main()
