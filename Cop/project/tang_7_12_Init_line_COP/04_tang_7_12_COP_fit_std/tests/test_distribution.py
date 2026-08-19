"""验证正式 Tangential SDK 的打包、资源和公共导入契约。

构建测试使用本地 setuptools/wheel 和 ``pip --no-index``，不访问网络。
"""

from __future__ import annotations

import os
import ast
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
    def test_legacy_root_files_and_imports_are_absent(self):
        legacy_files = {
            "data.py", "table.py", "realtime.py", "tangential_package.py",
            "tangential_other_package.py", "main.py", "example.py",
            "fit.py", "plot_static.py",
        }
        for filename in legacy_files:
            self.assertFalse(
                (PROJECT_ROOT / filename).exists(),
                f"旧根文件仍存在: {filename}",
            )

        forbidden_modules = {
            "data", "table", "realtime", "main", "example", "fit",
            "plot_static", "tangential_package", "tangential_other_package",
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
                        any(
                            module == legacy or module.startswith(legacy + ".")
                            for legacy in forbidden_modules
                        ),
                        f"{path} 使用了旧模块导入: {module}",
                    )

    def test_pyproject_declares_public_distribution_contract(self):
        with PYPROJECT.open("rb") as stream:
            config = tomllib.load(stream)

        project = config["project"]
        self.assertEqual(project["name"], "tangential-sensor")
        self.assertEqual(project["version"], "0.2.0")
        self.assertEqual(project["requires-python"], ">=3.11")
        self.assertEqual(project["scripts"], {"tangential": "tangential.cli:main"})
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
    "正式 tangential package 不存在，暂不运行公开 API 分发测试",
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
    "正式 tangential package 不存在，暂不运行 wheel 测试",
)
@unittest.skip("wheel 构建验收在阶段3执行")
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
