"""Build the private runtime modules as CPython extension modules.

The repository keeps the corresponding Python sources as the canonical,
readable implementation.  ``build_py`` omits only those implementation files
from wheels because the compiled extensions provide the same import names.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_py import build_py as _build_py


# One extension per implementation module keeps imports and tracebacks aligned
# with the source layout while allowing the public/configuration layer to stay
# readable in an installed wheel.
COMPILED_MODULES = {
    "tangential.runtime.sensor": "src/tangential/runtime/sensor.py",
    "tangential.runtime.session": "src/tangential/runtime/session.py",
    "tangential.runtime.synchronization": "src/tangential/runtime/synchronization.py",
    "tangential.acquisition.buffer": "src/tangential/acquisition/buffer.py",
    "tangential.sensors.pressure": "src/tangential/sensors/pressure.py",
    "tangential.sensors.force": "src/tangential/sensors/force.py",
    "tangential.processing.cop": "src/tangential/processing/cop.py",
    "tangential.processing.calibration": "src/tangential/processing/calibration.py",
    "tangential.processing.calconsistence": "src/tangential/processing/calconsistence.py",
    "tangential.processing.slip": "src/tangential/processing/slip.py",
    "tangential.storage.csv": "src/tangential/storage/csv.py",
}


class BinaryWheelBuildPy(_build_py):
    """Copy public Python sources but omit compiled implementation modules."""

    def run(self):
        """清理旧package输出后复制当前公开Python文件。

        删除或重命名模块后，setuptools默认会保留 ``build/lib*`` 中的旧文件，
        并可能把它们再次装入wheel。标准wheel构建中 ``build_py`` 先于
        ``build_ext``，因此这里只删除当前package输出，再由后续步骤写入
        最新Python文件、资源和扩展模块。
        """
        package_output = Path(self.build_lib) / "tangential"
        if package_output.exists():
            shutil.rmtree(package_output)
        super().run()

    def find_package_modules(self, package: str, package_dir: str):
        """Return package modules excluding names supplied by extensions."""
        modules = super().find_package_modules(package, package_dir)
        return [
            module
            for module in modules
            if f"{module[0]}.{module[1]}" not in COMPILED_MODULES
        ]


extensions = [
    Extension(
        module_name,
        [source_path],
        define_macros=[("CYTHON_TRACE", "0")],
        extra_compile_args=["-O3", "-g0"],
        extra_link_args=["-s"],
    )
    for module_name, source_path in COMPILED_MODULES.items()
]


setup(
    ext_modules=cythonize(
        extensions,
        build_dir="build/cython",
        force=True,
        compiler_directives={
            "language_level": 3,
            "annotation_typing": False,
            "binding": True,
            "embedsignature": True,
            "always_allow_keywords": True,
        },
    ),
    cmdclass={"build_py": BinaryWheelBuildPy},
)
