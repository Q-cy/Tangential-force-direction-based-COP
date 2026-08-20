"""Tangential SDK 的 package 资源容器。

本包不提供运行逻辑；``fit_coefs.bin`` 是随 wheel 分发的静态标定模型，
由 ``FitCalibrationModel.from_default()`` 通过 ``importlib.resources`` 定位
和读取。调用方不应把资源路径假定为源码目录或外部 ``share/`` 目录，也不
应在运行期改写该文件。
"""
