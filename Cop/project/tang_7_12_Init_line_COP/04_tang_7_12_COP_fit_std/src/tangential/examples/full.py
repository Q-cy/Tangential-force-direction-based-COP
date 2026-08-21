"""完整采集应用示例。"""

from __future__ import annotations

from ..application import run_application
from ..config import FullApplicationConfig


def main(config: FullApplicationConfig | None = None) -> int:
    """使用公开配置和公开入口启动完整采集应用。"""
    return run_application(config or FullApplicationConfig())


if __name__ == "__main__":
    raise SystemExit(main())
