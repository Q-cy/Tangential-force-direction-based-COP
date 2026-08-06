"""测试入口: python tests/run_tests.py — 纯 assert, 无框架"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import test_cop_state
import test_fit_roundtrip
import test_frame
import test_sync

MODULES = [test_frame, test_sync, test_cop_state, test_fit_roundtrip]


def main() -> int:
    total = failed = 0
    for mod in MODULES:
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            total += 1
            try:
                getattr(mod, name)()
                print(f"  PASS  {mod.__name__}.{name}")
            except Exception:
                failed += 1
                print(f"  FAIL  {mod.__name__}.{name}")
                traceback.print_exc()
    print(f"\n{'=' * 50}\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
