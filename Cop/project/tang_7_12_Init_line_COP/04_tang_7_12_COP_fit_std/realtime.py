"""旧版 realtime 模块兼容层。"""

try:
    from tangential.gui.realtime import *
except ModuleNotFoundError:
    from src.tangential.gui.realtime import *
