"""旧版完整应用模块兼容层。"""

try:
    from tangential.full import *
except ModuleNotFoundError:
    from src.tangential.full import *
