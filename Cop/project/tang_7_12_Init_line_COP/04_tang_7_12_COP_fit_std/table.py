"""旧版 table 模块兼容层。"""

try:
    from tangential.storage.csv import *
except ModuleNotFoundError:
    from src.tangential.storage.csv import *
