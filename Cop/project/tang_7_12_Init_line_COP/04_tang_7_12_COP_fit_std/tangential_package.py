"""旧版最小 API 模块兼容层；新代码请导入 tangential。"""

try:
    from tangential import *
except ModuleNotFoundError:
    from src.tangential import *
