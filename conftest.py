"""pytest 根配置：让测试无需安装即可 import 本包。

为什么不用 `pip install -e .`：
贡献者 clone 下来第一件事应该是"能跑测试"，而不是"先建虚拟环境再装包"。
把仓库根加进 sys.path 是最短路径。注意这里**不修改**任何环境变量 ——
测试自己负责把 provider 设成 mock，避免"本地碰巧能跑、CI 上连了真模型"。
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
