"""运行配置：数据目录等可由环境变量覆盖。"""

from __future__ import annotations

import os


def resolve_data_dir() -> str:
    explicit = os.getenv("ASSESSMENT_DATA_DIR") or os.getenv("DATA_DIR")
    if explicit:
        return explicit
    # 默认放到工作目录下的 .data（Docker 中通过 DATA_DIR=/data 覆盖）
    return os.path.join(os.getcwd(), ".data")


DATA_DIR = resolve_data_dir()
