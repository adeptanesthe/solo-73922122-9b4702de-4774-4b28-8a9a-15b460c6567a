#!/usr/bin/env bash
# 一键自测：依赖 -> 60 项 pytest（含 40 组暴力枚举对照）-> 真实 uvicorn + curl 全流程。
# 用法：bash selftest.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo ">> 创建虚拟环境 .venv"
  if python3 -m venv .venv 2>/dev/null; then
    :
  else
    # 某些 Debian/Ubuntu 缺 python3.11-venv/ensurepip
    python3 -m venv --without-pip .venv
    GETPIP="$(mktemp -d)/get-pip.py"
    curl -sS -o "$GETPIP" https://bootstrap.pypa.io/get-pip.py
    "$PY" -I "$GETPIP"
  fi
fi

echo ">> 安装/检查依赖"
"$PY" -m pip install -q -r requirements.txt "httpx>=0.27" "pytest>=8"

echo ">> 运行 pytest"
"$PY" -m pytest tests/ -q

PORT="${SELFTEST_PORT:-8099}"
DATA_DIR="$(mktemp -d)/order-audit-data"
mkdir -p "$DATA_DIR"
LOG="selftest_live.log"

echo ">> 启动 uvicorn (127.0.0.1:$PORT, DATA_DIR=$DATA_DIR)"
DATA_DIR="$DATA_DIR" "$PY" -m uvicorn app.main:app \
  --host 127.0.0.1 --port "$PORT" > uvicorn_selftest.log 2>&1 &
SRV_PID=$!
cleanup() { kill "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 0.5
done

echo ">> 运行 curl API 全流程"
PYTHON_BIN="$PY" bash examples/api_examples.sh "http://127.0.0.1:$PORT" 2>&1 | tee "$LOG"

echo
echo ">> 自测完成：pytest 全部通过，curl 流程见 $LOG"
