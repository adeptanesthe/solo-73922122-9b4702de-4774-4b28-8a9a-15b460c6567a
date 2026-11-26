#!/usr/bin/env bash
# 关键 API 调用示例 + 全流程自测（真实 HTTP / curl）。
#
# 前置：服务已启动，例如
#   docker compose up --build
# 或本地
#   DATA_DIR=/tmp/order-audit-data .venv/bin/uvicorn app.main:app --port 8000
#
# 用法：
#   bash examples/api_examples.sh [BASE_URL]
set -u

BASE_URL="${1:-${BASE_URL:-http://127.0.0.1:8000}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="python3"
GEN_DIR="$HERE/generated"
mkdir -p "$GEN_DIR"

echo "==[0] 生成示例 WAV / 请求体 =="
"$PYTHON_BIN" "$HERE/make_fixtures.py"

# 从 JSON 文件提取字段：jfield resp.json "['assessment_id']"
jfield() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json, sys
path, expr = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)
print(eval("data" + expr, {"__builtins__": {}}, {"data": data}))  # noqa: S307
PY
}

hr() { echo; echo "------------------------------------------------------------"; }

hr; echo "==[1] 健康检查  GET /health =="
curl -sS "$BASE_URL/health" | "$PYTHON_BIN" -m json.tool

hr; echo "==[2] multipart 创建严格约束核定（应：建议排查该阶，漏脉冲 [1,4]）=="
PULSES="$(jfield "$GEN_DIR/fixture_meta.json" "['pulse_samples']")"
curl -sS -X POST "$BASE_URL/api/v1/assessments" \
  -F "audio=@$GEN_DIR/case_order3.wav;type=audio/wav" \
  -F "pulse_samples=$PULSES" \
  -F "total_revolutions=8" \
  -F "speed_diff_limit_rpm=30" \
  -F "order=3" \
  -F "threshold=0.5" \
  -o "$GEN_DIR/resp_strict.json" -w "HTTP %{http_code}\n"
"$PYTHON_BIN" - "$GEN_DIR/resp_strict.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print("status =", d["status"], "/", d["status_label_zh"])
print("score_lower =", d["score_lower"])
print("score_upper =", d["score_upper"])
print("missed @lower =", d["lower_endpoint"]["missed_pulse_intervals"])
print("missed @upper =", d["upper_endpoint"]["missed_pulse_intervals"])
PY
STRICT_ID="$(jfield "$GEN_DIR/resp_strict.json" "['assessment_id']")"
echo "assessment_id = $STRICT_ID"

hr; echo "==[3] JSON(base64) 创建宽松约束核定（阈值取中点，应：待核）=="
curl -sS -X POST "$BASE_URL/api/v1/assessments" \
  -H "Content-Type: application/json" \
  --data @"$GEN_DIR/request_loose.json" \
  -o "$GEN_DIR/resp_loose.json" -w "HTTP %{http_code}\n"
LOOSE_ID="$(jfield "$GEN_DIR/resp_loose.json" "['assessment_id']")"
echo "assessment_id = $LOOSE_ID"
"$PYTHON_BIN" - "$GEN_DIR/resp_loose.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print("status =", d["status"], "/", d["status_label_zh"])
print(f"bounds = [{d['score_lower']:.6f}, {d['score_upper']:.6f}],",
      "threshold =", d["threshold"])
print("lower end: missed =", d["lower_endpoint"]["missed_pulse_intervals"],
      "assignment =", d["lower_endpoint"]["assignment_revolutions"])
print("upper end: missed =", d["upper_endpoint"]["missed_pulse_intervals"],
      "assignment =", d["upper_endpoint"]["assignment_revolutions"])
PY

hr; echo "==[4] GET 查询报告 =="
curl -sS "$BASE_URL/api/v1/assessments/$LOOSE_ID" \
  -o "$GEN_DIR/resp_get.json" -w "HTTP %{http_code}\n"
echo "已保存完整报告: $GEN_DIR/resp_get.json"

hr; echo "==[5] PUT 把全部间隔锁定为上界转法（上下界应坍缩到上界，revision=2）=="
LOCKS_JSON="$("$PYTHON_BIN" - "$GEN_DIR/resp_loose.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
locks = [{"interval": i, "revolutions": r}
         for i, r in enumerate(d["upper_endpoint"]["assignment_revolutions"])]
print(json.dumps({"locks": locks}, ensure_ascii=False))
PY
)"
echo "提交: $LOCKS_JSON"
curl -sS -X PUT "$BASE_URL/api/v1/assessments/$LOOSE_ID/locks" \
  -H "Content-Type: application/json" \
  -d "$LOCKS_JSON" \
  -o "$GEN_DIR/resp_locked.json" -w "HTTP %{http_code}\n"
"$PYTHON_BIN" - "$GEN_DIR/resp_locked.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print("revision =", d["revision"], "| status =", d["status_label_zh"])
print(f"bounds = [{d['score_lower']:.6f}, {d['score_upper']:.6f}]")
print("locks =", d["lock_map"])
PY

hr; echo "==[6] 提交无解更正：6 个间隔全锁 2 转但总转数=8（应 HTTP 409 拒绝）=="
curl -sS -X PUT "$BASE_URL/api/v1/assessments/$LOOSE_ID/locks" \
  -H "Content-Type: application/json" \
  -d '{"locks":[{"interval":0,"revolutions":2},{"interval":1,"revolutions":2},{"interval":2,"revolutions":2},{"interval":3,"revolutions":2},{"interval":4,"revolutions":2},{"interval":5,"revolutions":2}]}' \
  -o "$GEN_DIR/resp_rejected.json" -w "HTTP %{http_code}\n"
"$PYTHON_BIN" - "$GEN_DIR/resp_rejected.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
detail = d.get("detail", {})
print("code =", d.get("code"))
print("message =", d.get("message"))
print("current_revision =", detail.get("current_revision"),
      "| rejected =", detail.get("rejected"))
PY

hr; echo "==[7] 原报告未被改动（revision 仍为 2，锁定仍为第 5 步内容）=="
curl -sS "$BASE_URL/api/v1/assessments/$LOOSE_ID" \
  | "$PYTHON_BIN" -c 'import json,sys; d=json.load(sys.stdin); print("revision =", d["revision"], "| lock_map =", d["lock_map"])'

hr; echo "==[8] DELETE 解除全部锁定（revision=3，恢复宽上下界）=="
curl -sS -X DELETE "$BASE_URL/api/v1/assessments/$LOOSE_ID/locks" \
  -o "$GEN_DIR/resp_released.json" -w "HTTP %{http_code}\n"
"$PYTHON_BIN" - "$GEN_DIR/resp_released.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print("revision =", d["revision"], "| locks =", d["locks"])
print(f"bounds = [{d['score_lower']:.6f}, {d['score_upper']:.6f}]")
PY

hr; echo "==[9] 修订历史 / 拒绝审计 / 列表 =="
curl -sS "$BASE_URL/api/v1/assessments/$LOOSE_ID/revisions" | "$PYTHON_BIN" -m json.tool
curl -sS "$BASE_URL/api/v1/assessments/$LOOSE_ID/audit" \
  | "$PYTHON_BIN" -c 'import json,sys; d=json.load(sys.stdin); print("rejected_attempts =", len(d["rejected_attempts"]))'
curl -sS "$BASE_URL/api/v1/assessments" \
  | "$PYTHON_BIN" -c 'import json,sys; d=json.load(sys.stdin); print("assessment count =", d["count"])'

hr; echo "全部示例请求执行完毕。Swagger 文档: $BASE_URL/docs"
