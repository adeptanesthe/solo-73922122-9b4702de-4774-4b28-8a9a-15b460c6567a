# 试车检修异响阶次核定 API（FastAPI + NumPy）

给试车检修员用的核定服务：判断一段异响是否值得按指定阶次排查。

输入 ≤10 秒的 **16 kHz / 单声道 / PCM16 WAV**、探头脉冲采样点、控制器总转数、
相邻间隔平均转速差上限、待查阶次（1～8）和判定阈值。探头可能漏脉冲，因此每个
脉冲间隔可能是 **1 转或 2 转**。服务用 NumPy 按 **每转 64 点**均匀重采样，FFT
求各段去均值后的目标阶次能量占比（零能量记 0），按转数加权，并在
**总转数 + 转速差上限 + 人工锁定**三重约束下，用动态规划求得分的上下界，
同时还原两端分别是哪些间隔漏了脉冲（**不逐个枚举转法**）。

---

## 1. 判定规则

设下界 `L`、上界 `U`、阈值 `T`：

| 条件 | `status` | 中文结论 | 含义 |
| --- | --- | --- | --- |
| `L ≥ T`（两端都达标） | `recommend_investigate` | **建议排查该阶** | 所有满足约束的转法都达标 |
| `L < T ≤ U`（仅上界达标） | `pending_verification` | **待核** | 只有部分转法达标，需核对两端漏脉冲间隔 |
| `U < T`（两端均未达标） | `below_standard` | **未达标** | 没有任何可行转法达标 |

---

## 2. 目录结构

```
app/
  main.py       # FastAPI 路由：创建/查询/锁定更正，JSON + multipart
  payload.py    # 字段归一化、别名、锁定格式解析、OpenAPI 模型
  audio.py      # PCM16 WAV 校验读取、每转64点重采样、FFT 能量占比
  solver.py     # O(m²) 动态规划上下界 + 路径还原 + 无解诊断
  service.py    # 业务编排、三态判定、报告构建、锁定更正事务
  store.py      # 线程安全 JSON 文件持久化 + revision + 拒绝审计
examples/
  make_fixtures.py   # 合成已知阶次成分的示例 WAV/请求体
  api_examples.sh    # 可运行的 curl 全流程示例
tests/               # 60 项 pytest（含 40 组对拍暴力枚举）
Dockerfile
docker-compose.yml
selftest.sh          # 一键自测
```

---

## 3. Docker Compose 启动

```bash
docker compose up --build
# 可选：API_PORT=8020 docker compose up --build
```

服务监听 `http://localhost:8000`：

- Swagger UI：<http://localhost:8000/docs>
- 健康检查：`curl http://localhost:8000/health`
- 报告持久化在命名卷 `order-audit-data` 的 `/data/assessments.json`。

停止：`docker compose down`（加 `-v` 可同时删除报告数据卷）。

### 本地直接运行（不用 Docker）

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
DATA_DIR=.data .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

一键自测（自动装测试依赖、跑 pytest、起真实服务并跑 curl 全流程）：

```bash
bash selftest.sh
```

---

## 4. API 一览

路由同时挂在根路径、`/api`、`/api/v1` 三种前缀下（下表用 `/api/v1`）。

| 方法 & 路径 | 说明 |
| --- | --- |
| `POST /api/v1/assessments` | 按 `Content-Type` 自适应：JSON（base64 WAV）或 multipart（WAV 文件） |
| `POST /api/v1/assessments/json` | 带 OpenAPI schema 的 JSON 入口 |
| `POST /api/v1/assessments/upload` | 带 OpenAPI schema 的 multipart 入口 |
| `GET  /api/v1/assessments` | 报告列表摘要 |
| `GET  /api/v1/assessments/{id}` | 查询当前报告（被拒绝的更正不会改变它） |
| `PUT/POST /api/v1/assessments/{id}/locks` | 全量更换人工锁定并重算报告；无解返回 `409`，原报告不变 |
| `PATCH /api/v1/assessments/{id}` | 同上（PATCH 别名） |
| `DELETE /api/v1/assessments/{id}/locks` | 解除全部锁定并重算 |
| `POST /api/v1/assessments/{id}/locks/release` | 解除锁定别名 |
| `GET  /api/v1/assessments/{id}/revisions[/{rev}]` | 接受过的锁定版本快照 |
| `GET  /api/v1/assessments/{id}/audit` | 被拒绝的更正尝试与原因 |

### 请求字段

| 字段 | 类型 | 约束/说明 |
| --- | --- | --- |
| `audio`（multipart 文件）/ `audio_wav_base64`（JSON） | WAV | ≤10 s、16 kHz、单声道、PCM16；base64 可带 `data:audio/wav;base64,` 前缀 |
| `pulse_samples` | int[] | 严格递增的脉冲采样点，≥2 个点、≤100 个间隔，点位置在 WAV 范围内 |
| `total_revolutions` | int | 控制器总转数 R；必须等于 Σrᵢ |
| `speed_diff_limit_rpm` | float ≥0 | 相邻间隔平均转速差上限 Δ（rpm）。也接受 `speed_diff_unit:"rps"` |
| `order` | int | 待查阶次 1～8 |
| `threshold` | float 0～1 | 判定阈值（能量占比）；百分数可用 `threshold_percent` |
| `locks` | 见下 | 可选，人工锁定 |

常见别名字段也能识别：`pulses`、`total_revs`、`speed_diff_limit`、
`target_order`、中文键 `脉冲采样点 / 总转数 / 转速差上限 / 阶次 / 阈值 / 人工锁定`。

### 锁定格式（可任意混用）

```json
{ "locks": { "1": 2, "4": 2 } }
```

```json
{ "locks": [ {"interval": 1, "revolutions": 2},
             {"interval": 4, "missed_pulse": true} ] }
```

```json
{ "locks": [1, 4] }
```

以上三种等价：把间隔 1、4 锁定为 2 转（漏脉冲）。`{"interval":i,"revolutions":1}`
或 `{"interval":i,"missed_pulse":false}` 锁定为 1 转。PUT 的 `locks` 是**全量替换**；
`{"locks": []}`、`{"locks": {}}` 或 `{"release_all": true}` 都表示全部解除。

---

## 5. curl 调用例子

先造示例 WAV 和请求体（脚本会合成 2 秒音频）：

```bash
.venv/bin/python examples/make_fixtures.py
# 生成 examples/generated/case_order3.wav、request_strict.json、request_loose.json
```

### 5.1 multipart 上传 + 严格转速差约束（实测：建议排查，漏脉冲间隔 [1,4]）

```bash
curl -X POST http://localhost:8000/api/v1/assessments \
  -F "audio=@examples/generated/case_order3.wav;type=audio/wav" \
  -F "pulse_samples=[0,4000,12000,16000,20000,28000,32000]" \
  -F "total_revolutions=8" \
  -F "speed_diff_limit_rpm=30" \
  -F "order=3" \
  -F "threshold=0.5"
```

实测响应要点：

```json
{
  "status": "recommend_investigate",
  "status_label_zh": "建议排查该阶",
  "score_lower": 0.5528841309794317,
  "score_upper": 0.5528841309794317,
  "lower_endpoint": {
    "assignment_revolutions": [1, 2, 1, 1, 2, 1],
    "missed_pulse_intervals": [1, 4],
    "max_adjacent_speed_diff_rpm": 0.0
  }
}
```

### 5.2 JSON + base64，宽松约束（实测：待核，两端漏脉冲集合不同）

```bash
curl -X POST http://localhost:8000/api/v1/assessments \
  -H "Content-Type: application/json" \
  --data @examples/generated/request_loose.json
```

实测：阈值 `0.2965`，上下界 `[0.040202, 0.552884]`：

- 下界端：指派 `[1,1,1,2,1,2]`，漏脉冲间隔 `[3,5]`；
- 上界端：指派 `[1,2,1,1,2,1]`，漏脉冲间隔 `[1,4]`；
- 结论 `pending_verification / 待核`。

### 5.3 查询报告

```bash
curl http://localhost:8000/api/v1/assessments/asn_xxxxxxxx
```

完整报告还包含每个间隔在两种假设下的转速与能量占比：

```json
{
  "index": 1, "length_samples": 8000,
  "average_speed_rpm": {"1": 120.0, "2": 240.0},
  "target_energy_ratio": {"1": 2.24e-05, "2": 0.9427}
}
```

### 5.4 更改锁定 → 更新报告

```bash
curl -X PUT http://localhost:8000/api/v1/assessments/asn_xxxxxxxx/locks \
  -H "Content-Type: application/json" \
  -d '{"locks":[{"interval":1,"revolutions":2},{"interval":4,"revolutions":2}]}'
```

把全部间隔锁成上界指派后，实测上下界坍缩为同一个值 `0.552884`，
`revision` 从 1 变 2，结论变为“建议排查该阶”。

### 5.5 无解更正被拒绝（409），原锁定/原报告仍可查询

6 个间隔全锁 2 转 ⇒ 至少 12 转，但总转数是 8：

```bash
curl -i -X PUT http://localhost:8000/api/v1/assessments/asn_xxxxxxxx/locks \
  -H "Content-Type: application/json" \
  -d '{"locks":[{"interval":0,"revolutions":2},{"interval":1,"revolutions":2},
                {"interval":2,"revolutions":2},{"interval":3,"revolutions":2},
                {"interval":4,"revolutions":2},{"interval":5,"revolutions":2}]}'
# HTTP/1.1 409 Conflict
# code = incompatible_total_revolutions_or_locks
# detail.current_revision 仍为 2，detail.current_report 是原报告
```

随后 `GET /assessments/{id}` 看到的 `revision / lock_map / 上下界` 均未改变；
`GET /assessments/{id}/audit` 留有本次拒绝记录。

### 5.6 解除锁定

```bash
curl -X DELETE http://localhost:8000/api/v1/assessments/asn_xxxxxxxx/locks
```

实测 `revision=3`、`locks=[]`，上下界恢复为 `[0.040202, 0.552884]`。

可直接运行完整九步示例脚本（自动建夹具、提取 id、断言式打印）：

```bash
bash examples/api_examples.sh http://localhost:8000
```

---

## 6. 计算方法说明

### 6.1 间隔的两种转数假设与转速

间隔 i 长度 `dᵢ = pᵢ₊₁ − pᵢ` 个采样点。若指派 rᵢ ∈ {1,2} 转，其平均转速：

```
vᵢ(r) = r / (dᵢ / 16000) × 60 = 960000·r / dᵢ  (rpm)
```

相邻约束：`|vᵢ(rᵢ) − vᵢ₊₁(rᵢ₊₁)| ≤ Δ`。

### 6.2 重采样与 FFT 能量占比

把间隔段用线性插值均匀重采样到 `N = 64·r` 点（每转 64 点，采样位置取
`[0,d)` 半开区间，整周期信号恰好落在整数谱线上），去均值后做 FFT。
长度 N、r 转对应：

```
阶次 h  ↔  FFT 谱线 k = h·r        （k/r 的单位是 周/转 = 阶次）
占比 qᵢ(r) = (P[k] + P[N−k]) / ΣP   （计入共轭负频率谱线）
ΣP = 0（零能量/数字静音）时 qᵢ(r) = 0
```

加权得分（总转数 R 固定）：

```
S(r) = Σ rᵢ·qᵢ(rᵢ) / Σ rᵢ = Σ rᵢ·qᵢ(rᵢ) / R
```

### 6.3 动态规划求上下界（不枚举 2^m 转法）

状态 `dp[i][r][t]`：处理到第 i 个间隔、末尾转数 r∈{1,2}、累计 t 转时的
最优加权分子；转移只允许满足相邻转速差上限的 `(prev_r, r)`，锁定间隔只允许
锁定值。分别做最小/最大化并保存前驱，最后在 `t = R` 取最优并回溯出完整指派。

- 复杂度 `O(m·R·4) = O(m²)`，m ≤ 100，毫秒级；
- 回溯指派中 `rᵢ = 2` 的间隔即该端“漏了脉冲”的间隔；
- 无解时区分两类原因，并额外用一个 minimax DP 给出
  “当前总转数+锁定下至少需要多大的转速差上限”：
  - `incompatible_total_revolutions_or_locks`（总转数/锁定不相容）；
  - `speed_diff_limit_too_strict`（转速差上限过严，附最小可行 Δ）。

`tests/test_solver.py` 中 40 组随机 m≤9 用例会枚举全部 2^m 转法与 DP 对拍，
验证上下界数值、回溯指派与漏脉冲序号完全一致。

---

## 7. 已实现功能与实测结果

**已实现**

- PCM16 WAV 严格校验：非 WAV/非 PCM、非单声道、非 16 kHz、超 10 秒均返回
  带错误码的 422；
- 脉冲点校验：数量、严格递增、范围、≤100 间隔；
- 每间隔 1/2 转双假设的 64 点/转重采样、去均值 FFT、目标阶次双谱线能量占比、
  零能量记 0；
- 按转数加权得分；总转数/转速差/锁定三重约束的 DP 上下界与两端漏脉冲间隔；
- 三态判定（建议排查 / 待核 / 未达标）及边界相等按达标处理；
- 人工锁定的增改、全量替换、解除；锁定后重算并生成 revision；
- 无解更正返回 409 且**拒绝落库修改**，原锁定/原报告/历史版本/审计均可查询；
- JSON 文件持久化（原子替换写），重启后报告仍可查询；
- Dockerfile + docker-compose（健康检查 + 命名卷）；
- Swagger `/docs`；根路径、`/api`、`/api/v1` 三种路由前缀；
- 字段别名/中文键/data-URL base64/百分数阈值/rps 单位等对接宽容处理。

**实测（2026-11-26，Python 3.11 + FastAPI 0.165 + NumPy 2.4）**

- `pytest`：**60 passed**，其中 40 组随机用例与暴力枚举完全一致，
  另有 100 间隔规模/性能、WAV 校验、锁定事务、持久化模型等测试；
- 真实 uvicorn + curl 九步流程全部符合预期，记录见
  `selftest_live.log`、`selftest_pytest.log`；
- 合成 3 阶样本：严格 Δ=30 rpm 唯一还原真实转法 `[1,2,1,1,2,1]`、
  漏脉冲 `[1,4]`、得分 0.552884 → 建议排查；宽松 Δ=400 rpm 时
  上下界 `[0.040202, 0.552884]`，两端漏脉冲集合分别为 `[3,5]` 与 `[1,4]`，
  中点阈值 0.2965 → 待核；
- 无解锁定更正实测 `HTTP 409`，GET 原报告 revision/lock_map/上下界不变，
  `/audit` 记录 1 次拒绝；DELETE 解除后 bounds 复原。

## 8. 未完成项 / 限制（如实说明）

1. **本沙箱没有 Docker CLI/daemon**，因此 `docker compose up --build`
   未能在当前环境实机执行；`docker-compose.yml` 已通过 YAML 解析校验，
   依赖均为提供 cp311 manylinux wheel 的常规包，且等价流程已用本地
   uvicorn 实测。Docker 构建/启动需在有 Docker 的环境最终确认一次。
2. 服务定位为单机/单车核定工具，持久化用单个 JSON 文件 + 线程锁，
   **未做多副本并发写、鉴权认证与 HTTPS**；上线到车队平台前应加网关鉴权、
   对象存储保存原始 WAV（当前只持久化数值报告，不保存音频字节）。
3. 转速差上限字段按 **rpm** 定义（另支持显式 `rps`）；若现场口径是
   “采样点数/转”或归一化转速差，需要再加一个单位适配。
4. FFT 采用整段双谱线能量口径并计入 ±k 两根谱线；若评审要求只取单边谱或
   加窗，可在 `app/audio.py` 一处调整，DP/接口无需改动。
5. 同一上下界可能存在多个等价最优指派，当前按固定转移顺序返回其中一个
   确定性代表，未枚举全部并列方案（符合“不能逐个枚举转法”的要求）。
