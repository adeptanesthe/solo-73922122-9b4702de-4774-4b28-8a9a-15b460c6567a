"""请求字段归一化与人工锁定解析。

为方便现场对接，同一参数接受多个常见字段名（含中文键），锁定支持：

* ``{"3": 2, "7": 1}``                      —— 间隔序号 -> 锁定转数；
* ``[{"interval": 3, "revolutions": 2}]``   —— 显式转数；
* ``[{"interval": 3, "missed_pulse": true}]``—— 是否漏脉冲（true=2 转）；
* ``[3, 7]``                                 —— 这些间隔锁定为 2 转（漏脉冲）；
* 另支持 ``locked_missed_intervals`` /
  ``locked_single_rev_intervals`` 两个便捷数组。
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from pydantic import BaseModel, Field

PULSE_KEYS = ("pulse_samples", "pulse_points", "pulses",
              "pulse_sample_points", "sample_points", "脉冲采样点")
TOTAL_REV_KEYS = ("total_revolutions", "controller_total_revolutions",
                  "total_revs", "controller_revolutions",
                  "controller_total_revs", "总转数", "控制器总转数")
SPEED_LIMIT_KEYS = ("speed_diff_limit_rpm", "adjacent_speed_diff_limit_rpm",
                    "speed_difference_limit_rpm", "max_speed_diff_rpm",
                    "speed_diff_limit", "rpm_diff_limit",
                    "相邻间隔平均转速差上限", "转速差上限")
SPEED_UNIT_KEYS = ("speed_diff_unit", "speed_unit", "转速差单位")
ORDER_KEYS = ("order", "target_order", "investigate_order",
              "order_to_check", "阶次", "待查阶次")
THRESHOLD_KEYS = ("threshold", "decision_threshold",
                  "energy_ratio_threshold", "judgement_threshold",
                  "判定阈值", "阈值")
THRESHOLD_PERCENT_KEYS = ("threshold_percent", "threshold_pct")
LOCK_KEYS = ("locks", "manual_locks", "locked_intervals",
             "manual_lock", "人工锁定", "锁定")
LOCKED_MISSED_KEYS = ("locked_missed_intervals", "missed_intervals_locked",
                      "锁定漏脉冲间隔")
LOCKED_SINGLE_KEYS = ("locked_single_rev_intervals",
                      "single_revolution_intervals_locked",
                      "锁定单转间隔")
AUDIO_B64_KEYS = ("audio_wav_base64", "wav_base64", "audio_base64",
                  "wav", "audio")
AUDIO_FILE_PARTS = ("audio", "wav", "file", "audio_file", "wav_file")


class PayloadError(ValueError):
    """请求字段无法解析。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------- OpenAPI 用的规范模型 ----------

class LockSpec(BaseModel):
    interval: int = Field(ge=0, description="间隔序号（从 0 开始）")
    revolutions: int | None = Field(default=None, description="锁定转数 1 或 2")
    missed_pulse: bool | None = Field(
        default=None, description="true 表示该间隔漏了脉冲，即锁定为 2 转")


class AssessmentJsonRequest(BaseModel):
    """JSON 核定请求；WAV 以 base64（可带 data: URL 前缀）放入。"""

    audio_wav_base64: str = Field(description="16kHz/单声道/PCM16 WAV 的 base64")
    pulse_samples: list[int] = Field(description="脉冲采样点序号，至少 2 个")
    total_revolutions: int = Field(ge=1, description="控制器总转数 R")
    speed_diff_limit_rpm: float = Field(
        ge=0, description="相邻间隔平均转速差上限 Δ（rpm）")
    order: int = Field(ge=1, le=8, description="待查阶次 1~8")
    threshold: float = Field(ge=0, le=1, description="判定阈值（能量占比）")
    locks: Any | None = Field(default=None, description="人工锁定，格式见模块说明")
    speed_diff_unit: str | None = Field(
        default="rpm", description="转速差上限单位：rpm（默认）或 rps")


class LockUpdateRequest(BaseModel):
    locks: Any | None = Field(
        default=None,
        description="全量锁定表（替换旧锁定）；空对象/空数组表示全部解除")
    release_all: bool = Field(
        default=False, description="为 true 时解除全部人工锁定")


# ---------- 取值/转换辅助 ----------

def _lookup(raw: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
    # 大小写/连字符不敏感兜底
    folded = {str(k).lower().replace("-", "_"): v for k, v in raw.items()}
    for key in keys:
        v = folded.get(key.lower())
        if v is not None:
            return v
    return None


def coerce_jsonish(value: Any, field_name: str) -> Any:
    """表单字段可能是 JSON 字符串或逗号分隔列表，尽量还原。"""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("[") or text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise PayloadError(
                "invalid_json_field",
                f"字段 {field_name} 不是合法 JSON：{exc}") from exc
    if "," in text and field_name in ("pulse_samples",):
        return [part.strip() for part in text.split(",") if part.strip()]
    return value


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise PayloadError("invalid_integer", f"{field_name} 不能是布尔值。")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise PayloadError(
                "invalid_integer", f"{field_name} 不是整数：{value!r}") from exc
    raise PayloadError("invalid_integer", f"{field_name} 不是整数：{value!r}")


def _as_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise PayloadError("invalid_number", f"{field_name} 不能是布尔值。")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PayloadError(
            "invalid_number", f"{field_name} 不是数值：{value!r}") from exc


def parse_locks(raw_locks: Any) -> dict[int, int]:
    """把各种锁定写法归一为 ``{interval_index: 1|2}``。"""
    if raw_locks is None:
        return {}
    if isinstance(raw_locks, str):
        raw_locks = coerce_jsonish(raw_locks, "locks")
    if raw_locks in ({}, []):
        return {}

    locks: dict[int, int] = {}

    def put(idx: Any, rev: Any, *, missed: Any = None) -> None:
        i = _as_int(idx, "locks.interval")
        if i < 0:
            raise PayloadError("invalid_lock_index",
                               f"锁定间隔序号不能为负：{i}")
        if missed is not None:
            if not isinstance(missed, bool):
                raise PayloadError(
                    "invalid_lock_value",
                    f"locks[{i}].missed_pulse 必须是布尔值。")
            r = 2 if missed else 1
        else:
            if isinstance(rev, bool):
                r = 2 if rev else 1
            else:
                r = _as_int(rev, f"locks[{i}].revolutions")
        if r not in (1, 2):
            raise PayloadError(
                "invalid_lock_value",
                f"间隔 {i} 只能锁定为 1 转或 2 转，收到 {r}。")
        if i in locks and locks[i] != r:
            raise PayloadError(
                "conflicting_lock",
                f"间隔 {i} 同时被锁定为 {locks[i]} 转和 {r} 转。")
        locks[i] = r

    if isinstance(raw_locks, dict):
        for key, value in raw_locks.items():
            if isinstance(value, dict):
                put(value.get("interval", key),
                    value.get("revolutions", value.get("r", value.get("revs"))),
                    missed=value.get("missed_pulse", value.get("missed")))
            else:
                put(key, value)
    elif isinstance(raw_locks, list):
        for item in raw_locks:
            if isinstance(item, int) and not isinstance(item, bool):
                # 纯序号数组：约定为“这些间隔锁定漏脉冲（2 转）”
                put(item, 2)
            elif isinstance(item, dict):
                idx = item.get("interval", item.get("index", item.get("i")))
                if idx is None:
                    raise PayloadError(
                        "invalid_lock", f"锁定项缺少 interval：{item!r}")
                put(idx,
                    item.get("revolutions",
                             item.get("r", item.get("revs"))),
                    missed=item.get("missed_pulse", item.get("missed")))
            else:
                raise PayloadError("invalid_lock", f"无法识别的锁定项：{item!r}")
    else:
        raise PayloadError("invalid_lock",
                           "locks 必须是对象、数组或 JSON 字符串。")
    return dict(sorted(locks.items()))


def _parse_index_list(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = coerce_jsonish(raw, "interval_list")
    if not isinstance(raw, list):
        raise PayloadError("invalid_interval_list",
                           "间隔序号数组必须是列表或 JSON 数组字符串。")
    return [_as_int(v, "interval_index") for v in raw]


def normalize_request(raw: dict[str, Any]) -> dict[str, Any]:
    """从任意（JSON 或表单拍平后的）字典提取并校验核定参数。

    返回 canonical dict：
    pulse_samples / total_revolutions / speed_diff_limit_rpm /
    order / threshold / locks。
    """
    pulse_raw = _lookup(raw, PULSE_KEYS)
    if pulse_raw is None:
        raise PayloadError("missing_field", "缺少 pulse_samples（脉冲采样点）。")
    pulse_raw = coerce_jsonish(pulse_raw, "pulse_samples")
    if not isinstance(pulse_raw, (list, tuple)):
        raise PayloadError("invalid_pulse_samples",
                           "pulse_samples 必须是整数数组。")
    pulse_samples = [_as_int(v, "pulse_samples") for v in pulse_raw]

    total_raw = _lookup(raw, TOTAL_REV_KEYS)
    if total_raw is None:
        raise PayloadError("missing_field",
                           "缺少 total_revolutions（控制器总转数）。")
    total_revolutions = _as_int(total_raw, "total_revolutions")
    if total_revolutions < 1:
        raise PayloadError("invalid_total_revolutions",
                           "total_revolutions 必须为正整数。")

    speed_raw = _lookup(raw, SPEED_LIMIT_KEYS)
    if speed_raw is None:
        raise PayloadError(
            "missing_field",
            "缺少 speed_diff_limit_rpm（相邻间隔平均转速差上限）。")
    speed_limit = _as_float(speed_raw, "speed_diff_limit_rpm")
    if speed_limit < 0:
        raise PayloadError("invalid_speed_limit",
                           "转速差上限不能为负。")
    unit = _lookup(raw, SPEED_UNIT_KEYS)
    if unit is not None and str(unit).strip().lower() in ("rps", "hz", "1/s"):
        speed_limit *= 60.0  # 转/秒差 → rpm 差

    order_raw = _lookup(raw, ORDER_KEYS)
    if order_raw is None:
        raise PayloadError("missing_field", "缺少 order（待查阶次）。")
    order = _as_int(order_raw, "order")
    if not 1 <= order <= 8:
        raise PayloadError("invalid_order", "order 必须在 1~8 之间。")

    threshold: float | None = None
    threshold_raw = _lookup(raw, THRESHOLD_KEYS)
    if threshold_raw is not None:
        threshold = _as_float(threshold_raw, "threshold")
        if not 0.0 <= threshold <= 1.0:
            raise PayloadError(
                "invalid_threshold",
                "threshold 必须在 0~1 之间（能量占比）；"
                "若给的是百分数请使用 threshold_percent。")
    percent_raw = _lookup(raw, THRESHOLD_PERCENT_KEYS)
    if percent_raw is not None:
        percent = _as_float(percent_raw, "threshold_percent")
        if not 0.0 <= percent <= 100.0:
            raise PayloadError("invalid_threshold",
                               "threshold_percent 必须在 0~100 之间。")
        threshold = percent / 100.0
    if threshold is None:
        raise PayloadError("missing_field", "缺少 threshold（判定阈值）。")

    locks = parse_locks(_lookup(raw, LOCK_KEYS))
    for idx in _parse_index_list(_lookup(raw, LOCKED_MISSED_KEYS)):
        if idx in locks and locks[idx] != 2:
            raise PayloadError("conflicting_lock",
                               f"间隔 {idx} 的便捷锁定与 locks 冲突。")
        locks[idx] = 2
    for idx in _parse_index_list(_lookup(raw, LOCKED_SINGLE_KEYS)):
        if idx in locks and locks[idx] != 1:
            raise PayloadError("conflicting_lock",
                               f"间隔 {idx} 的便捷锁定与 locks 冲突。")
        locks[idx] = 1

    return {
        "pulse_samples": pulse_samples,
        "total_revolutions": total_revolutions,
        "speed_diff_limit_rpm": speed_limit,
        "order": order,
        "threshold": threshold,
        "locks": dict(sorted(locks.items())),
    }


def decode_audio_b64(value: str) -> bytes:
    """解码 base64 WAV，兼容 ``data:audio/wav;base64,xxxx`` 前缀。"""
    if not isinstance(value, str):
        raise PayloadError("invalid_audio", "audio_wav_base64 必须是字符串。")
    text = value.strip()
    if "," in text and text.startswith("data:"):
        text = text.split(",", 1)[1]
    try:
        return base64.b64decode(text, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise PayloadError("invalid_audio_base64",
                           f"WAV base64 解码失败：{exc}") from exc
