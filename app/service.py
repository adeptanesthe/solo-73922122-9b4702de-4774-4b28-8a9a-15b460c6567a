"""核定业务编排：校验 → 特征 → DP 上下界 → 结论/报告 → 锁定更正。"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import Any

import numpy as np

from .audio import (SAMPLE_RATE, WavValidationError,
                    energy_ratio_for_revs, read_pcm16_wav)
from .solver import actual_max_speed_diff, solve_bounds
from .store import AssessmentStore

MAX_INTERVALS = 100
# 阈值比较容差：消除 0.30000000000000004 这类末位误差
_VERDICT_EPS = 1e-12


class ServiceError(Exception):
    def __init__(self, status_code: int, code: str, message: str,
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _locks_list(locks: dict[int, int]) -> list[dict[str, Any]]:
    return [{
        "interval": i,
        "revolutions": r,
        "missed_pulse": r == 2,
    } for i, r in sorted(locks.items())]


def validate_pulses(pulse_samples: list[int], audio_length: int) -> list[int]:
    if len(pulse_samples) < 2:
        raise ServiceError(
            422, "not_enough_pulses",
            "至少需要 2 个脉冲采样点才能形成间隔。")
    if len(pulse_samples) - 1 > MAX_INTERVALS:
        raise ServiceError(
            422, "too_many_intervals",
            f"至多 {MAX_INTERVALS} 个间隔，当前 "
            f"{len(pulse_samples) - 1} 个（{len(pulse_samples)} 个脉冲点）。")
    for idx, p in enumerate(pulse_samples):
        if p < 0 or p > audio_length:
            raise ServiceError(
                422, "pulse_out_of_range",
                f"第 {idx} 个脉冲采样点 {p} 超出 WAV 采样范围 "
                f"[0, {audio_length}]。")
    for idx in range(1, len(pulse_samples)):
        if pulse_samples[idx] <= pulse_samples[idx - 1]:
            raise ServiceError(
                422, "pulses_not_strictly_increasing",
                "脉冲采样点必须严格递增，"
                f"但位置 {idx - 1}/{idx} 为 "
                f"{pulse_samples[idx - 1]} -> {pulse_samples[idx]}。")
    return pulse_samples


def validate_locks_range(locks: dict[int, int], m: int) -> None:
    for i in locks:
        if i >= m:
            raise ServiceError(
                422, "lock_index_out_of_range",
                f"锁定的间隔序号 {i} 不存在：共有 {m} 个间隔"
                f"（合法序号 0~{m - 1}）。")


def verdict_for(lower: float, upper: float,
                threshold: float) -> dict[str, Any]:
    """三态判定：两端达标=建议排查；仅上界=待核；否则=未达标。"""
    lower_ok = lower + _VERDICT_EPS >= threshold
    upper_ok = upper + _VERDICT_EPS >= threshold
    if lower_ok and upper_ok:
        code, label = "recommend_investigate", "建议排查该阶"
        explanation = ("得分下界与上界均达到阈值，所有满足约束的转法都值得"
                       "按该阶次排查。")
    elif upper_ok:
        code, label = "pending_verification", "待核"
        explanation = ("仅得分上界达到阈值：存在值得排查的转法，也存在达不到"
                       "阈值的转法，需人工核对两端对应的漏脉冲间隔后再定。")
    else:
        code, label = "below_standard", "未达标"
        explanation = "得分上界仍低于阈值，没有任何满足约束的转法达到排查标准。"
    return {
        "code": code,
        "status": code,
        "label_zh": label,
        "lower_reaches_threshold": bool(lower_ok),
        "upper_reaches_threshold": bool(upper_ok),
        "threshold": threshold,
        "explanation_zh": explanation,
    }


class AssessmentService:
    def __init__(self, store: AssessmentStore):
        self.store = store

    # ---------- 特征 + 求解 + 报告 ----------

    def _compute_interval_features(
            self, samples: np.ndarray, pulses: list[int],
            order: int) -> tuple[list[int], list[dict[int, float]],
                                 list[dict[str, Any]]]:
        lengths = [pulses[i + 1] - pulses[i]
                   for i in range(len(pulses) - 1)]
        ratios: list[dict[int, float]] = []
        details: list[dict[str, Any]] = []
        for i, d in enumerate(lengths):
            segment = samples[pulses[i]:pulses[i + 1]]
            q1 = energy_ratio_for_revs(segment, 1, order)
            q2 = energy_ratio_for_revs(segment, 2, order)
            ratios.append({1: q1, 2: q2})
            details.append({
                "index": i,
                "start_sample": pulses[i],
                "end_sample": pulses[i + 1],
                "length_samples": d,
                "duration_seconds": round(d / SAMPLE_RATE, 6),
                "average_speed_rpm": {
                    "1": round(960_000.0 / d, 6),       # r=1: 60*16000/d
                    "2": round(1_920_000.0 / d, 6),     # r=2
                },
                "target_energy_ratio": {
                    "1": q1,
                    "2": q2,
                },
            })
        return lengths, ratios, details

    def _build_endpoint(self, assignment: list[int] | None,
                        missed: list[int] | None,
                        locks: dict[int, int],
                        lengths: list[int],
                        ratios: list[dict[int, float]],
                        total_revolutions: int,
                        score: float | None) -> dict[str, Any] | None:
        if assignment is None:
            return None
        speeds = [{r: 60.0 * SAMPLE_RATE * r / lengths[i]
                   for r in (1, 2)} for i in range(len(lengths))]
        used_speeds = [speeds[i][r] for i, r in enumerate(assignment)]
        return {
            "score": score,
            "assignment_revolutions": assignment,
            "missed_pulse_intervals": missed,
            "inferred_missed_pulse_intervals": [
                i for i in (missed or []) if i not in locks],
            "locked_intervals_in_assignment": [
                i for i in sorted(locks) if assignment[i] == locks[i]],
            "total_revolutions": sum(assignment),
            "missed_pulse_count": len(missed or []),
            "speeds_rpm": used_speeds,
            "max_adjacent_speed_diff_rpm": actual_max_speed_diff(
                assignment, speeds),
        }

    def _build_report(self, *, assessment_id: str | None,
                      created_at: str | None, revision: int,
                      audio_meta: dict[str, Any],
                      pulses: list[int],
                      total_revolutions: int,
                      speed_limit: float,
                      order: int, threshold: float,
                      locks: dict[int, int],
                      lengths: list[int],
                      ratios: list[dict[int, float]],
                      interval_details: list[dict[str, Any]],
                      updated_at: str | None = None) -> dict[str, Any]:
        result = solve_bounds(
            lengths=lengths, ratios=ratios,
            total_revolutions=total_revolutions,
            speed_diff_limit_rpm=speed_limit, locks=locks)
        if not result.feasible:
            raise ServiceError(
                422 if assessment_id is None else 409,
                result.reason_code or "no_feasible_solution",
                result.reason or "约束无解。",
                {"min_required_speed_diff_limit_rpm":
                 result.min_required_speed_diff_limit_rpm,
                 **result.diagnostics})

        lower_ep = self._build_endpoint(
            result.lower_assignment, result.lower_missed, locks,
            lengths, ratios, total_revolutions, result.lower_score)
        upper_ep = self._build_endpoint(
            result.upper_assignment, result.upper_missed, locks,
            lengths, ratios, total_revolutions, result.upper_score)
        verdict = verdict_for(result.lower_score, result.upper_score, threshold)

        return {
            "assessment_id": assessment_id or f"asn_{uuid.uuid4().hex[:16]}",
            "revision": revision,
            "created_at": created_at or _now(),
            "updated_at": updated_at or _now(),
            "audio": audio_meta,
            "sample_rate_hz": SAMPLE_RATE,
            "pulse_samples": pulses,
            "interval_count": len(lengths),
            "total_revolutions": total_revolutions,
            "speed_diff_limit_rpm": speed_limit,
            "order": order,
            "threshold": threshold,
            "locks": _locks_list(locks),
            "lock_map": {str(i): r for i, r in sorted(locks.items())},
            "feasible": True,
            "score_lower": result.lower_score,
            "score_upper": result.upper_score,
            # 同值别名，方便不同客户端取值
            "lower_bound": result.lower_score,
            "upper_bound": result.upper_score,
            "lower_missed_pulse_intervals": result.lower_missed,
            "upper_missed_pulse_intervals": result.upper_missed,
            "bounds": {
                "lower": result.lower_score,
                "upper": result.upper_score,
                "lower_missed_pulse_intervals": result.lower_missed,
                "upper_missed_pulse_intervals": result.upper_missed,
            },
            "verdict": verdict,
            # 顶层冗余三态字段，方便列表/网关直接取用
            "status": verdict["status"],
            "status_label_zh": verdict["label_zh"],
            "lower_endpoint": lower_ep,
            "upper_endpoint": upper_ep,
            "interval_details": interval_details,
            "algorithm": {
                "points_per_revolution": 64,
                "fft_target_bin_formula": "k = order * revolutions",
                "energy_ratio": "(P[k] + P[N-k]) / sum(P); zero energy -> 0",
                "weighting": "sum(revolutions_i * ratio_i) / total_revolutions",
                "bounds_method": "dynamic_programming_O(m^2)",
                "enumerates_assignments": False,
            },
        }

    # ---------- 创建核定 ----------

    def create_assessment(self, *, wav_bytes: bytes,
                          pulse_samples: list[int],
                          total_revolutions: int,
                          speed_diff_limit_rpm: float,
                          order: int, threshold: float,
                          locks: dict[int, int]) -> dict[str, Any]:
        try:
            samples = read_pcm16_wav(wav_bytes)
        except WavValidationError as exc:
            raise ServiceError(422, exc.code, exc.message) from exc

        pulses = validate_pulses(pulse_samples, samples.size)
        m = len(pulses) - 1
        validate_locks_range(locks, m)

        lengths, ratios, details = self._compute_interval_features(
            samples, pulses, order)
        audio_meta = {
            "sample_rate_hz": SAMPLE_RATE,
            "channels": 1,
            "sample_width_bytes": 2,
            "sample_count": int(samples.size),
            "duration_seconds": round(samples.size / SAMPLE_RATE, 6),
            "format": "PCM16 WAV",
        }
        # 创建阶段无解：直接 422，不落库
        report = self._build_report(
            assessment_id=None, created_at=None, revision=1,
            audio_meta=audio_meta, pulses=pulses,
            total_revolutions=total_revolutions,
            speed_limit=speed_diff_limit_rpm,
            order=order, threshold=threshold, locks=locks,
            lengths=lengths, ratios=ratios, interval_details=details)
        self.store.create(report)
        return report

    # ---------- 从已存报告重建求解输入（无需保存 WAV） ----------

    @staticmethod
    def _inputs_from_report(report: dict[str, Any]):
        pulses = report["pulse_samples"]
        lengths = [d["length_samples"] for d in report["interval_details"]]
        ratios = [{1: float(d["target_energy_ratio"]["1"]),
                   2: float(d["target_energy_ratio"]["2"])}
                  for d in report["interval_details"]]
        return pulses, lengths, ratios

    def _rebuild_with_locks(self, report: dict[str, Any],
                            new_locks: dict[int, int]) -> dict[str, Any]:
        pulses, lengths, ratios = self._inputs_from_report(report)
        validate_locks_range(new_locks, len(lengths))
        # interval_details 中的 locked 标记在新报告里由 locks 重新表达，
        # 特征数值不变，直接复用。
        return self._build_report(
            assessment_id=report["assessment_id"],
            created_at=report["created_at"],
            revision=report["revision"] + 1,
            audio_meta=report["audio"],
            pulses=pulses,
            total_revolutions=report["total_revolutions"],
            speed_limit=report["speed_diff_limit_rpm"],
            order=report["order"],
            threshold=report["threshold"],
            locks=new_locks,
            lengths=lengths,
            ratios=ratios,
            interval_details=report["interval_details"],
            updated_at=_now())

    # ---------- 锁定更正 / 解除 ----------

    def change_locks(self, assessment_id: str,
                     new_locks: dict[int, int]) -> dict[str, Any]:
        current = self.store.get_current_report(assessment_id)
        if current is None:
            raise ServiceError(404, "assessment_not_found",
                               f"找不到核定报告 {assessment_id}。")
        old_locks = {int(k): v for k, v in current.get("lock_map", {}).items()}
        try:
            new_report = self._rebuild_with_locks(current, new_locks)
        except ServiceError as exc:
            # 无解：拒绝更正，记录审计；current/revisions 原封不动
            self.store.add_rejected_attempt(assessment_id, {
                "at": _now(),
                "submitted_locks": _locks_list(new_locks),
                "previous_locks": _locks_list(old_locks),
                "error": {"code": exc.code, "message": exc.message,
                          "details": exc.details},
            })
            exc.details = {
                **exc.details,
                "rejected": True,
                "assessment_id": assessment_id,
                "current_revision": current["revision"],
                "current_locks": current.get("locks"),
                "current_report": current,
                "hint": "更正已被拒绝，原锁定与原报告保持不变，可继续 GET 查询。",
            }
            raise
        self.store.commit_revision(new_report)
        return new_report

    def release_locks(self, assessment_id: str) -> dict[str, Any]:
        return self.change_locks(assessment_id, {})

    # ---------- 查询 ----------

    def get_report(self, assessment_id: str) -> dict[str, Any]:
        report = self.store.get_current_report(assessment_id)
        if report is None:
            raise ServiceError(404, "assessment_not_found",
                               f"找不到核定报告 {assessment_id}。")
        return report
