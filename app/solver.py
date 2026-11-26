"""阶次得分上下界求解。

业务模型
--------
探头可能漏脉冲，因此相邻两个脉冲采样点之间的间隔只可能对应 1 转或 2 转
（2 转即中间漏了 1 个脉冲）。设共 m 个间隔，间隔 i 的采样长度为 d_i，
指派转数 r_i ∈ {1, 2}，则：

* 总转数约束：Σ r_i = R（控制器总转数）；
* 转速差约束：|v_i(r_i) - v_{i+1}(r_{i+1})| ≤ Δ，
  其中平均转速 v_i(r) = 60 · 16000 · r / d_i (rpm)；
* 人工锁定：被锁定间隔只能取指定转数。

每个间隔在两种转数假设下的目标阶次能量占比 q_i(1)、q_i(2) 可由
``audio.energy_ratio_for_revs`` 预先算好。得分（按转数加权）为::

    S(r) = Σ r_i · q_i(r_i) / Σ r_i = Σ r_i · q_i(r_i) / R

R 固定，因此上下界等价于对加权分子做最小/最大化。

不能逐个枚举 2^m 种转法。这里用动态规划：状态为
``dp[i][r][t]`` = 处理到间隔 i、末尾转数为 r、累计转数为 t 时的最优分子，
并保存前驱用于还原“哪些间隔是 2 转（漏脉冲）”。
复杂度 O(m · R · 2 · 2) = O(m²)，m ≤ 100。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

SAMPLE_RATE = 16_000
REV_CHOICES = (1, 2)


def average_speed_rpm(length_samples: int, revolutions: int,
                      sample_rate: int = SAMPLE_RATE) -> float:
    """间隔平均转速（rpm）= 转数 / 间隔秒数 × 60。"""
    duration_seconds = length_samples / sample_rate
    return revolutions * 60.0 / duration_seconds


def score_assignment(assignment: list[int],
                     ratios: list[dict[int, float]],
                     total_revolutions: int) -> float:
    """按转数加权的平均能量占比；总转数为 0 时按零能量约定记 0。"""
    if total_revolutions <= 0:
        return 0.0
    numerator = sum(r * ratios[i][r] for i, r in enumerate(assignment))
    return numerator / total_revolutions


@dataclass
class BoundsResult:
    """DP 求解结果。无解时 feasible=False 并带诊断信息。"""

    feasible: bool
    lower_score: float | None = None
    upper_score: float | None = None
    lower_assignment: list[int] | None = None
    upper_assignment: list[int] | None = None
    lower_missed: list[int] | None = None
    upper_missed: list[int] | None = None
    reason_code: str | None = None
    reason: str | None = None
    # 在总转数 + 锁定约束下，达到可行所需的最小“最大相邻转速差”。
    # None 表示仅靠总转数/锁定就已无解。
    min_required_speed_diff_limit_rpm: float | None = None
    diagnostics: dict = field(default_factory=dict)


def _edge_diff(speed: list[dict[int, float]], i: int,
               prev_r: int, r: int) -> float:
    """间隔 i-1（prev_r 转）与间隔 i（r 转）的平均转速差。"""
    return abs(speed[i - 1][prev_r] - speed[i][r])


def _diff_within_limit(diff: float, limit: float) -> bool:
    # 容差仅消除浮点末位误差，量级远小于 0.001 rpm。
    tol = 1e-9 * max(1.0, abs(diff), abs(limit))
    return diff <= limit + tol


def _optimize(direction: str,
              m: int,
              total_revolutions: int,
              locks: dict[int, int],
              speed: list[dict[int, float]],
              weight: list[dict[int, float]],
              speed_diff_limit_rpm: float):
    """求最小/最大加权分子，并返回 (最优值, 指派, 前驱表)。"""
    minimizing = direction == "min"
    sentinel = math.inf if minimizing else -math.inf

    def is_better(candidate: float, incumbent: float) -> bool:
        if not math.isfinite(incumbent):
            return True
        return candidate < incumbent if minimizing else candidate > incumbent

    # parent[i][r][t] = 上一间隔的转数
    parent = [[[-1] * (total_revolutions + 1) for _ in range(3)]
              for _ in range(m)]

    # vals[r][t]：滚动 DP
    vals = [[sentinel] * (total_revolutions + 1) for _ in range(3)]
    lock0 = locks.get(0)
    for r in REV_CHOICES:
        if lock0 is not None and lock0 != r:
            continue
        if r <= total_revolutions:
            vals[r][r] = weight[0][r]

    for i in range(1, m):
        new_vals = [[sentinel] * (total_revolutions + 1) for _ in range(3)]
        locked_r = locks.get(i)
        for r in REV_CHOICES:
            if locked_r is not None and locked_r != r:
                continue
            wi = weight[i][r]
            # t 至少要容下前 i 个间隔各 1 转再加当前 r
            t_min = i + r
            for t in range(t_min, total_revolutions + 1):
                prev_t = t - r
                best_value = sentinel
                best_prev_r = -1
                for prev_r in REV_CHOICES:
                    prev_value = vals[prev_r][prev_t]
                    if not math.isfinite(prev_value):
                        continue
                    diff = _edge_diff(speed, i, prev_r, r)
                    if not _diff_within_limit(diff, speed_diff_limit_rpm):
                        continue
                    candidate = prev_value + wi
                    if best_prev_r == -1 or is_better(candidate, best_value):
                        best_value = candidate
                        best_prev_r = prev_r
                if best_prev_r != -1:
                    new_vals[r][t] = best_value
                    parent[i][r][t] = best_prev_r
        vals = new_vals

    best_r = -1
    best_value = sentinel
    for r in REV_CHOICES:
        value = vals[r][total_revolutions]
        if math.isfinite(value) and (best_r == -1 or is_better(value, best_value)):
            best_value = value
            best_r = r

    if best_r == -1:
        return None, None

    assignment = [0] * m
    r = best_r
    t = total_revolutions
    for i in range(m - 1, -1, -1):
        assignment[i] = r
        if i > 0:
            prev_r = parent[i][r][t]
            t -= r
            r = prev_r
    return best_value, assignment


def _minimax_speed_diff(m: int,
                        total_revolutions: int,
                        locks: dict[int, int],
                        speed: list[dict[int, float]]) -> float | None:
    """诊断用：在总转数+锁定可行的前提下，最小可达到的最大相邻转速差。

    返回 None 表示总转数/锁定本身无解（与转速差上限无关）。
    """
    inf = math.inf
    vals = [[inf] * (total_revolutions + 1) for _ in range(3)]
    lock0 = locks.get(0)
    for r in REV_CHOICES:
        if (lock0 is None or lock0 == r) and r <= total_revolutions:
            vals[r][r] = 0.0

    for i in range(1, m):
        new_vals = [[inf] * (total_revolutions + 1) for _ in range(3)]
        locked_r = locks.get(i)
        for r in REV_CHOICES:
            if locked_r is not None and locked_r != r:
                continue
            t_min = i + r
            for t in range(t_min, total_revolutions + 1):
                prev_t = t - r
                best = inf
                for prev_r in REV_CHOICES:
                    prev_value = vals[prev_r][prev_t]
                    if not math.isfinite(prev_value):
                        continue
                    diff = _edge_diff(speed, i, prev_r, r)
                    candidate = max(prev_value, diff)
                    if candidate < best:
                        best = candidate
                new_vals[r][t] = best
        vals = new_vals

    answer = min(vals[r][total_revolutions] for r in REV_CHOICES)
    return None if not math.isfinite(answer) else answer


def actual_max_speed_diff(assignment: list[int],
                          speed: list[dict[int, float]]) -> float:
    """给定指派下实际的最大相邻转速差（rpm）；单间隔时为 0。"""
    worst = 0.0
    for i in range(1, len(assignment)):
        worst = max(worst, _edge_diff(speed, i, assignment[i - 1], assignment[i]))
    return worst


def solve_bounds(*,
                 lengths: list[int],
                 ratios: list[dict[int, float]],
                 total_revolutions: int,
                 speed_diff_limit_rpm: float,
                 locks: dict[int, int],
                 sample_rate: int = SAMPLE_RATE) -> BoundsResult:
    """计算可行转法下的得分下界/上界，并还原两端的漏脉冲间隔。

    调用方负责校验间隔序号、脉冲点数等；本函数对“无解”给出结构化原因。
    """
    m = len(lengths)
    if m == 0:
        return BoundsResult(False, reason_code="no_intervals",
                            reason="至少需要 2 个脉冲采样点以形成 1 个间隔。")

    # —— 总转数 / 锁定的快速可行性边界 ——
    min_total = sum(locks.get(i, 1) for i in range(m))
    max_total = sum(locks.get(i, 2) for i in range(m))
    count_possible = min_total <= total_revolutions <= max_total

    speed: list[dict[int, float]] = []
    weight: list[dict[int, float]] = []
    for i, d in enumerate(lengths):
        speed.append({r: average_speed_rpm(d, r, sample_rate) for r in REV_CHOICES})
        weight.append({r: r * ratios[i][r] for r in REV_CHOICES})

    minimax = _minimax_speed_diff(m, total_revolutions, locks, speed)

    if not count_possible or minimax is None:
        return BoundsResult(
            False,
            reason_code="incompatible_total_revolutions_or_locks",
            reason=("总转数与间隔数、人工锁定不相容："
                    f"{m} 个间隔在当前锁定下总转数只能落在 "
                    f"[{min_total}, {max_total}]，但控制器总转数为 "
                    f"{total_revolutions}。"),
            min_required_speed_diff_limit_rpm=None,
            diagnostics={"minimum_possible_total_revolutions": min_total,
                         "maximum_possible_total_revolutions": max_total})

    lower_value, lower_assignment = _optimize(
        "min", m, total_revolutions, locks, speed, weight, speed_diff_limit_rpm)
    upper_value, upper_assignment = _optimize(
        "max", m, total_revolutions, locks, speed, weight, speed_diff_limit_rpm)

    if lower_assignment is None or upper_assignment is None:
        return BoundsResult(
            False,
            reason_code="speed_diff_limit_too_strict",
            reason=("在满足总转数与人工锁定的转法中，没有任何转法能让全部相邻"
                    f"间隔的平均转速差 ≤ {speed_diff_limit_rpm:g} rpm；"
                    f"该组约束至少需要允许约 {minimax:.3f} rpm 的最大相邻转速差。"),
            min_required_speed_diff_limit_rpm=minimax,
            diagnostics={"minimum_achievable_max_speed_diff_rpm": minimax,
                         "submitted_speed_diff_limit_rpm": speed_diff_limit_rpm})

    lower_score = score_assignment(lower_assignment, ratios, total_revolutions)
    upper_score = score_assignment(upper_assignment, ratios, total_revolutions)
    return BoundsResult(
        True,
        lower_score=lower_score,
        upper_score=upper_score,
        lower_assignment=lower_assignment,
        upper_assignment=upper_assignment,
        lower_missed=[i for i, r in enumerate(lower_assignment) if r == 2],
        upper_missed=[i for i, r in enumerate(upper_assignment) if r == 2],
        min_required_speed_diff_limit_rpm=minimax)
