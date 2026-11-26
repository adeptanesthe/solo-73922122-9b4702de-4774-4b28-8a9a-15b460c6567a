"""DP 上下界与小规模暴力枚举的对照测试。

小规模（m<=10）时枚举全部 2^m 转法，逐一过滤约束并计算加权得分，
断言 DP 的上下界、还原指派、漏脉冲序号全部一致。另含 m=100 性能/规模测试。
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from app.solver import (REV_CHOICES, _diff_within_limit,
                        average_speed_rpm, score_assignment, solve_bounds)


def brute_force(lengths, ratios, R, delta, locks):
    m = len(lengths)
    speed = [{r: average_speed_rpm(d, r) for r in REV_CHOICES}
             for d in lengths]
    feasible = []
    for assignment in itertools.product(REV_CHOICES, repeat=m):
        assignment = list(assignment)
        if sum(assignment) != R:
            continue
        if any(i in locks and assignment[i] != r for i, r in locks.items()):
            continue
        ok = True
        for i in range(1, m):
            diff = abs(speed[i - 1][assignment[i - 1]]
                       - speed[i][assignment[i]])
            if not _diff_within_limit(diff, delta):
                ok = False
                break
        if not ok:
            continue
        feasible.append(assignment)
    if not feasible:
        return None
    scored = [(score_assignment(a, ratios, R), a) for a in feasible]
    lo = min(s for s, _ in scored)
    hi = max(s for s, _ in scored)
    return lo, hi, scored


@pytest.mark.parametrize("seed", range(40))
def test_dp_matches_brute_force_random(seed):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(1, 10))
    lengths = [int(x) for x in rng.integers(200, 20000, size=m)]
    ratios = [{1: float(rng.random()), 2: float(rng.random())}
              for _ in range(m)]
    R = int(rng.integers(m, 2 * m + 1))
    delta = float(rng.choice([0.0, 30.0, 120.0, 400.0, 10_000.0]))

    locks = {}
    for i in range(m):
        pick = rng.random()
        if pick < 0.2:
            locks[i] = int(rng.choice([1, 2]))

    expected = brute_force(lengths, ratios, R, delta, locks)
    result = solve_bounds(lengths=lengths, ratios=ratios,
                          total_revolutions=R,
                          speed_diff_limit_rpm=delta, locks=locks)

    if expected is None:
        assert not result.feasible
        return

    exp_lo, exp_hi, scored = expected
    assert result.feasible
    assert result.lower_score == pytest.approx(exp_lo, abs=1e-12)
    assert result.upper_score == pytest.approx(exp_hi, abs=1e-12)

    # 还原出的指派必须真实可行且确实达到对应界
    for endpoint, bound in ((result.lower_assignment, result.lower_score),
                            (result.upper_assignment, result.upper_score)):
        assert sum(endpoint) == R
        assert all(i not in locks or endpoint[i] == r
                   for i, r in locks.items())
        assert score_assignment(endpoint, ratios, R) == pytest.approx(
            bound, abs=1e-12)
        # 与暴力集合中的某个最优转法一致（容差内）
        assert any(endpoint == a and abs(s - bound) < 1e-12
                   for s, a in scored)

    # 漏脉冲序号与指派中的 2 转位置一致
    assert result.lower_missed == [
        i for i, r in enumerate(result.lower_assignment) if r == 2]
    assert result.upper_missed == [
        i for i, r in enumerate(result.upper_assignment) if r == 2]
    assert result.lower_score <= result.upper_score + 1e-12


def test_infeasible_total_revolutions():
    m = 4
    lengths = [4000] * m
    ratios = [{1: 0.1, 2: 0.2} for _ in range(m)]
    result = solve_bounds(lengths=lengths, ratios=ratios,
                          total_revolutions=99,
                          speed_diff_limit_rpm=1e9, locks={})
    assert not result.feasible
    assert result.reason_code == "incompatible_total_revolutions_or_locks"
    assert result.diagnostics["minimum_possible_total_revolutions"] == 4
    assert result.diagnostics["maximum_possible_total_revolutions"] == 8


def test_locks_force_too_many_double_revs():
    m = 5
    lengths = [4000] * m
    ratios = [{1: 0.1, 2: 0.2} for _ in range(m)]
    locks = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}
    result = solve_bounds(lengths=lengths, ratios=ratios,
                          total_revolutions=6,
                          speed_diff_limit_rpm=1e9, locks=locks)
    assert not result.feasible


def test_speed_limit_infeasible_has_minimax_diagnostic():
    # 长/短间隔交替：任何满足总转数的指派都有很大转速落差，
    # 给 0 rpm 上限必然无解，诊断值应为正且有限。
    lengths = [1000, 20000, 1000, 20000]
    ratios = [{1: 0.1, 2: 0.2} for _ in lengths]
    result = solve_bounds(lengths=lengths, ratios=ratios,
                          total_revolutions=6,
                          speed_diff_limit_rpm=0.0, locks={})
    assert not result.feasible
    assert result.reason_code == "speed_diff_limit_too_strict"
    assert result.min_required_speed_diff_limit_rpm > 0


def test_max_100_intervals_performance():
    m = 100
    rng = np.random.default_rng(0)
    lengths = [int(x) for x in rng.integers(3000, 9000, size=m)]
    ratios = [{1: float(rng.random()), 2: float(rng.random())}
              for _ in range(m)]
    R = 130
    start = time.perf_counter()
    result = solve_bounds(lengths=lengths, ratios=ratios,
                          total_revolutions=R,
                          speed_diff_limit_rpm=300.0, locks={5: 2, 9: 1})
    elapsed = time.perf_counter() - start
    assert result.feasible
    assert len(result.lower_assignment) == m
    assert sum(result.upper_assignment) == R
    assert 5 in result.upper_missed and result.upper_assignment[9] == 1
    assert elapsed < 2.0  # O(m^2) 应当极快
