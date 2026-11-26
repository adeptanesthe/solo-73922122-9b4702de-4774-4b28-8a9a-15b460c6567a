"""生成 API 示例所需的 WAV 与请求 JSON。

构造一个 6 间隔、真实转数 [1,2,1,1,2,1]（总转数 8）的样本：
每转基准 4000 个采样点（240 rpm），在两个真实 2 转间隔里加入很强的
3 阶正弦成分。严格转速差上限会唯一锁定真实转法；放宽上限后上下界拉开，
脚本自动取两端中点作为阈值，稳定得到“待核”。

用法（在仓库根目录）：
    .venv/bin/python examples/make_fixtures.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.audio import energy_ratio_for_revs  # noqa: E402
from app.solver import solve_bounds  # noqa: E402

SR = 16_000
SAMPLES_PER_REV = 4000
TRUTH = [1, 2, 1, 1, 2, 1]
ORDER = 3
STRONG_INTERVALS = {1, 4}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")


def write_wav(path: str, samples: np.ndarray) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm16.tobytes())


def build_audio() -> tuple[np.ndarray, list[int], list[int]]:
    lengths = [r * SAMPLES_PER_REV for r in TRUTH]
    pulses = [0]
    for d in lengths:
        pulses.append(pulses[-1] + d)

    rng = np.random.default_rng(20261126)
    chunks: list[np.ndarray] = []
    for i, (rt, d) in enumerate(zip(TRUTH, lengths)):
        n = np.arange(d, dtype=np.float64)
        sig = 0.22 * np.sin(2 * np.pi * 1 * rt * n / d)          # 1 阶背景
        amp = 0.9 if i in STRONG_INTERVALS else 0.10
        sig += amp * np.sin(2 * np.pi * ORDER * rt * n / d + 0.7)  # 目标 3 阶
        sig += 0.02 * rng.standard_normal(d)
        sig /= max(1.0, np.max(np.abs(sig)) / 0.95)
        chunks.append(sig)
    audio = np.concatenate(chunks)
    return audio, pulses, lengths


def wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    pcm16 = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    audio, pulses, lengths = build_audio()
    wav_path = os.path.join(OUT_DIR, "case_order3.wav")
    write_wav(wav_path, audio)

    data = wav_bytes(audio)
    b64 = base64.b64encode(data).decode("ascii")

    ratios = [{1: energy_ratio_for_revs(audio[pulses[i]:pulses[i + 1]], 1, ORDER),
               2: energy_ratio_for_revs(audio[pulses[i]:pulses[i + 1]], 2, ORDER)}
              for i in range(len(lengths))]

    strict = solve_bounds(lengths=lengths, ratios=ratios,
                          total_revolutions=sum(TRUTH),
                          speed_diff_limit_rpm=30.0, locks={})
    loose = solve_bounds(lengths=lengths, ratios=ratios,
                         total_revolutions=sum(TRUTH),
                         speed_diff_limit_rpm=400.0, locks={})
    assert strict.feasible and loose.feasible

    mid_threshold = round((loose.lower_score + loose.upper_score) / 2, 4)
    # 向两侧各留一点距离，避免末位误差
    mid_threshold = min(max(mid_threshold, 0.01), 0.99)

    def payload(limit: float, threshold: float) -> dict:
        return {
            "audio_wav_base64": b64,
            "pulse_samples": pulses,
            "total_revolutions": sum(TRUTH),
            "speed_diff_limit_rpm": limit,
            "order": ORDER,
            "threshold": threshold,
            "locks": [],
        }

    strict_req = payload(30.0, 0.5)
    loose_req = payload(400.0, mid_threshold)

    with open(os.path.join(OUT_DIR, "request_strict.json"), "w",
              encoding="utf-8") as fh:
        json.dump(strict_req, fh, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "request_loose.json"), "w",
              encoding="utf-8") as fh:
        json.dump(loose_req, fh, ensure_ascii=False)

    meta = {
        "wav_path": wav_path,
        "pulse_samples": pulses,
        "total_revolutions": sum(TRUTH),
        "order": ORDER,
        "truth_assignment": TRUTH,
        "strict": {
            "speed_diff_limit_rpm": 30.0,
            "threshold": 0.5,
            "lower_score": strict.lower_score,
            "upper_score": strict.upper_score,
            "assignment": strict.lower_assignment,
            "missed": strict.lower_missed,
        },
        "loose": {
            "speed_diff_limit_rpm": 400.0,
            "threshold": mid_threshold,
            "lower_score": loose.lower_score,
            "upper_score": loose.upper_score,
            "lower_assignment": loose.lower_assignment,
            "upper_assignment": loose.upper_assignment,
            "lower_missed": loose.lower_missed,
            "upper_missed": loose.upper_missed,
        },
    }
    with open(os.path.join(OUT_DIR, "fixture_meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"WAV 已生成: {wav_path} ({len(audio) / SR:.2f}s)")
    print("脉冲点:", pulses)
    print(f"严格 Δ=30rpm: 下界={strict.lower_score:.6f} "
          f"上界={strict.upper_score:.6f} 指派={strict.lower_assignment} "
          f"漏脉冲间隔={strict.lower_missed}")
    print(f"宽松 Δ=400rpm: 下界={loose.lower_score:.6f} "
          f"上界={loose.upper_score:.6f}")
    print(f"  下端指派={loose.lower_assignment} 漏={loose.lower_missed}")
    print(f"  上端指派={loose.upper_assignment} 漏={loose.upper_missed}")
    print(f"  自动阈值={mid_threshold} (两端中点)")


if __name__ == "__main__":
    main()
