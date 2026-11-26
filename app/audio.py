"""PCM16 WAV 读取与基于阶次的 FFT 能量占比计算。

约定（与核定口径一致）：

* 输入必须是 ≤10 s、16 kHz、单声道、PCM16 WAV；
* 两个脉冲点之间的一个间隔视为一个“段”，若该间隔指派 r 转
  （r ∈ {1,2}），则用线性插值在该段内均匀重采样到 64·r 点
  （每转 64 点）；
* 去均值后做 FFT。长度 N=64r、目标阶次 h 对应 FFT 第 k=h·r 根谱线
  （k/r 的单位是“周/转”，即阶次）；
* 能量占比 = (P[k] + P[N-k]) / ΣP，同时计入共轭负频率谱线；
* 全段零能量时占比记 0。
"""

from __future__ import annotations

import io
import wave

import numpy as np

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes = PCM16
MAX_DURATION_SECONDS = 10.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION_SECONDS)  # 160000
POINTS_PER_REVOLUTION = 64


class WavValidationError(ValueError):
    """WAV 不符合输入约定。带稳定错误码。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def read_pcm16_wav(data: bytes) -> np.ndarray:
    """读取并校验 WAV，返回 float64 单声道样本，幅度归一化到 [-1, 1]。"""
    if not data:
        raise WavValidationError("audio_empty", "未收到 WAV 内容。")
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            if wf.getcomptype() != "NONE":
                raise WavValidationError(
                    "wav_not_pcm",
                    f"仅支持未压缩 PCM WAV，当前压缩类型为 {wf.getcompname()!r}。")
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except WavValidationError:
        raise
    except (wave.Error, EOFError, Exception) as exc:  # noqa: BLE001
        raise WavValidationError(
            "wav_unreadable", f"WAV 解析失败：{exc}") from exc

    if n_channels != CHANNELS:
        raise WavValidationError(
            "wav_not_mono",
            f"仅支持单声道 WAV，当前为 {n_channels} 声道。")
    if sample_width != SAMPLE_WIDTH:
        raise WavValidationError(
            "wav_not_pcm16",
            f"仅支持 PCM16（16-bit）WAV，当前采样位宽为 "
            f"{8 * sample_width}-bit。")
    if frame_rate != SAMPLE_RATE:
        raise WavValidationError(
            "wav_bad_sample_rate",
            f"仅支持 {SAMPLE_RATE} Hz 采样率，当前为 {frame_rate} Hz。")

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if samples.size > MAX_SAMPLES:
        raise WavValidationError(
            "wav_too_long",
            f"音频至多 {MAX_DURATION_SECONDS:g} 秒（{MAX_SAMPLES} 个采样点），"
            f"实际约 {samples.size / SAMPLE_RATE:.2f} 秒"
            f"（{samples.size} 点）。")
    if samples.size == 0:
        raise WavValidationError("audio_empty", "WAV 中没有音频采样帧。")
    return samples


def energy_ratio_for_revs(segment: np.ndarray,
                          revolutions: int,
                          order: int) -> float:
    """单个间隔段在指定转数假设下的目标阶次能量占比。

    零能量（如数字静音）返回 0.0。
    """
    if revolutions not in (1, 2):
        raise ValueError("revolutions 只能是 1 或 2")
    n_out = POINTS_PER_REVOLUTION * revolutions
    d = int(segment.shape[0])
    if d <= 0:
        return 0.0

    # 目标采样位置取 [0, d) 半开区间，保证整周期信号恰好落在整数 FFT 谱线上。
    source_pos = np.arange(d, dtype=np.float64)
    target_pos = np.arange(n_out, dtype=np.float64) * (d / n_out)
    resampled = np.interp(target_pos, source_pos, segment).astype(np.float64)

    resampled -= resampled.mean()
    spectrum = np.fft.fft(resampled)
    power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
    total_energy = float(power.sum())
    if total_energy <= 0.0:
        return 0.0

    target_bin = order * revolutions  # = order / (1/revs)，见模块说明
    target_energy = float(power[target_bin] + power[n_out - target_bin])
    ratio = target_energy / total_energy
    # 浮点误差可能产生 ±1e-16 级越界，夹回数学区间 [0, 1]。
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        return 1.0
    return ratio
