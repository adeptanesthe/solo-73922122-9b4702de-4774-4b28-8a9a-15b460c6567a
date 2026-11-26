"""关键 API 请求自测：核定创建、三态判定、锁定更正/拒绝/解除/查询。"""

from __future__ import annotations

import base64
import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.service import verdict_for

SR = 16_000


def make_wav(samples: np.ndarray, *, channels: int = 1,
             sample_rate: int = SR, sample_width: int = 2) -> bytes:
    if sample_width == 2:
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
        raw = pcm.tobytes()
    else:
        raw = samples.astype("u1").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)
    return buf.getvalue()


def tone_fixture() -> tuple[bytes, list[int]]:
    """与 examples/make_fixtures 同构：6 间隔，真实 [1,2,1,1,2,1]。"""
    truth = [1, 2, 1, 1, 2, 1]
    s_per_rev = 4000
    lengths = [r * s_per_rev for r in truth]
    pulses = [0]
    for d in lengths:
        pulses.append(pulses[-1] + d)
    rng = np.random.default_rng(20261126)
    chunks = []
    for i, (rt, d) in enumerate(zip(truth, lengths)):
        n = np.arange(d)
        sig = 0.22 * np.sin(2 * np.pi * rt * n / d)
        amp = 0.9 if i in (1, 4) else 0.10
        sig += amp * np.sin(2 * np.pi * 3 * rt * n / d + 0.7)
        sig += 0.02 * rng.standard_normal(d)
        sig /= max(1.0, np.max(np.abs(sig)) / 0.95)
        chunks.append(sig)
    return make_wav(np.concatenate(chunks)), pulses


@pytest.fixture()
def client(tmp_path):
    app = create_app(data_dir=str(tmp_path / "data"))
    return TestClient(app)


def _multipart_kwargs(wav: bytes, pulses, **overrides):
    data = {
        "pulse_samples": json.dumps(pulses),
        "total_revolutions": "8",
        "speed_diff_limit_rpm": "400",
        "order": "3",
        "threshold": "0.3",
    }
    data.update({k: str(v) for k, v in overrides.items()})
    return {
        "files": {"audio": ("case.wav", wav, "audio/wav")},
        "data": data,
    }


# ---------- 基础 ----------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert client.get("/api/v1/assessments").status_code == 200


def test_silence_multipart_below_standard(client):
    wav = make_wav(np.zeros(32000, dtype=np.float64))
    pulses = [0, 4000, 8000, 12000, 16000]  # 4 个等长间隔, R=4
    r = client.post("/api/v1/assessments",
                    **_multipart_kwargs(
                        wav, pulses,
                        total_revolutions="4",
                        speed_diff_limit_rpm="0",
                        threshold="0.2"))
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["interval_count"] == 4
    assert body["score_lower"] == 0.0
    assert body["score_upper"] == 0.0
    assert body["status"] == "below_standard"
    assert body["status_label_zh"] == "未达标"
    assert body["lower_endpoint"]["missed_pulse_intervals"] == []
    assert body["algorithm"]["enumerates_assignments"] is False
    # 已落库可查
    aid = body["assessment_id"]
    assert client.get(f"/api/v1/assessments/{aid}").status_code == 200


def test_json_base64_endpoint_and_aliases(client):
    wav, pulses = tone_fixture()
    payload = {
        "audio_wav_base64": base64.b64encode(wav).decode(),
        "pulse_samples": pulses,
        "total_revolutions": 8,
        "speed_diff_limit_rpm": 30,
        "order": 3,
        "threshold": 0.5,
    }
    r = client.post("/api/v1/assessments/json", json=payload)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "recommend_investigate"
    assert body["lower_endpoint"]["assignment_revolutions"] == [1, 2, 1, 1, 2, 1]
    assert body["lower_endpoint"]["missed_pulse_intervals"] == [1, 4]

    # 统一入口 + 中文字段别名也应成功
    alias_payload = {
        "audio_base64": payload["audio_wav_base64"],
        "脉冲采样点": pulses,
        "总转数": 8,
        "转速差上限": 30,
        "阶次": 3,
        "阈值": 0.5,
    }
    r2 = client.post("/api/v1/assessments", json=alias_payload)
    assert r2.status_code == 201, r2.text
    assert r2.json()["status"] == "recommend_investigate"

    # /api 无前缀版本同样可用
    r3 = client.post("/api/assessments/json", json=payload)
    assert r3.status_code == 201


def test_pending_verification_spread(client):
    wav, pulses = tone_fixture()
    # 先取宽松约束下的真实上下界，阈值取中点，稳定构造“待核”
    r = client.post("/api/v1/assessments/json", json={
        "audio_wav_base64": base64.b64encode(wav).decode(),
        "pulse_samples": pulses, "total_revolutions": 8,
        "speed_diff_limit_rpm": 400, "order": 3, "threshold": 0.0})
    assert r.status_code == 201
    b = r.json()
    lo, hi = b["score_lower"], b["score_upper"]
    assert hi - lo > 0.1
    mid = (lo + hi) / 2
    r2 = client.post("/api/v1/assessments/json", json={
        "audio_wav_base64": base64.b64encode(wav).decode(),
        "pulse_samples": pulses, "total_revolutions": 8,
        "speed_diff_limit_rpm": 400, "order": 3, "threshold": mid})
    assert r2.json()["status"] == "pending_verification"
    assert r2.json()["status_label_zh"] == "待核"
    assert r2.json()["verdict"]["lower_reaches_threshold"] is False
    assert r2.json()["verdict"]["upper_reaches_threshold"] is True


def test_verdict_three_ways():
    assert verdict_for(0.6, 0.8, 0.5)["status"] == "recommend_investigate"
    assert verdict_for(0.4, 0.8, 0.5)["status"] == "pending_verification"
    assert verdict_for(0.1, 0.2, 0.5)["status"] == "below_standard"
    # 边界相等算达标
    assert verdict_for(0.5, 0.5, 0.5)["status"] == "recommend_investigate"


# ---------- 输入校验 ----------

def test_order_and_threshold_validation(client):
    wav = make_wav(np.zeros(8000))
    r = client.post("/api/v1/assessments",
                    **_multipart_kwargs(wav, [0, 4000], order="9"))
    assert r.status_code == 422
    r = client.post("/api/v1/assessments",
                    **_multipart_kwargs(wav, [0, 4000], threshold="1.5"))
    assert r.status_code == 422


def test_bad_wav_formats(client):
    cases = [
        (b"not a wav file at all", "wav_unreadable"),
        (make_wav(np.zeros(16000), channels=2), "wav_not_mono"),
        (make_wav(np.zeros(8000), sample_rate=8000), "wav_bad_sample_rate"),
        (make_wav(np.zeros(160001)), "wav_too_long"),
    ]
    for data, code in cases:
        r = client.post(
            "/api/v1/assessments",
            files={"audio": ("x.wav", data, "audio/wav")},
            data={"pulse_samples": "[0,100]", "total_revolutions": "1",
                  "speed_diff_limit_rpm": "100", "order": "1",
                  "threshold": "0.2"})
        assert r.status_code == 422, (code, r.status_code, r.text)
        assert r.json()["code"] == code


def test_pulse_validation(client):
    wav = make_wav(np.zeros(40000))
    base = dict(total_revolutions="4", speed_diff_limit_rpm="100",
                order="2", threshold="0.2")

    r = client.post("/api/v1/assessments", files={"audio": ("a.wav", wav)},
                    data={**base, "pulse_samples": "[0]"})
    assert r.status_code == 422 and r.json()["code"] == "not_enough_pulses"

    many = list(range(0, 202, 2))  # 101 个点 -> 100 间隔，合法边界
    r = client.post("/api/v1/assessments", files={"audio": ("a.wav", wav)},
                    data={**base, "pulse_samples": json.dumps(many)})
    # 间隔长度只有 2，且总转数 4 远小于 100 -> 无解（不是“超 100”错误）
    assert r.status_code == 422

    too_many = list(range(0, 204, 2))  # 102 点 -> 101 间隔
    r = client.post("/api/v1/assessments", files={"audio": ("a.wav", wav)},
                    data={**base, "pulse_samples": json.dumps(too_many)})
    assert r.status_code == 422 and r.json()["code"] == "too_many_intervals"

    r = client.post("/api/v1/assessments", files={"audio": ("a.wav", wav)},
                    data={**base, "pulse_samples": "[0, 5, 5]"})
    assert r.json()["code"] == "pulses_not_strictly_increasing"

    r = client.post("/api/v1/assessments", files={"audio": ("a.wav", wav)},
                    data={**base, "pulse_samples": "[0, 999999]"})
    assert r.json()["code"] == "pulse_out_of_range"


def test_infeasible_total_revolutions_on_create(client):
    wav = make_wav(np.zeros(20000))
    pulses = [0, 4000, 8000, 12000]  # 3 间隔，最多 6 转
    r = client.post("/api/v1/assessments",
                    **_multipart_kwargs(
                        wav, pulses, total_revolutions="40"))
    assert r.status_code == 422
    assert r.json()["code"] == "incompatible_total_revolutions_or_locks"


# ---------- 锁定更正事务 ----------

def _create_loose(client) -> str:
    wav, pulses = tone_fixture()
    r = client.post("/api/v1/assessments", json={
        "audio_wav_base64": base64.b64encode(wav).decode(),
        "pulse_samples": pulses, "total_revolutions": 8,
        "speed_diff_limit_rpm": 400, "order": 3, "threshold": 0.3})
    assert r.status_code == 201
    return r.json()["assessment_id"]


def test_lock_change_collapses_bounds_and_revisions(client):
    aid = _create_loose(client)
    original = client.get(f"/api/v1/assessments/{aid}").json()
    upper = original["upper_endpoint"]

    # 把所有间隔锁定为上界指派 -> 上下界坍缩为该上界
    locks = [{"interval": i, "revolutions": r}
             for i, r in enumerate(upper["assignment_revolutions"])]
    r = client.put(f"/api/v1/assessments/{aid}/locks", json={"locks": locks})
    assert r.status_code == 200, r.text
    changed = r.json()
    assert changed["revision"] == 2
    assert changed["score_lower"] == pytest.approx(upper["score"], abs=1e-12)
    assert changed["score_upper"] == pytest.approx(upper["score"], abs=1e-12)
    assert changed["lower_endpoint"]["missed_pulse_intervals"] == \
        upper["missed_pulse_intervals"]

    revs = client.get(f"/api/v1/assessments/{aid}/revisions").json()["revisions"]
    assert [x["revision"] for x in revs] == [1, 2]
    rev1 = client.get(f"/api/v1/assessments/{aid}/revisions/1").json()
    assert rev1["revision"] == 1
    assert rev1["locks"] == []


def test_rejected_lock_update_keeps_original_report(client):
    aid = _create_loose(client)
    before = client.get(f"/api/v1/assessments/{aid}").json()

    # 6 个间隔全部锁 2 转 -> 至少 12 转，但总转数为 8 -> 无解
    impossible = [{"interval": i, "revolutions": 2} for i in range(6)]
    r = client.put(f"/api/v1/assessments/{aid}/locks",
                   json={"locks": impossible})
    assert r.status_code == 409
    body = r.json()
    assert body["code"] == "incompatible_total_revolutions_or_locks"
    assert body["detail"]["rejected"] is True
    assert body["detail"]["current_revision"] == before["revision"]

    after = client.get(f"/api/v1/assessments/{aid}").json()
    assert after["revision"] == before["revision"] == 1
    assert after["lock_map"] == before["lock_map"]
    assert after["score_lower"] == before["score_lower"]
    assert after["score_upper"] == before["score_upper"]
    assert after["assessment_id"] == aid

    audit = client.get(f"/api/v1/assessments/{aid}/audit").json()
    assert len(audit["rejected_attempts"]) == 1
    assert audit["rejected_attempts"][0]["error"]["code"] == \
        "incompatible_total_revolutions_or_locks"

    # POST 别名同样可更正
    ok_lock = [{"interval": 1, "missed_pulse": True},
               {"interval": 4, "missed_pulse": True}]
    r = client.post(f"/api/v1/assessments/{aid}/locks",
                    json={"locks": ok_lock})
    assert r.status_code == 200
    assert r.json()["revision"] == 2
    assert set(r.json()["lock_map"].keys()) == {"1", "4"}

    # DELETE 解除全部锁定 -> 恢复原始宽上下界，revision 再 +1
    r = client.delete(f"/api/v1/assessments/{aid}/locks")
    assert r.status_code == 200
    released = r.json()
    assert released["revision"] == 3
    assert released["locks"] == []
    assert released["score_lower"] == pytest.approx(before["score_lower"])
    assert released["score_upper"] == pytest.approx(before["score_upper"])


def test_lock_alias_formats_and_404(client):
    aid = _create_loose(client)
    # 对象映射 + 纯序号数组（锁 2 转）
    r = client.patch(f"/api/v1/assessments/{aid}",
                     json={"locks": {"1": 2, "4": 2}})
    assert r.status_code == 200
    assert r.json()["lock_map"] == {"1": 2, "4": 2}

    r = client.put(f"/api/v1/assessments/does_not_exist/locks",
                   json={"release_all": True})
    assert r.status_code == 404
    assert client.get("/api/v1/assessments/nope").status_code == 404


def test_unsupported_media_type(client):
    r = client.post("/api/v1/assessments", content="hello",
                    headers={"content-type": "text/plain"})
    assert r.status_code == 415


def test_typed_upload_endpoint(client):
    wav, pulses = tone_fixture()
    r = client.post(
        "/api/v1/assessments/upload",
        files={"audio": ("case.wav", wav, "audio/wav")},
        data={"pulse_samples": json.dumps(pulses),
              "total_revolutions": "8",
              "speed_diff_limit_rpm": "30",
              "order": "3", "threshold": "0.5",
              "locks": json.dumps([{"interval": 1, "revolutions": 2},
                                   {"interval": 4, "revolutions": 2}])})
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "recommend_investigate"


def test_openapi_schema_available(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/api/v1/assessments" in paths
    assert "/api/v1/assessments/{assessment_id}/locks" in paths
    assert client.get("/docs").status_code == 200


def test_dataurl_percent_rps_and_comma_pulses(client):
    wav, pulses = tone_fixture()
    b64 = base64.b64encode(wav).decode()
    data_url = "data:audio/wav;base64," + b64
    # threshold_percent=50 -> 0.5；rps 单位 0.5 rps = 30 rpm（唯一还原真值）
    r = client.post("/api/v1/assessments", json={
        "audio_wav_base64": data_url,
        "pulse_samples": pulses,
        "total_revolutions": 8,
        "speed_diff_limit_rpm": 0.5,
        "speed_diff_unit": "rps",
        "order": 3,
        "threshold_percent": 50})
    assert r.status_code == 201, r.text
    d = r.json()
    assert d["speed_diff_limit_rpm"] == pytest.approx(30.0)
    assert d["threshold"] == pytest.approx(0.5)
    assert d["status"] == "recommend_investigate"

    # 表单里的逗号分隔脉冲点 + JSON 字符串锁定
    r = client.post(
        "/api/v1/assessments",
        files={"audio": ("c.wav", wav, "audio/wav")},
        data={"pulse_samples": ",".join(map(str, pulses)),
              "total_revolutions": "8", "speed_diff_limit_rpm": "30",
              "order": "3", "threshold": "0.5",
              "locks": '[{"interval":1,"revolutions":2}]'})
    assert r.status_code == 201, r.text
    assert r.json()["lock_map"] == {"1": 2}
