"""FastAPI 入口：核定创建（JSON / multipart）、查询、锁定更正。

路由同时挂在 ``/api`` 与 ``/api/v1`` 前缀下，核心资源为 ``/assessments``。
启动：``uvicorn app.main:app --host 0.0.0.0 --port 8000``。
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from . import __version__
from .config import DATA_DIR
from .payload import (AUDIO_FILE_PARTS, AssessmentJsonRequest,
                      LockUpdateRequest, PayloadError, coerce_jsonish,
                      decode_audio_b64, normalize_request, parse_locks)
from .service import AssessmentService, ServiceError
from .store import AssessmentStore


def create_app(data_dir: str | None = None) -> FastAPI:
    app = FastAPI(
        title="试车检修异响阶次核定 API",
        description=(
            "判断异响是否值得按指定阶次排查。接收 ≤10 秒 16kHz 单声道 "
            "PCM16 WAV、脉冲采样点、控制器总转数、相邻间隔平均转速差上限、"
            "待查阶次(1~8)与判定阈值；用每转 64 点重采样 + FFT 求各间隔"
            "目标阶次能量占比，并以动态规划求可行转法下的得分上下界及两端"
            "对应的漏脉冲间隔。支持人工锁定与锁定更正审计。"),
        version=__version__,
    )

    store = AssessmentStore(data_dir or DATA_DIR)
    service = AssessmentService(store)
    app.state.store = store
    app.state.service = service

    # ---------- 统一错误体 ----------

    @app.exception_handler(ServiceError)
    async def service_error_handler(_: Request, exc: ServiceError):
        detail = {"code": exc.code, "message": exc.message, **exc.details}
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message,
                     "detail": detail})

    @app.exception_handler(PayloadError)
    async def payload_error_handler(_: Request, exc: PayloadError):
        return JSONResponse(
            status_code=422,
            content={"code": exc.code, "message": exc.message,
                     "detail": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"code": "request_validation_error",
                     "message": "请求参数校验失败。",
                     "detail": jsonable_encoder(exc.errors())})

    # ---------- 健康检查 / 根信息 ----------

    @app.get("/health", tags=["meta"])
    async def health():
        return {"status": "ok", "service": "order-audit-api",
                "version": __version__}

    @app.get("/", tags=["meta"])
    async def root():
        return {
            "service": "order-audit-api",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
            "endpoints": [
                "POST /api/v1/assessments        (JSON 或 multipart 自适应)",
                "POST /api/v1/assessments/json   (JSON + base64 WAV)",
                "POST /api/v1/assessments/upload (multipart WAV 上传)",
                "GET  /api/v1/assessments",
                "GET  /api/v1/assessments/{id}",
                "PUT  /api/v1/assessments/{id}/locks",
                "DELETE /api/v1/assessments/{id}/locks",
                "GET  /api/v1/assessments/{id}/revisions",
            ],
        }

    # 路由逻辑放在闭包内，再分别挂 /api/v1 与 /api 前缀
    from fastapi import APIRouter
    router = APIRouter(tags=["assessments"])

    def _is_upload(value: Any) -> bool:
        # 兼容不同 FastAPI/Starlette 版本中两个 UploadFile 类不相同的情况
        return isinstance(value, StarletteUploadFile) or (
            hasattr(value, "read") and hasattr(value, "filename")
            and callable(getattr(value, "read", None)))

    def _create(audio_bytes: bytes | None, raw: dict[str, Any]) -> dict[str, Any]:
        canonical = normalize_request(raw)
        if not audio_bytes:
            b64 = raw.get("audio_wav_base64")
            if b64 is None:
                for key in ("wav_base64", "audio_base64"):
                    if raw.get(key):
                        b64 = raw[key]
                        break
            if not b64:
                raise PayloadError(
                    "missing_audio",
                    "缺少 WAV：multipart 请上传 audio 文件，"
                    "JSON 请提供 audio_wav_base64。")
            audio_bytes = decode_audio_b64(b64)
        return service.create_assessment(wav_bytes=audio_bytes, **canonical)

    # ----- 统一入口：按 Content-Type 自适应 -----

    @router.post("/assessments", status_code=201)
    async def create_assessment_auto(request: Request):
        content_type = request.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            try:
                body = await request.json()
            except (json.JSONDecodeError, ValueError) as exc:
                raise PayloadError("invalid_json",
                                   f"请求体不是合法 JSON：{exc}") from exc
            if not isinstance(body, dict):
                raise PayloadError("invalid_json", "JSON 顶层必须是对象。")
            return _create(None, body)

        if ("multipart/form-data" in content_type
                or "application/x-www-form-urlencoded" in content_type):
            form = await request.form()
            raw: dict[str, Any] = {}
            audio_bytes: bytes | None = None
            for key, value in form.multi_items():
                if _is_upload(value):
                    if audio_bytes is None and (
                            key in AUDIO_FILE_PARTS
                            or (getattr(value, "content_type", "") or "").startswith("audio")
                            or (getattr(value, "filename", "") or "").lower().endswith(".wav")):
                        audio_bytes = await value.read()
                    continue
                # 同名字段多次出现时保留为列表
                if key in raw:
                    existing = raw[key]
                    if isinstance(existing, list):
                        existing.append(value)
                    else:
                        raw[key] = [existing, value]
                else:
                    raw[key] = value
            if audio_bytes is None:
                for part in AUDIO_FILE_PARTS:
                    upload = form.get(part)
                    if _is_upload(upload):
                        audio_bytes = await upload.read()
                        break
            return _create(audio_bytes, raw)

        raise ServiceError(
            415, "unsupported_media_type",
            "请使用 application/json（audio_wav_base64）或 "
            "multipart/form-data（audio 文件）提交。")

    # ----- 带 OpenAPI schema 的 JSON 入口 -----

    @router.post("/assessments/json", status_code=201)
    async def create_assessment_json(payload: AssessmentJsonRequest):
        raw = payload.model_dump()
        return _create(None, raw)

    # ----- 带 OpenAPI schema 的 multipart 入口 -----

    @router.post("/assessments/upload", status_code=201)
    async def create_assessment_upload(
            audio: UploadFile = File(..., description="16kHz/单声道/PCM16 WAV"),
            pulse_samples: str = Form(..., description="脉冲采样点 JSON 数组"),
            total_revolutions: int = Form(...),
            speed_diff_limit_rpm: float = Form(...),
            order: int = Form(..., ge=1, le=8),
            threshold: float = Form(..., ge=0, le=1),
            locks: str | None = Form(None),
            speed_diff_unit: str | None = Form("rpm")):
        audio_bytes = await audio.read()
        raw = {
            "pulse_samples": coerce_jsonish(pulse_samples, "pulse_samples"),
            "total_revolutions": total_revolutions,
            "speed_diff_limit_rpm": speed_diff_limit_rpm,
            "order": order,
            "threshold": threshold,
            "speed_diff_unit": speed_diff_unit,
        }
        if locks:
            raw["locks"] = coerce_jsonish(locks, "locks")
        return _create(audio_bytes, raw)

    # ----- 查询 -----

    @router.get("/assessments")
    async def list_assessments():
        items = store.list_summaries()
        return {"assessments": items, "count": len(items)}

    @router.get("/assessments/{assessment_id}")
    async def get_assessment(assessment_id: str):
        return service.get_report(assessment_id)

    @router.get("/assessments/{assessment_id}/revisions")
    async def get_revisions(assessment_id: str):
        revisions = store.list_revisions(assessment_id)
        if revisions is None:
            raise ServiceError(404, "assessment_not_found",
                               f"找不到核定报告 {assessment_id}。")
        return {"assessment_id": assessment_id, "revisions": revisions}

    @router.get("/assessments/{assessment_id}/revisions/{revision}")
    async def get_revision(assessment_id: str, revision: int):
        report = store.get_revision(assessment_id, revision)
        if report is None:
            raise ServiceError(
                404, "revision_not_found",
                f"报告 {assessment_id} 不存在 revision {revision}。")
        return report

    @router.get("/assessments/{assessment_id}/audit")
    async def get_audit(assessment_id: str):
        attempts = store.list_rejected_attempts(assessment_id)
        if attempts is None:
            raise ServiceError(404, "assessment_not_found",
                               f"找不到核定报告 {assessment_id}。")
        return {"assessment_id": assessment_id,
                "rejected_attempts": attempts}

    # ----- 锁定更正 / 解除（多种方法别名，便于对接） -----

    async def _apply_lock_update(assessment_id: str,
                                 request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise PayloadError("invalid_json",
                               f"锁定更正请求体必须是 JSON：{exc}") from exc
        if not isinstance(body, dict):
            raise PayloadError("invalid_json", "JSON 顶层必须是对象。")

        if body.get("release_all") is True:
            locks: dict[int, int] = {}
        elif "locks" in body:
            locks = parse_locks(body["locks"])
        elif all(str(k).isdigit() for k in body.keys()):
            # 兼容直接提交 {"3": 2, "7": 1}
            locks = parse_locks(body)
        else:
            raise PayloadError(
                "missing_locks",
                "请提供 locks 全量锁定表，或设置 release_all=true。")
        report = service.change_locks(assessment_id, locks)
        return JSONResponse(status_code=200, content=report)

    @router.put("/assessments/{assessment_id}/locks")
    async def put_locks(assessment_id: str, request: Request):
        return await _apply_lock_update(assessment_id, request)

    @router.post("/assessments/{assessment_id}/locks")
    async def post_locks(assessment_id: str, request: Request):
        return await _apply_lock_update(assessment_id, request)

    @router.patch("/assessments/{assessment_id}")
    async def patch_assessment(assessment_id: str, request: Request):
        # 仅支持用 PATCH 修改 locks
        return await _apply_lock_update(assessment_id, request)

    @router.delete("/assessments/{assessment_id}/locks")
    async def delete_locks(assessment_id: str):
        return service.release_locks(assessment_id)

    @router.post("/assessments/{assessment_id}/locks/release")
    async def release_locks_alias(assessment_id: str):
        return service.release_locks(assessment_id)

    # 同时暴露根路径、/api、/api/v1 三种前缀，方便不同对接约定
    app.include_router(router, prefix="")
    app.include_router(router, prefix="/api/v1")
    app.include_router(router, prefix="/api")
    return app


app = create_app()
