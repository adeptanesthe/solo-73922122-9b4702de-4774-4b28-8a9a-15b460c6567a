"""核定报告存储：线程安全 + JSON 文件持久化。

每次被接受的锁定变更都会生成一个新 revision 快照；被拒绝的更正只进入
``rejected_attempts`` 审计列表，绝不改动 current / revisions，因此原锁定
和原报告始终可查询。
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Any


class AssessmentStore:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, "assessments.json")
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        os.makedirs(data_dir, exist_ok=True)
        self._load()

    # ---------- 持久化 ----------

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self._records = payload.get("assessments", {})
        except (json.JSONDecodeError, OSError):
            # 损坏文件不应让服务起不来；保留现场文件后从空库开始。
            backup = self.path + ".corrupt"
            try:
                os.replace(self.path, backup)
            except OSError:
                pass
            self._records = {}

    def _persist_locked(self) -> None:
        fd, tmp_path = tempfile.mkstemp(
            prefix=".assessments-", suffix=".json", dir=self.data_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"assessments": self._records}, fh,
                          ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ---------- CRUD ----------

    def create(self, report: dict[str, Any]) -> None:
        aid = report["assessment_id"]
        with self._lock:
            self._records[aid] = {
                "current": report,
                "revisions": [report],
                "rejected_attempts": [],
            }
            self._persist_locked()

    def get(self, assessment_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(assessment_id)
            return json.loads(json.dumps(record)) if record else None

    def get_current_report(self, assessment_id: str) -> dict[str, Any] | None:
        record = self.get(assessment_id)
        return record["current"] if record else None

    def list_summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            reports = [r["current"] for r in self._records.values()]
        reports.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return [{
            "assessment_id": r["assessment_id"],
            "created_at": r.get("created_at"),
            "updated_at": r.get("updated_at"),
            "revision": r.get("revision"),
            "order": r.get("order"),
            "threshold": r.get("threshold"),
            "status": r.get("status"),
            "status_label_zh": r.get("status_label_zh"),
            "score_lower": r.get("score_lower"),
            "score_upper": r.get("score_upper"),
            "locks": r.get("locks"),
        } for r in reports]

    def commit_revision(self, report: dict[str, Any]) -> None:
        """接受一次锁定更正：更新 current 并追加 revision 快照。"""
        aid = report["assessment_id"]
        with self._lock:
            record = self._records.get(aid)
            if record is None:
                raise KeyError(aid)
            record["current"] = report
            revisions = record.setdefault("revisions", [])
            # 同 revision 覆盖（理论上不会），否则追加
            if revisions and revisions[-1].get("revision") == report["revision"]:
                revisions[-1] = report
            else:
                revisions.append(report)
            self._persist_locked()

    def add_rejected_attempt(self, assessment_id: str,
                             attempt: dict[str, Any]) -> None:
        with self._lock:
            record = self._records.get(assessment_id)
            if record is None:
                raise KeyError(assessment_id)
            attempts = record.setdefault("rejected_attempts", [])
            attempts.append(attempt)
            # 审计列表只保留最近 50 条，避免无限增长
            del attempts[:-50]
            self._persist_locked()

    def list_revisions(self, assessment_id: str) -> list[dict[str, Any]] | None:
        record = self.get(assessment_id)
        if record is None:
            return None
        return [{
            "revision": r.get("revision"),
            "updated_at": r.get("updated_at"),
            "locks": r.get("locks"),
            "score_lower": r.get("score_lower"),
            "score_upper": r.get("score_upper"),
            "status": r.get("status"),
        } for r in record["revisions"]]

    def get_revision(self, assessment_id: str,
                     revision: int) -> dict[str, Any] | None:
        record = self.get(assessment_id)
        if record is None:
            return None
        for report in record["revisions"]:
            if report.get("revision") == revision:
                return report
        return None

    def list_rejected_attempts(self, assessment_id: str):
        record = self.get(assessment_id)
        if record is None:
            return None
        return record.get("rejected_attempts", [])
