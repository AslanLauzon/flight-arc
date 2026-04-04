from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def create(self, request_id: str, run_type: str) -> dict[str, Any]:
        job = {
            "request_id": request_id,
            "run_type": run_type,
            "status": "accepted",
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[request_id] = job
        return deepcopy(job)

    def update(self, request_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            job = self._jobs[request_id]
            job.update(updates)
            return deepcopy(job)

    def get(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(request_id)
            return deepcopy(job) if job is not None else None


job_store = JobStore()
