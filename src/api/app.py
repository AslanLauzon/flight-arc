from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.jobs import job_store
from src.api.models import (
    MonteCarloPayload,
    RunAcceptedPayload,
    RunJobPayload,
    MonteCarloRunRequest,
    NominalRunPayload,
    NominalRunRequest,
)
from src.api.overrides import config_from_request
from src.api.serializers import build_montecarlo_payload, build_nominal_payload
from src.config import load_config
from src.mission_runner import run_nominal_mission
from src.montecarlo.runner import run_montecarlo

logger = logging.getLogger("flight_arc.api")


def _cors_origins_from_env() -> list[str]:
    raw = os.getenv("FLIGHT_ARC_CORS_ORIGINS", "*").strip()
    if not raw:
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app() -> FastAPI:
    logging.basicConfig(
        level=os.getenv("FLIGHT_ARC_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    app = FastAPI(
        title="Flight Arc API",
        version="0.1.0",
        description="Backend API for launch vehicle simulation, orbit assessment, and Monte Carlo analysis.",
    )

    cors_origins = _cors_origins_from_env()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4())[:8])
        request.state.request_id = request_id
        started_at = time.perf_counter()
        logger.info(
            "request_received request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            logger.exception(
                "request_failed request_id=%s method=%s path=%s elapsed_ms=%.1f",
                request_id,
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        response.headers["x-request-id"] = request_id
        logger.info(
            "request_completed request_id=%s method=%s path=%s status_code=%s elapsed_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "service": "flight-arc-api",
            "status": "ok",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/config")
    def get_config() -> dict:
        return load_config().model_dump()

    def _run_nominal_job(request_id: str, cfg) -> None:
        try:
            job_store.update(request_id, status="running")
            logger.info("simulation_start request_id=%s run_type=nominal", request_id)
            result = run_nominal_mission(cfg)
            payload = build_nominal_payload(result, request_id=request_id).model_dump(mode="json")
            job_store.update(request_id, status="completed", result=payload)
            logger.info(
                "simulation_complete request_id=%s run_type=nominal mission=%s status=%s final_time_s=%.1f",
                request_id,
                result.cfg.mission.name,
                "success" if result.insertion and result.insertion.success else "completed",
                result.final_state.t,
            )
        except Exception as exc:
            job_store.update(request_id, status="failed", error=str(exc))
            logger.exception("simulation_failed request_id=%s run_type=nominal", request_id)

    def _run_montecarlo_job(request_id: str, cfg) -> None:
        try:
            job_store.update(request_id, status="running")
            logger.info("simulation_start request_id=%s run_type=montecarlo", request_id)
            results = run_montecarlo(cfg)
            payload = build_montecarlo_payload(cfg, results, request_id=request_id).model_dump(mode="json")
            job_store.update(request_id, status="completed", result=payload)
            logger.info(
                "simulation_complete request_id=%s run_type=montecarlo mission=%s runs=%s",
                request_id,
                cfg.mission.name,
                len(results),
            )
        except Exception as exc:
            job_store.update(request_id, status="failed", error=str(exc))
            logger.exception("simulation_failed request_id=%s run_type=montecarlo", request_id)

    @app.post("/runs/nominal", response_model=RunAcceptedPayload, status_code=status.HTTP_202_ACCEPTED)
    def create_nominal_run(
        background_tasks: BackgroundTasks,
        http_request: Request,
        request: NominalRunRequest | None = None,
    ) -> RunAcceptedPayload:
        request_id = getattr(http_request.state, "request_id", str(uuid.uuid4())[:8])
        cfg = config_from_request(request)
        job_store.create(request_id, "nominal")
        logger.info("request_accepted request_id=%s run_type=nominal", request_id)
        background_tasks.add_task(_run_nominal_job, request_id, cfg)
        return RunAcceptedPayload(
            request_id=request_id,
            status="accepted",
            run_type="nominal",
            poll_path=f"/runs/{request_id}",
        )

    @app.post("/runs/montecarlo", response_model=RunAcceptedPayload, status_code=status.HTTP_202_ACCEPTED)
    def create_montecarlo_run(
        background_tasks: BackgroundTasks,
        http_request: Request,
        request: MonteCarloRunRequest | None = None,
    ) -> RunAcceptedPayload:
        request_id = getattr(http_request.state, "request_id", str(uuid.uuid4())[:8])
        cfg = config_from_request(request)
        job_store.create(request_id, "montecarlo")
        logger.info("request_accepted request_id=%s run_type=montecarlo", request_id)
        background_tasks.add_task(_run_montecarlo_job, request_id, cfg)
        return RunAcceptedPayload(
            request_id=request_id,
            status="accepted",
            run_type="montecarlo",
            poll_path=f"/runs/{request_id}",
        )

    @app.get("/runs/{request_id}", response_model=RunJobPayload)
    def get_run(request_id: str, response: Response) -> RunJobPayload:
        job = job_store.get(request_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if job["status"] == "accepted":
            response.status_code = status.HTTP_202_ACCEPTED
        return RunJobPayload(**job)

    return app


app = create_app()
