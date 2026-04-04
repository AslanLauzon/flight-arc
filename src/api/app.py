from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    MonteCarloPayload,
    MonteCarloRunRequest,
    NominalRunPayload,
    NominalRunRequest,
)
from src.api.overrides import config_from_request
from src.api.serializers import build_montecarlo_payload, build_nominal_payload
from src.config import load_config
from src.mission_runner import run_nominal_mission
from src.montecarlo.runner import run_montecarlo


def _cors_origins_from_env() -> list[str]:
    raw = os.getenv("FLIGHT_ARC_CORS_ORIGINS", "*").strip()
    if not raw:
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app() -> FastAPI:
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

    @app.post("/runs/nominal", response_model=NominalRunPayload)
    def create_nominal_run(request: NominalRunRequest | None = None) -> NominalRunPayload:
        cfg = config_from_request(request)
        result = run_nominal_mission(cfg)
        return build_nominal_payload(result)

    @app.post("/runs/montecarlo", response_model=MonteCarloPayload)
    def create_montecarlo_run(request: MonteCarloRunRequest | None = None) -> MonteCarloPayload:
        cfg = config_from_request(request)
        results = run_montecarlo(cfg)
        return build_montecarlo_payload(cfg, results)

    return app


app = create_app()
