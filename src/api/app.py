from __future__ import annotations

from fastapi import FastAPI

from src.api.models import MonteCarloPayload, NominalRunPayload
from src.api.serializers import build_montecarlo_payload, build_nominal_payload
from src.config import load_config
from src.mission_runner import run_nominal_mission
from src.montecarlo.runner import run_montecarlo

app = FastAPI(title="Flight Arc API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def get_config() -> dict:
    return load_config().model_dump()


@app.post("/runs/nominal", response_model=NominalRunPayload)
def create_nominal_run(include_plot_image: bool = True) -> NominalRunPayload:
    result = run_nominal_mission()
    return build_nominal_payload(result, include_plot_image=include_plot_image)


@app.post("/runs/montecarlo", response_model=MonteCarloPayload)
def create_montecarlo_run() -> MonteCarloPayload:
    cfg = load_config()
    results = run_montecarlo(cfg)
    return build_montecarlo_payload(cfg, results)
