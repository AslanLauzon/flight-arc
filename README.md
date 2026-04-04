# Flight Arc

Flight Arc is a Python launch vehicle simulation toolkit for modeling nominal ascent, orbital insertion, and mission dispersion. It combines a staged vehicle model, 3DOF trajectory propagation, configurable guidance, event sequencing, orbit assessment, plotting, and JSON export for backend or UI use.

## Capabilities

- Simulates multi-stage ascent with thrust, drag, gravity, staging, fairing deploy, and SECO events.
- Supports pitch program, gravity turn, and PEG-style orbital insertion guidance.
- Estimates achieved orbit and evaluates insertion against mission targets.
- Runs Monte Carlo campaigns for dispersion and success-rate analysis.
- Produces terminal summaries, plots, and structured JSON outputs for frontend integration.
- Includes a FastAPI outline for exposing simulation runs through an API.

## Project Layout

```text
config/      YAML mission, vehicle, simulation, and uncertainty inputs
scripts/     runnable entry points for nominal, Monte Carlo, and config validation
src/         simulation, guidance, event, reporting, and API code
tests/       unit and integration coverage
outputs/     generated JSON and report artifacts
```

## Quick Start

```bash
uv sync
uv run python -m scripts.validate_config
uv run python -m scripts.run_nominal
uv run python -m scripts.run_montecarlo
```

## API

Run the backend locally with:

```bash
uv sync --extra api
uv run uvicorn src.api.app:app --reload
```

Available endpoints:

- `GET /`
- `GET /health`
- `GET /config`
- `POST /runs/nominal`
- `POST /runs/montecarlo`

The API returns simulation results as structured JSON with event data, orbit results, and plot series for frontend rendering.

Both run endpoints can accept config overrides in the request body. The backend starts from the default YAML config, then merges any provided `mission`, `vehicle`, `simulation`, or `uncertainties` fields before validating and running the sim.

Example nominal request:

```json
{
  "mission": {
    "name": "DEMO-1",
    "launch_site": {
      "latitude_deg": 28.5,
      "longitude_deg": -80.6,
      "altitude_m": 3.0,
      "azimuth_deg": 90.0
    },
    "target_orbit": {
      "apogee_km": 190.0,
      "perigee_km": 160.0,
      "inclination_deg": 28.5
    }
  },
  "vehicle": {
    "stages": [
      {
        "id": 1,
        "name": "First Stage",
        "propellant_mass_kg": 18000.0,
        "dry_mass_kg": 2000.0,
        "thrust_vac_N": 500000.0,
        "isp_vac_s": 290.0,
        "isp_sl_s": 255.0,
        "burn_time_s": 145.0
      }
    ],
    "payload": {
      "mass_kg": 300.0,
      "fairing_mass_kg": 120.0,
      "fairing_jettisoned": true
    },
    "separation": {
      "spring_impulse_Ns": 500.0,
      "tip_off_rate_deg_s": 0.5
    }
  }
}
```

Stage overrides are merged by `id`, so you can update stage 1 without resending every stage field from the default config.
If a vehicle deploys the fairing but keeps that mass attached, set `"fairing_jettisoned": false`.

## Deploying To Render

This repo includes [render.yaml](/C:/Code/flight-arc/flight-arc/render.yaml) for a Render web service. The service installs the API dependencies and starts:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
```

Set `FLIGHT_ARC_CORS_ORIGINS` in Render if your frontend is hosted on a different origin.

## Outputs

A nominal run produces:

- an event timeline and state summary in the terminal
- trajectory plots for the current mission
- a JSON payload in `outputs/nominal_latest.json`

A Monte Carlo run produces:

- aggregate statistics in the terminal
- a JSON payload in `outputs/montecarlo_latest.json`
