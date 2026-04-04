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

## Outputs

A nominal run produces:

- an event timeline and state summary in the terminal
- trajectory plots for the current mission
- a JSON payload in `outputs/nominal_latest.json`

A Monte Carlo run produces:

- aggregate statistics in the terminal
- a JSON payload in `outputs/montecarlo_latest.json`
