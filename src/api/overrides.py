from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.api.models import MonteCarloRunRequest, NominalRunRequest
from src.config import MissionToolkitConfig, load_config


def config_from_request(
    request: NominalRunRequest | MonteCarloRunRequest | None,
) -> MissionToolkitConfig:
    cfg = load_config()
    if request is None:
        return cfg

    payload = cfg.model_dump(mode="python")
    overrides = request.model_dump(exclude_none=True)
    merged = _deep_merge(payload, overrides)
    return MissionToolkitConfig.model_validate(merged)


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = deepcopy(base)
        for key, value in override.items():
            if key == "stages" and isinstance(merged.get(key), list) and isinstance(value, list):
                merged[key] = _merge_stages(merged[key], value)
                continue
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged

    return deepcopy(override)


def _merge_stages(base_stages: list[Any], override_stages: list[Any]) -> list[Any]:
    if not all(isinstance(stage, dict) for stage in base_stages):
        return deepcopy(override_stages)
    if not all(isinstance(stage, dict) and "id" in stage for stage in override_stages):
        return deepcopy(override_stages)

    merged = [deepcopy(stage) for stage in base_stages]
    index_by_id = {int(stage["id"]): idx for idx, stage in enumerate(merged) if "id" in stage}

    for override in override_stages:
        stage_id = int(override["id"])
        if stage_id in index_by_id:
            idx = index_by_id[stage_id]
            merged[idx] = _deep_merge(merged[idx], override)
        else:
            merged.append(deepcopy(override))

    return merged
