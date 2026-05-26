from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from .schema import AgentConfig, AgentMemoryConfig, AgentProfile, ModelSettings


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_profile(data: Dict[str, Any], default_role: str) -> AgentProfile:
    return AgentProfile(
        role=default_role,
        persona=data.get("persona", ""),
        collaboration_style=data.get("collaboration_style", ""),
        metadata=dict(data.get("metadata", {}) or {}),
    )


def _build_model(data: Dict[str, Any], default_model: str) -> ModelSettings:
    config = data.get("config", {}) or {}
    return ModelSettings(
        name=data.get("name", default_model),
        temperature=float(config.get("temperature", 0.2)),
        max_tokens=int(config.get("max_tokens", 1024)),
        top_p=float(config.get("top_p", 1.0)),
    )


def _build_memory(data: Dict[str, Any]) -> AgentMemoryConfig:
    return AgentMemoryConfig(
        storage_path=data.get("storage_path", ""),
        structure_name=data.get("structure_name", "base"),
        retrieval_top_k=int(data.get("retrieval_top_k", 5)),
        metadata=dict(data.get("metadata", {}) or {}),
    )


def load_agent_configs(
    orchestrator_config_path: str | Path,
    executor_config_path: str | Path,
    default_model: str = "gpt-4o-mini",
) -> Tuple[AgentConfig, List[AgentConfig]]:
    orchestrator_raw = _read_yaml(orchestrator_config_path)
    executor_raw = _read_yaml(executor_config_path)

    orchestrator_data = orchestrator_raw.get("agent", {})
    orchestrator = AgentConfig(
        name=orchestrator_data.get("name", "Orchestrator"),
        role="orchestrator",
        profile=_build_profile(orchestrator_data.get("profile", {}), "orchestrator"),
        model=_build_model(orchestrator_data.get("model", {}), default_model),
        memory=_build_memory(orchestrator_data.get("memory", {})),
    )

    executors: List[AgentConfig] = []
    for item in executor_raw.get("agents", [])[:3]:
        executors.append(
            AgentConfig(
                name=item.get("name", f"Executor_{len(executors) + 1}"),
                role="executor",
                profile=_build_profile(item.get("profile", {}), "executor"),
                model=_build_model(item.get("model", {}), default_model),
                memory=_build_memory(item.get("memory", {})),
            )
        )

    if len(executors) != 3:
        raise ValueError("Executor config must define exactly three executors in the first implementation stage.")

    return orchestrator, executors
