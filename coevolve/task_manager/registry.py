from __future__ import annotations

from .alfworld import AlfworldTaskManager
from .base import BaseTaskManager
from .webshop import WebShopTaskManager


def build_task_manager(
    task_manager_name: str,
    tasks_path: str = "",
    env_config_path: str = "",
    max_trials: int = 50,
    few_shots_path: str = "",
    mode: str = "train",
    server_url: str = "",
) -> BaseTaskManager:
    if task_manager_name == "alfworld":
        split_override = ""
        if mode == "train":
            split_override = "train"
        elif mode == "test":
            split_override = "eval_out_of_distribution"
        return AlfworldTaskManager(
            tasks_path=tasks_path,
            env_config_path=env_config_path,
            max_trials=max_trials,
            few_shots_path=few_shots_path,
            split_override=split_override,
        )
    if task_manager_name == "webshop":
        return WebShopTaskManager(
            tasks_path=tasks_path,
            server_url=server_url or "http://localhost:5000",
            few_shots_path=few_shots_path,
            max_trials=max_trials,
            mode=mode,
        )
    raise ValueError(f"Unsupported task manager: {task_manager_name}")
