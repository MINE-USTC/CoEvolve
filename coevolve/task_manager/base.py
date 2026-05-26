from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..schema import TaskRecord, TrajectoryRecord


class BaseTaskManager(ABC):
    def __init__(self, tasks_path: str = ""):
        self.tasks_path = tasks_path

    @abstractmethod
    def load_tasks(self) -> List[TaskRecord]:
        raise NotImplementedError

    @abstractmethod
    def sample_tasks(self, tasks: List[TaskRecord], sample_size: int, seed: int) -> List[TaskRecord]:
        raise NotImplementedError

    @abstractmethod
    def build_initial_state(self, task: TaskRecord) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_trajectory(self, trajectory: TrajectoryRecord) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def start_task(self, task: TaskRecord) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def step_task(self, episode: Dict[str, Any], action: str) -> Dict[str, Any]:
        raise NotImplementedError

    # ── Task-specific environment metadata (non-abstract, override per task) ──

    def get_action_format_prompt(self) -> str:
        """Return task-specific action format instructions for executor prompts."""
        return ""

    def prepare_observation(
        self,
        observation: str,
        available_actions: Dict[str, Any] | None = None,
        task: TaskRecord | None = None,
    ) -> str:
        """Normalize environment feedback before it is stored in state/trace."""
        del available_actions, task
        return str(observation or "").strip()

    def get_trivial_feedbacks(self) -> set:
        """Set of lowercased feedback strings excluded from exploration scoring."""
        return {"nothing happens.", "no environment action in this step.",
                "no environment action in this stage.", ""}

    def get_failure_reward_threshold(self) -> float:
        """Reward strictly below this value on an action step signals a failed action.
        ALFWorld: 0.0 (reward=-1 is failure). WebShop: -1.0 (never triggers)."""
        return 0.0

    def get_success_reward_threshold(self) -> float:
        """When done=True, reward >= this threshold means task success."""
        return 0.5

    def compute_average_reward(self, steps: list) -> float:
        """Compute trajectory-level average reward from steps.
        Default: mean of all step rewards. Override for environments
        where intermediate rewards are meaningless (e.g. WebShop)."""
        if not steps:
            return 0.0
        return sum(getattr(s, "reward", 0.0) if not isinstance(s, dict) else s.get("reward", 0.0)
                    for s in steps) / len(steps)

    def use_reward_for_best_round(self) -> bool:
        """If True, best round is determined by average_reward instead of success_rate.
        Override for environments with continuous rewards (e.g. WebShop)."""
        return False

    def get_metric_label(self) -> str:
        """Human-readable label for trajectory.average_reward in logs and summaries."""
        return "final_metric"
