from __future__ import annotations

from typing import Any, Dict, List

from ..schema import TaskRecord, TrajectoryRecord, TrajectoryStep


class TrajectoryRecorder:
    def start(self, task: TaskRecord, round_id: int) -> Dict[str, Any]:
        return {
            "trajectory_id": f"{task.task_id}__round_{round_id}",
            "task": task,
            "steps": [],
            "metadata": {"round_id": round_id},
        }

    def append_step(self, buffer: Dict[str, Any], step: TrajectoryStep) -> None:
        steps: List[TrajectoryStep] = buffer["steps"]
        steps.append(step)

    def finalize(self, buffer: Dict[str, Any], success: bool, average_reward: float) -> TrajectoryRecord:
        return TrajectoryRecord(
            trajectory_id=buffer["trajectory_id"],
            task=buffer["task"],
            steps=list(buffer["steps"]),
            success=success,
            average_reward=average_reward,
            metadata=dict(buffer["metadata"]),
        )
