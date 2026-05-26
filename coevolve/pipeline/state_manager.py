from __future__ import annotations

from pathlib import Path
from typing import List

from ..schema import BestRoundState, CounterfactualRecord, RoundStateSnapshot, SkillDecision, TaskRecord, TrajectoryRecord, TrajectoryStep
from ..storage.io import dump_json, ensure_dir, load_json


class RoundStateManager:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.best_counterfactual_records: List[CounterfactualRecord] = []

    def create_round_dir(self, round_id: int):
        return ensure_dir(self.output_dir / f"train_{round_id:03d}")

    def persist_round_state(self, snapshot: RoundStateSnapshot) -> None:
        dump_json(Path(snapshot.round_dir) / "round_state.json", snapshot)

    def persist_best_state(self, best_state: BestRoundState) -> None:
        dump_json(self.output_dir / "best_state.json", best_state)

    def load_best_state(self) -> BestRoundState:
        payload = load_json(self.output_dir / "best_state.json", default=None)
        if not payload:
            return BestRoundState()
        return BestRoundState(**payload)

    def load_round_payload(self, round_dir: str | Path, filename: str, default=None):
        return load_json(Path(round_dir) / filename, default=default)

    def load_round_data(self, round_dir: str | Path, default=None):
        return load_json(Path(round_dir) / "round_data.json", default=default)

    def resolve_counterfactual_records(
        self,
        is_best: bool,
        round_records: List[CounterfactualRecord],
    ) -> tuple[List[CounterfactualRecord], str]:
        if is_best:
            self.best_counterfactual_records = list(round_records)
            return list(round_records), "current_round"
        return list(self.best_counterfactual_records), "best_round"

    def load_best_counterfactuals(self, best_state: BestRoundState) -> List[CounterfactualRecord]:
        if not best_state.round_dir:
            return []
        return self.load_round_counterfactuals(best_state.round_dir)

    def load_round_counterfactuals(self, round_dir: str | Path) -> List[CounterfactualRecord]:
        round_dir = Path(round_dir)
        payload = load_json(round_dir / "counterfactual_records.json", default=None)
        if payload is not None:
            return [CounterfactualRecord(**item) for item in payload]
        cf_dir = round_dir / "counterfactuals"
        if not cf_dir.exists():
            return []
        records: List[CounterfactualRecord] = []
        for path in sorted(cf_dir.glob("*.json")):
            item = load_json(path, default=None)
            if item:
                records.append(CounterfactualRecord(**item))
        return records

    def load_best_trajectories(self, best_state: BestRoundState) -> List[TrajectoryRecord]:
        if not best_state.round_dir:
            return []
        round_dir = Path(best_state.round_dir)
        round_data = load_json(round_dir / "round_data.json", default=None)
        if round_data is not None:
            return [
                self._build_trajectory_record(item)
                for item in list(round_data.get("trajectories", []) or [])
            ]
        payload = load_json(round_dir / "trajectories.json", default=None)
        if payload is not None:
            return [self._build_trajectory_record(item) for item in payload]
        task_dir = round_dir / "tasks"
        if not task_dir.exists():
            return []
        trajectories: List[TrajectoryRecord] = []
        for path in sorted(task_dir.glob("*.json")):
            item = load_json(path, default=None)
            if not item:
                continue
            trajectory_payload = item.get("trajectory", item)
            trajectories.append(self._build_trajectory_record(trajectory_payload))
        return trajectories

    @staticmethod
    def _build_trajectory_record(payload: dict) -> TrajectoryRecord:
        task = TaskRecord(**payload["task"])
        steps = [RoundStateManager._build_trajectory_step(item) for item in payload.get("steps", [])]
        return TrajectoryRecord(
            trajectory_id=payload["trajectory_id"],
            task=task,
            steps=steps,
            success=bool(payload.get("success", False)),
            average_reward=float(payload.get("average_reward", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )

    @staticmethod
    def _build_trajectory_step(payload: dict) -> TrajectoryStep:
        raw_decision = dict(payload.get("decision", {}))
        decision = SkillDecision(
            skill_name=str(raw_decision.get("skill_name", "")),
            participants=list(raw_decision.get("participants", [])),
            subgoal=str(raw_decision.get("subgoal", "")),
            assignment=dict(raw_decision.get("assignment", {})),
            stage_subgoals=dict(raw_decision.get("stage_subgoals", {})),
        )
        return TrajectoryStep(
            step_index=int(payload["step_index"]),
            decision=decision,
            executor_outputs=dict(payload.get("executor_outputs", {})),
            reward=float(payload.get("reward", 0.0)),
            success=bool(payload.get("success", False)),
            executor_memory_hits=dict(payload.get("executor_memory_hits", {})),
            metadata=dict(payload.get("metadata", {})),
        )
