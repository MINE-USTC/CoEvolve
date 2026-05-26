from __future__ import annotations

from dataclasses import asdict
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List

from ..schema import CounterfactualRecord, TrajectoryRecord


class CounterfactualSampler:
    def generate(
        self,
        failed_trajectories: List[TrajectoryRecord],
        seed: int,
        max_samples: int | None = None,
        rerun_fn: Callable[[TrajectoryRecord, str, int], TrajectoryRecord] | None = None,
        on_record: Callable[[CounterfactualRecord], None] | None = None,
        max_workers: int = 1,
        completed_source_ids: set | None = None,
    ) -> List[CounterfactualRecord]:
        rng = random.Random(seed)

        trajectory_pool = list(failed_trajectories)
        if max_samples is not None and max_samples < len(trajectory_pool):
            trajectory_pool = rng.sample(trajectory_pool, max_samples)

        # Skip already-completed counterfactuals (for resume)
        skip_ids = completed_source_ids or set()

        # Pre-compute sampling params (RNG is not thread-safe)
        sample_params = []
        for trajectory in trajectory_pool:
            participants = self._collect_participants(trajectory)
            if not participants:
                continue
            removed_agent = rng.choice(participants)
            # Fork point must be between step 10 and 15 (inclusive).
            # If trajectory is shorter, clamp to available range.
            total_steps = len(trajectory.steps)
            lo = min(10, total_steps)
            hi = min(15, total_steps)
            if lo > hi:
                lo = hi
            removal_start_step = rng.randint(lo, hi)
            if trajectory.trajectory_id in skip_ids:
                continue
            sample_params.append((trajectory, removed_agent, removal_start_step))

        def _run_one(params):
            trajectory, removed_agent, removal_start_step = params
            cf_trajectory = rerun_fn(trajectory, removed_agent, removal_start_step) if rerun_fn else None
            record = CounterfactualRecord(
                source_trajectory_id=trajectory.trajectory_id,
                task_id=trajectory.task.task_id,
                removed_agent=removed_agent,
                removal_start_step=removal_start_step,
                counterfactual_trajectory_id=(
                    cf_trajectory.trajectory_id
                    if cf_trajectory
                    else f"{trajectory.trajectory_id}__cf__{removed_agent}__{removal_start_step}"
                ),
                success=cf_trajectory.success if cf_trajectory else False,
                average_reward=cf_trajectory.average_reward if cf_trajectory else max(0.0, trajectory.average_reward - 0.1),
                metadata={
                    "status": "rerun" if cf_trajectory else "placeholder",
                    "counterfactual_trajectory": asdict(cf_trajectory) if cf_trajectory else {},
                },
            )
            # Immediately persist each completed record
            if on_record is not None:
                try:
                    on_record(record)
                except Exception:
                    pass
            return record

        results: List[CounterfactualRecord] = []
        if max_workers > 1 and len(sample_params) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_run_one, p): i for i, p in enumerate(sample_params)}
                ordered = [None] * len(sample_params)
                for future in as_completed(futures):
                    idx = futures[future]
                    ordered[idx] = future.result()
                results = [r for r in ordered if r is not None]
        else:
            for params in sample_params:
                results.append(_run_one(params))

        return results

    @staticmethod
    def _collect_participants(trajectory: TrajectoryRecord) -> List[str]:
        seen: List[str] = []
        for step in trajectory.steps:
            for agent_name in step.decision.participants:
                if agent_name not in seen:
                    seen.append(agent_name)
        return seen
