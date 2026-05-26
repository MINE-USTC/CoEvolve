from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

from ..agents import ExecutorAgent, OrchestratorAgent
from ..counterfactual import CounterfactualSampler
from ..evolution import ExecutorMemoryEvolutionEngine, SkillEvolutionEngine, SkillMemoryEvolutionEngine
from ..memory import ExecutorMemoryStore, ProfileStore, SkillMemoryStore
from ..memory.embedding import init_embedder
from ..prompts import build_selection_state_trace_summary, build_combined_memory_update_prompt, build_failure_memory_repair_prompt
from ..scoring import compute_step_scores
from ..schema import BestRoundState, CounterfactualRecord, EvolutionArtifact, ExecutorMemoryEntry, PromptStepContext, RoundMetrics, RoundStateSnapshot, SkillDecision, TaskRecord, TrajectoryRecord, TrajectoryStep
from ..skills import SkillLibraryStore, SkillRuntime
from ..storage import PlainTextLogger
from ..storage.io import dump_json, load_json, ensure_dir
from ..task_manager import BaseTaskManager, build_task_manager
from ..trajectory import TrajectoryRecorder
from .state_manager import RoundStateManager


class RoundRunner:
    def __init__(
        self,
        orchestrator: OrchestratorAgent,
        executors: List[ExecutorAgent],
        task_manager: BaseTaskManager,
        output_dir: str,
        fallback_skill_names: List[str] | None = None,
        skills_path: str = "",
        enable_skill_memory: bool = True,
        enable_executor_memory: bool = True,
        enable_profiles: bool = True,
        enable_counterfactual: bool = True,
        enable_skill_evolution: bool = True,
        enable_skill_memory_evolution: bool = True,
        enable_executor_memory_evolution: bool = True,
        max_reuse_skills_per_round: int = 2,
        max_failure_summary_skills_per_round: int = 1,
        max_evidence_per_candidate: int = 5,
        memory_structure_max_cases: int = 3,
        memory_score_initial: float = 0.5,
        memory_score_lambda: float = 0.2,
        memory_score_omega: float = 0.05,
        memory_score_min: float = 0.05,
        max_workers: int = 1,
        clean_start: bool = False,
        task_manager_name: str = "alfworld",
        task_manager_kwargs: dict | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.executors = executors
        self.task_manager = task_manager
        self.task_manager_name = task_manager_name
        self.task_manager_kwargs = dict(task_manager_kwargs or {})
        self.output_dir = ensure_dir(output_dir)
        self.skills_path = skills_path
        primitive_fallbacks = fallback_skill_names or ["plan", "execute", "reflect"]
        self.skill_library = self._load_initial_skill_library(skills_path, primitive_fallbacks)
        self.enable_skill_memory = enable_skill_memory
        self.enable_executor_memory = enable_executor_memory
        self.enable_profiles = enable_profiles
        self.enable_counterfactual = enable_counterfactual
        self.enable_skill_evolution = enable_skill_evolution
        self.enable_skill_memory_evolution = enable_skill_memory_evolution
        self.enable_executor_memory_evolution = enable_executor_memory_evolution
        self.skill_trial_max_uses_per_task = 2
        self.skill_trial_min_tasks = 5
        self.skill_trial_promote_success_rate = 0.60
        self.skill_trial_max_failed_rounds = 1

        # Use user-provided max_workers value (do not cap to executor count)
        self.max_workers = max(1, max_workers)

        self.recorder = TrajectoryRecorder()
        self.skill_runtime = SkillRuntime()
        self.state_manager = RoundStateManager(output_dir=str(self.output_dir))
        self.counterfactual_sampler = CounterfactualSampler()
        self.skill_evolution_engine = SkillEvolutionEngine(
            max_new_skills_per_round=max_reuse_skills_per_round,
            max_failure_summary_skills_per_round=max_failure_summary_skills_per_round,
            max_evidence_per_candidate=max_evidence_per_candidate,
            llm_client=self.orchestrator.client,
            llm_model=self.orchestrator.config.model.name,
            llm_temperature=self.orchestrator.config.model.temperature,
        )
        self.skill_memory_evolution_engine = SkillMemoryEvolutionEngine(
            max_contrast_cases=memory_structure_max_cases,
            llm_client=self.orchestrator.client,
            llm_model=self.orchestrator.config.model.name,
            llm_temperature=self.orchestrator.config.model.temperature,
        )
        self.executor_memory_evolution_engine = ExecutorMemoryEvolutionEngine(
            max_contrast_cases=memory_structure_max_cases,
            llm_client=self.orchestrator.client,
            llm_model=self.orchestrator.config.model.name,
            llm_temperature=self.orchestrator.config.model.temperature,
        )
        # Keep runtime memories scoped to this run's output directory. The YAML
        # memory.storage_path values describe legacy/default locations only; test
        # and train runs should not read from or write to data/memory implicitly.
        run_memory_root = self.output_dir / "memory"
        self.skill_memory = SkillMemoryStore(
            str(run_memory_root / "orchestrator"),
            initial_score=memory_score_initial,
            score_lambda=memory_score_lambda,
            score_omega=memory_score_omega,
            min_score_to_keep=memory_score_min,
        )
        for spec in self.skill_library.skills.values():
            fixed_field = str(dict(spec.metadata or {}).get("fixed_skill_memory_field", "")).strip()
            if fixed_field:
                self.skill_memory.register_fixed_field(fixed_field)
        self.executor_memories = {
            agent.name: ExecutorMemoryStore(
                str(run_memory_root / agent.name),
                agent_name=agent.name,
                initial_score=memory_score_initial,
                score_lambda=memory_score_lambda,
                score_omega=memory_score_omega,
                min_score_to_keep=memory_score_min,
            )
            for agent in executors
        }
        if clean_start:
            # Clear all memory storage files
            import shutil
            paths_to_clear = [self.skill_memory.storage_dir] + [s.storage_dir for s in self.executor_memories.values()]
            for path in paths_to_clear:
                if path and Path(path).exists():
                    shutil.rmtree(path, ignore_errors=True)
        else:
            self._load_runtime_memories_from_storage()
        init_embedder()
        self.profile_store = ProfileStore(
            storage_path=str(self.output_dir / "profiles"),
            agent_names=[agent.name for agent in executors],
            collaboration_styles={agent.name: agent.config.profile.collaboration_style for agent in executors},
        )
        self.log_dir = ensure_dir(self.output_dir / "logs")
        self.detail_logger: PlainTextLogger | None = None
        self.simple_logger: PlainTextLogger | None = None
        
        # Statistics tracking
        self.run_start_time = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._stats_lock = threading.Lock()
        self._task_stats_local = threading.local()
        self._task_log_local = threading.local()
        self._memory_snapshot_lock = threading.Lock()
        self._active_round_output_dir: Path | None = None
        self._install_usage_callback(self.orchestrator.client)
        
        self._console("Initialization complete.")

    def _create_fresh_task_manager(self) -> BaseTaskManager:
        """Create a brand-new task manager instance, identical to the original."""
        return build_task_manager(self.task_manager_name, **self.task_manager_kwargs)

    def run(
        self,
        mode: str,
        rounds: int,
        sample_size: int,
        max_steps_per_task: int,
        base_seed: int,
        counterfactual_samples_per_round: int,
        resume: bool = False,
        evaluate_best: bool = False,
    ) -> BestRoundState:
        import time
        self.run_start_time = time.time()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        
        tasks = self.task_manager.load_tasks()
        best_state = self.state_manager.load_best_state() if (resume or evaluate_best or mode == "test") else BestRoundState()
        if best_state.round_dir:
            self._restore_state(best_state)

        if mode == "test":
            return self._run_test(tasks, sample_size, max_steps_per_task, base_seed, best_state, resume=resume)

        start_round = int(best_state.round_id or 0) + 1 if resume and best_state.round_id else 1
        for round_id in range(start_round, rounds + 1):
            round_dir = self.state_manager.create_round_dir(round_id)
            self._active_round_output_dir = round_dir
            round_data_path = round_dir / "round_data.json"
            round_data = self.state_manager.load_round_data(round_dir, default=None) if resume else None
            is_resuming_this_round = resume and round_data is not None
            self.detail_logger = PlainTextLogger(self.log_dir / f"round_{round_id:03d}.detail.log", overwrite=not is_resuming_this_round)
            self.simple_logger = PlainTextLogger(self.log_dir / f"round_{round_id:03d}.simple.log", overwrite=not is_resuming_this_round)
            self._console(f"Round {round_id} started.")
            self._write_pair(f"Round {round_id} started.", f"mode={mode} sample_seed={base_seed + round_id}")
            self._write_blank()
            if round_id > 1 and not is_resuming_this_round:
                self._clear_memories_for_next_round()
                self._write_pair(
                    "Prepared fresh round memory state.",
                    f"round_id={round_id} mode={mode}",
                )
            # Fresh task manager for each round (avoids stale env state)
            self.task_manager = self._create_fresh_task_manager()
            if is_resuming_this_round and round_data is not None:
                sampled_tasks = [TaskRecord(**item) for item in list(round_data.get("sampled_tasks", []) or [])]
            else:
                sampled_tasks = self.task_manager.sample_tasks(tasks, sample_size=sample_size, seed=base_seed + round_id)

            # Resume: load partial progress from previous crash
            trajectories: List[TrajectoryRecord] = []
            completed_task_ids: set = set()
            if is_resuming_this_round and round_data is not None:
                partial = list(round_data.get("trajectories", []) or [])
                if partial:
                    trajectories = [self.state_manager._build_trajectory_record(item) for item in partial]
                    completed_task_ids = {t.task.task_id for t in trajectories}
                    self._console(f"Resuming round {round_id}: {len(trajectories)} tasks already completed")
                self._restore_executor_memories_from_output_dir(round_dir)
                # Restore profiles
                if self.enable_profiles and self.profile_store.storage_dir:
                    from pathlib import Path as _Path
                    prof_path = _Path(self.profile_store.storage_dir) / "profiles.json"
                    if prof_path.exists():
                        self.profile_store.load_snapshot(load_json(prof_path, default=None))
                        self._console(f"Restored profiles from disk")
            else:
                self._prepare_round_partial_files(round_dir)
                self._persist_round_data(
                    target_path=round_data_path,
                    sampled_tasks=sampled_tasks,
                    trajectories=[],
                    total_tasks=len(sampled_tasks),
                )

            total_tasks = len(sampled_tasks)
            remaining_tasks = [(i, t) for i, t in enumerate(sampled_tasks, start=1) if t.task_id not in completed_task_ids]

            if self.max_workers > 1 and len(remaining_tasks) > 1:
                # Parallel execution: tasks run concurrently, memory updated immediately on completion
                import threading
                from concurrent.futures import ThreadPoolExecutor, as_completed
                memory_lock = threading.Lock()

                def run_and_update(task_index, task):
                    trajectory = self._execute_task(
                        task=task,
                        round_id=round_id,
                        max_steps_per_task=max_steps_per_task,
                        update_memory=False,
                    )
                    memory_update = self._prepare_memory_update_from_trajectory(trajectory)
                    with memory_lock:
                        self._apply_memory_update_from_trajectory(trajectory, memory_update)
                        trajectories.append(trajectory)
                        # Use len(trajectories) as completed count to handle out-of-order completion
                        self._persist_task_trajectory(round_dir, sampled_tasks, trajectory, trajectories, len(trajectories), total_tasks)
                    return trajectory

                # Submit up to self.max_workers futures at a time; when one completes, submit next.
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    task_iter = iter(remaining_tasks)
                    futures = {}

                    # seed initial batch
                    for _ in range(min(self.max_workers, len(remaining_tasks))):
                        item = next(task_iter, None)
                        if item is None:
                            break
                        task_index, task = item
                        fut = pool.submit(run_and_update, task_index, task)
                        futures[fut] = (task_index, task)

                    # keep submitting one new task per completed future
                    while futures:
                        # as_completed yields completed futures as they finish
                        for future in as_completed(list(futures.keys())):
                            # propagate exceptions
                            future.result()
                            # remove completed
                            futures.pop(future, None)
                            # submit next task if available
                            next_item = next(task_iter, None)
                            if next_item is not None:
                                ni, nt = next_item
                                newf = pool.submit(run_and_update, ni, nt)
                                futures[newf] = (ni, nt)
            else:
                # Sequential execution
                for task_index, task in remaining_tasks:
                    trajectory = self._execute_task(
                        task=task,
                        round_id=round_id,
                        max_steps_per_task=max_steps_per_task,
                        update_memory=True,
                    )
                    trajectories.append(trajectory)
                    self._persist_task_trajectory(round_dir, sampled_tasks, trajectory, trajectories, task_index, total_tasks)
            metrics = self._build_round_metrics(round_id=round_id, trajectories=trajectories, best_state=best_state)
            failed_trajectories = [item for item in trajectories if not item.success]
            success_trajectories = [item for item in trajectories if item.success]
            # Resume: load already-completed counterfactual records
            existing_cf_records = self.state_manager.load_round_counterfactuals(round_dir) if resume else []
            completed_cf_source_ids = {record.source_trajectory_id for record in existing_cf_records}
            if existing_cf_records:
                self._console(f"Resuming counterfactual: {len(existing_cf_records)} already completed")
            # Fresh task manager for counterfactual sampling (clean env, no stale state)
            self.task_manager = self._create_fresh_task_manager()
            new_counterfactuals = self.counterfactual_sampler.generate(
                failed_trajectories=failed_trajectories,
                seed=base_seed + round_id,
                max_samples=counterfactual_samples_per_round,
                rerun_fn=self._rerun_counterfactual if self.enable_counterfactual else None,
                on_record=lambda record: self._on_counterfactual_generated(
                    round_dir=round_dir,
                    record=record,
                ),
                max_workers=self.max_workers,
                completed_source_ids=completed_cf_source_ids,
            )
            fresh_counterfactuals = existing_cf_records + new_counterfactuals
            (
                evolution_successes,
                evolution_failures,
                counterfactual_records,
                counterfactual_source,
                fallback_metadata,
            ) = self._resolve_evolution_context(
                best_state=best_state,
                metrics=metrics,
                current_trajectories=trajectories,
                fresh_counterfactuals=fresh_counterfactuals,
            )
            self._persist_selected_counterfactuals(
                round_dir=round_dir,
                counterfactual_records=counterfactual_records,
                counterfactual_source=counterfactual_source,
            )
            self._write_pair(
                f"Counterfactual sampling: {len(fresh_counterfactuals)} generated from {len(failed_trajectories)} failed trajectories",
                f"counterfactual_source={counterfactual_source} "
                + "; ".join(
                    f"src={r.source_trajectory_id} removed={r.removed_agent}@{r.removal_start_step} success={r.success}"
                    for r in fresh_counterfactuals[:5]
                ),
            )
            skill_trial_artifact = self._evaluate_skill_trials(
                round_id=round_id,
                trajectories=trajectories,
            )
            evolution_artifacts = [skill_trial_artifact] + self._run_evolution(
                success_trajectories=evolution_successes,
                failed_trajectories=evolution_failures,
                counterfactual_records=counterfactual_records,
            )
            for artifact in evolution_artifacts:
                self._write_pair(
                    f"Evolution [{artifact.name}]: applied={artifact.applied} - {artifact.summary[:120]}",
                    f"payload_keys={list(artifact.payload.keys())}",
                )
            self._write_pair(
                f"Round {round_id} summary: success_rate={metrics.average_success_rate:.3f} {self._metric_label()}={metrics.average_reward:.3f}",
                f"failed_trajectories={len(failed_trajectories)} counterfactual_source={counterfactual_source} "
                f"counterfactual_count={len(counterfactual_records)} evolution_artifacts={[item.name for item in evolution_artifacts]}",
            )
            self._write_blank()

            self._clear_memories_for_next_round()
            self._write_pair(f"Memory entries cleared for next round.", f"round_id={round_id}")

            self._persist_round_data(
                target_path=round_data_path,
                sampled_tasks=sampled_tasks,
                trajectories=trajectories,
                total_tasks=total_tasks,
            )
            # Core state files (loaded by _restore_state / best-round replay)
            self._persist_round_memory_snapshots(round_dir)
            dump_json(round_dir / "summary.json", {
                "round_id": round_id,
                "task_count": metrics.task_count,
                "success_count": metrics.success_count,
                "success_rate": metrics.average_success_rate,
                "metric_label": self._metric_label(),
                "metric_value": metrics.average_reward,
                "failed_trajectory_count": len(failed_trajectories),
                "counterfactual_count": len(counterfactual_records),
                "evolution_summaries": [a.summary for a in evolution_artifacts],
                "best_so_far": metrics.best_so_far,
            })

            round_state = RoundStateSnapshot(
                round_id=round_id,
                round_dir=str(round_dir),
                average_success_rate=metrics.average_success_rate,
                is_best=metrics.best_so_far,
                failed_trajectory_ids=[item.trajectory_id for item in failed_trajectories],
                counterfactual_source=counterfactual_source,
                metadata={
                    "average_reward": metrics.average_reward,
                    "counterfactual_count": len(counterfactual_records),
                    "fresh_counterfactual_count": len(fresh_counterfactuals),
                    "skill_library_size": len(self.skill_library),
                    "evolution_summaries": [artifact.summary for artifact in evolution_artifacts],
                    **fallback_metadata,
                },
            )
            self.state_manager.persist_round_state(round_state)

            if metrics.best_so_far:
                best_state = BestRoundState(
                    round_id=round_id,
                    average_success_rate=metrics.average_success_rate,
                    round_dir=str(round_dir),
                    metadata={
                        "average_reward": metrics.average_reward,
                        "best_average_reward": metrics.average_reward,
                        "counterfactual_count": len(counterfactual_records),
                    },
                )
                self.state_manager.persist_best_state(best_state)
                self._write_pair(
                    f"Best state updated: round={round_id} success_rate={metrics.average_success_rate:.3f}",
                    f"metadata={best_state.metadata}",
                )

            self._console(f"Round {round_id} finished. success_rate={metrics.average_success_rate:.3f}, {self._metric_label()}={metrics.average_reward:.3f}")
            self._active_round_output_dir = None

        # Output final statistics
        if self.run_start_time:
            import time
            elapsed_time = time.time() - self.run_start_time
            self._write_pair(
                f"Run completed. Total execution time: {elapsed_time:.2f} seconds",
                f"elapsed_seconds={elapsed_time:.2f}",
            )
            self._write_pair(
                f"Total token usage: completion={self.total_completion_tokens}, prompt={self.total_prompt_tokens}",
                f"completion_tokens={self.total_completion_tokens}, prompt_tokens={self.total_prompt_tokens}",
            )
            self._console(f"Total execution time: {elapsed_time:.2f} seconds")
            self._console(f"completion_tokens:{self.total_completion_tokens}, prompt_tokens:{self.total_prompt_tokens}")

        return best_state

    def _execute_task(
        self,
        task: TaskRecord,
        round_id: int,
        max_steps_per_task: int,
        removed_agent: str | None = None,
        removal_start_step: int | None = None,
        update_memory: bool = True,
        trajectory_suffix: str = "",
        run_label: str | None = None,
        run_metadata: Dict[str, Any] | None = None,
    ) -> TrajectoryRecord:
        task_start_time = time.time()
        self._task_stats_local.active = {"prompt_tokens": 0, "completion_tokens": 0, "detail_lines": []}
        episode = self.task_manager.start_task(task)
        initial_state = self.task_manager.build_initial_state(task)
        buffer = self.recorder.start(task=task, round_id=round_id)
        task_run_metadata = dict(run_metadata or {})
        if run_label:
            buffer["trajectory_id"] = f"{task.task_id}__{run_label}"
            buffer["metadata"] = {"mode": run_label, **task_run_metadata}
        elif trajectory_suffix:
            buffer["trajectory_id"] = f"{buffer['trajectory_id']}__{trajectory_suffix}"
            if task_run_metadata:
                buffer["metadata"].update(task_run_metadata)
        elif task_run_metadata:
            buffer["metadata"].update(task_run_metadata)
        state_summary = episode.get("state_summary", initial_state["summary"])
        last_observation = episode.get("last_observation", state_summary)
        final_success = False
        log_context = None
        if run_label:
            parts = [f"mode={run_label}"]
            source_round = task_run_metadata.get("source_best_round_id")
            if source_round is not None:
                parts.append(f"source_best_round={source_round}")
            log_context = " ".join(parts)
        self._log_task_start(
            task=task,
            round_id=round_id,
            trajectory_id=buffer["trajectory_id"],
            state_summary=state_summary,
            removed_agent=removed_agent,
            removal_start_step=removal_start_step,
            run_context=log_context,
        )
        self._console(f"Task {task.task_id} started.")
        task_few_shots = list(task.metadata.get("few_shots", []))
        total_stages_used = 0

        if episode.get("done"):
            # Task failed to load, skip execution
            self._console(f"Task {task.task_id} skipped (failed to load).")

        for step_index in range(1, max_steps_per_task + 1):
            if episode.get("done"):
                break
            available_executor_names = [
                agent.name
                for agent in self.executors
                if not (removed_agent and removal_start_step and step_index >= removal_start_step and agent.name == removed_agent)
            ]
            profile_snapshots = self.profile_store.all() if self.enable_profiles else []
            prompt_context = self._build_prompt_step_context(
                buffer=buffer,
                state_summary=state_summary,
                last_observation=last_observation,
                current_step_index=total_stages_used + 1,
                max_steps=max_steps_per_task,
                remaining_steps=max(0, max_steps_per_task - total_stages_used),
            )
            selection = self.orchestrator.select_skill(
                task_description=task.description,
                task_goal=task.task_goal,
                prompt_context=prompt_context,
                available_skills=self._available_skill_names(buffer=buffer),
                available_skill_metadata=self._available_skill_metadata_with_golden(buffer=buffer),
                few_shots=task_few_shots,
            )
            selection = self._promote_execute_when_stuck(
                selection=selection,
                prompt_context=prompt_context,
            )
            skill_spec = self._skill_spec_payload(selection["skill_name"]) or {}
            golden_example_id = str(dict(skill_spec.get("selection_metadata", {})).get("golden_example_id", ""))
            skill_detail_memory_hits = self.skill_memory.retrieve_for_skill(
                skill_name=selection["skill_name"],
                task_description=task.description,
                task_goal=task.task_goal,
                state_trace_summary=prompt_context.state_trace_summary,
                memory_metadata=self._skill_memory_query_metadata(skill_spec, selection["skill_name"]),
                recent_environment_feedback=prompt_context.recent_environment_feedback,
                subgoal=selection["subgoal"],
                top_k=self._skill_memory_top_k(skill_spec),
                golden_example_id=golden_example_id,
            ) if self.enable_skill_memory else []
            assignment_result = self.orchestrator.assign_skill(
                task_description=task.description,
                task_goal=task.task_goal,
                skill_subgoal=selection["subgoal"],
                prompt_context=prompt_context,
                selected_skill_name=selection["skill_name"],
                selected_skill_detail=dict(skill_spec.get("detail", {})),
                skill_memory_hits=skill_detail_memory_hits,
                executor_names=available_executor_names,
                profile_snapshots=profile_snapshots,
                executor_personas={agent.name: agent.config.profile.persona for agent in self.executors},
            )
            assignment = dict(assignment_result.get("assignment", {}))
            participants = self._participants_from_assignment(assignment)
            decision = SkillDecision(
                skill_name=selection["skill_name"],
                participants=participants,
                subgoal=selection["subgoal"],
                assignment=assignment,
                stage_subgoals=dict(assignment_result.get("stage_subgoals", {})),
            )

            selected = [agent for agent in self.executors if agent.name in decision.participants]
            stage_roles = {
                str(stage.get("stage_name", "")): str(stage.get("stage_role", ""))
                for stage in dict(skill_spec.get("detail", {})).get("stages", [])
            }
            assigned_parts: dict[str, List[str]] = {}
            for stage_name, names in decision.assignment.items():
                role = stage_roles.get(stage_name, "")
                for name in names:
                    assigned_parts.setdefault(name, []).append(f"{stage_name}:{role}" if role else stage_name)
            executor_memory_hits = {
                agent.name: self.executor_memories[agent.name].retrieve(
                    task_description=task.description,
                    stage_subgoal=self._assigned_stage_subgoal_text(decision, agent.name),
                    subgoal=decision.subgoal,
                    state_trace_summary=prompt_context.state_trace_summary,
                    recent_environment_feedback=prompt_context.recent_environment_feedback,
                    memory_metadata={},
                    skill_name=decision.skill_name,
                    assigned_parts=assigned_parts.get(agent.name, []),
                    top_k=agent.config.memory.retrieval_top_k,
                ) if self.enable_executor_memory else []
                for agent in selected
            }
            executor_failure_hits = {
                agent.name: self.executor_memories[agent.name].retrieve_failure(
                    task_description=task.description,
                    stage_subgoal=self._assigned_stage_subgoal_text(decision, agent.name),
                    subgoal=decision.subgoal,
                    state_trace_summary=prompt_context.state_trace_summary,
                    recent_environment_feedback=prompt_context.recent_environment_feedback,
                    memory_metadata={},
                ) if self.enable_executor_memory else None
                for agent in selected
            }
            # Inject failure hit into memory_hits with marker
            for agent in selected:
                fail_hit = executor_failure_hits.get(agent.name)
                if fail_hit:
                    fail_hit["_is_failure"] = True
                    executor_memory_hits[agent.name].append(fail_hit)
            self._console(
                f"Task {task.task_id} step {step_index}: "
                f"skill={selection['skill_name']} "
                f"participants={decision.participants} assignment_status={assignment_result.get('model_status', 'unknown')}"
            )
            runtime_result = self.skill_runtime.run(
                decision=decision,
                task_description=task.description,
                task_goal=task.task_goal,
                prompt_context=prompt_context,
                selected_executors=selected,
                executor_memory_hits=executor_memory_hits,
                skill_spec=skill_spec,
                few_shots=task_few_shots,
                env_step_fn=lambda action: self.task_manager.step_task(episode, action),
                action_format_prompt=self.task_manager.get_action_format_prompt(),
                current_step_index=prompt_context.current_step_index,
                max_steps=max_steps_per_task,
                remaining_steps=prompt_context.remaining_steps,
            )
            step_result = runtime_result["env_result"]
            executor_outputs = runtime_result["executor_outputs"]
            success = bool(step_result["success"])
            reward = float(step_result.get("total_reward", step_result["reward"]))
            self._log_task_step(
                task=task,
                step_index=step_index,
                state_summary=state_summary,
                last_observation=last_observation,
                selection=selection,
                decision=decision,
                skill_spec=skill_spec,
                skill_memory_hits=skill_detail_memory_hits,
                assignment_result=assignment_result,
                executor_outputs=executor_outputs,
                executor_memory_hits=executor_memory_hits,
                env_result=step_result,
            )
            self._console(
                f"Task {task.task_id} step {step_index} runtime: "
                f"chosen_action={runtime_result.get('chosen_action', '') or '<none>'} "
                f"env_interactions={step_result.get('interaction_count', 0)} "
                f"success={success} reward={reward:.3f}"
            )
            for agent_name, payload in executor_outputs.items():
                self._console(
                    f"  {agent_name} stage={payload.get('stage_name', '')} "
                    f"status={payload.get('model_status', 'unknown')} "
                    f"action={payload.get('proposed_action', '') or '<none>'} "
                    f"summary={self._short_text(payload.get('output') or '', 120)}"
                )

            self.recorder.append_step(
                buffer,
                TrajectoryStep(
                    step_index=step_index,
                    decision=decision,
                    executor_memory_hits=executor_memory_hits,
                    executor_outputs=executor_outputs,
                    reward=reward,
                    success=success,
                    metadata={
                        **({"round_id": round_id} if not run_label else {"mode": run_label, **task_run_metadata}),
                        "chosen_action": runtime_result["chosen_action"],
                        "last_observation": last_observation,
                        "state_trace_summary": prompt_context.state_trace_summary,
                        "environment_feedback": (
                            str(step_result.get("last_observation", "")).strip()
                            if int(step_result.get("interaction_count", 0)) > 0
                            else "No environment action in this step."
                        ),
                        "score": float(step_result.get("score", (step_result.get("info", {}) or {}).get("score", 0.0))),
                        "raw_score": float(step_result.get("raw_score", (step_result.get("info", {}) or {}).get("raw_score", 0.0))),
                        "env_interaction_count": step_result.get("interaction_count", 0),
                        "skill_detail_memory_hits": skill_detail_memory_hits,
                        "workflow_trace": runtime_result.get("workflow_trace", {}),
                        "removed_agent": removed_agent,
                        "removal_start_step": removal_start_step,
                    },
                ),
            )
            episode.update(
                {
                    "state_summary": step_result["state_summary"],
                    "last_observation": step_result["last_observation"],
                    "done": step_result["done"],
                    "score": step_result.get("score") or (step_result.get("info", {}) or {}).get("score", 0.0),
                    "raw_score": step_result.get("raw_score") or (step_result.get("info", {}) or {}).get("raw_score", 0.0),
                    "step_count": (step_result.get("info", {}) or {}).get("step_count", episode.get("step_count", 0)),
                    "stop_interaction": bool(step_result.get("stop_interaction", False)),
                }
            )
            state_summary = step_result["state_summary"]
            last_observation = step_result["last_observation"]
            final_success = success
            step_stage_count = max(1, len(runtime_result.get("workflow_trace", {}).get("stage_runs", executor_outputs)))
            total_stages_used += step_stage_count
            if step_result["done"] or step_result.get("stop_interaction", False):
                break
            if total_stages_used >= max_steps_per_task:
                break

        steps = buffer["steps"]
        average_reward = self.task_manager.compute_average_reward(steps)
        trajectory = self.recorder.finalize(buffer=buffer, success=final_success if steps else False, average_reward=average_reward)
        task_stats = dict(getattr(self._task_stats_local, "active", {}) or {})
        trajectory.metadata["elapsed_seconds"] = round(time.time() - task_start_time, 3)
        trajectory.metadata["prompt_tokens"] = int(task_stats.get("prompt_tokens", 0))
        trajectory.metadata["completion_tokens"] = int(task_stats.get("completion_tokens", 0))
        if update_memory:
            self._update_memories_from_trajectory(trajectory)
        self._log_task_end(task=task, trajectory=trajectory, memory_updated=update_memory)
        task_stats = dict(getattr(self._task_stats_local, "active", {}) or {})
        trajectory.metadata["detail_log_lines"] = list(task_stats.get("detail_lines", []))
        self._task_stats_local.active = None
        self._console(
            f"Task {task.task_id} finished. success={trajectory.success} "
            f"{self._metric_label()}={trajectory.average_reward:.3f} steps={len(trajectory.steps)} "
            f"time={trajectory.metadata.get('elapsed_seconds', 0.0):.2f}s "
            f"prompt_tokens={trajectory.metadata.get('prompt_tokens', 0)} "
            f"completion_tokens={trajectory.metadata.get('completion_tokens', 0)}"
        )
        return trajectory

    def _build_round_metrics(self, round_id: int, trajectories: List[TrajectoryRecord], best_state: BestRoundState) -> RoundMetrics:
        task_count = len(trajectories)
        success_count = sum(1 for item in trajectories if item.success)
        average_success_rate = success_count / max(1, task_count)
        average_reward = sum(item.average_reward for item in trajectories) / max(1, task_count)
        if self.task_manager.use_reward_for_best_round():
            best_reward = (
                best_state.metadata.get("average_reward", best_state.metadata.get("best_average_reward", 0.0))
                if best_state.metadata
                else 0.0
            )
            is_best = average_reward >= best_reward
        else:
            is_best = average_success_rate >= best_state.average_success_rate
        return RoundMetrics(
            round_id=round_id,
            task_count=task_count,
            success_count=success_count,
            average_success_rate=average_success_rate,
            average_reward=average_reward,
            best_so_far=is_best,
            metadata={"best_average_reward": average_reward} if self.task_manager.use_reward_for_best_round() else {},
        )

    def _prepare_memory_update_from_trajectory(self, trajectory: TrajectoryRecord) -> Dict[str, Any] | None:
        if not trajectory.steps:
            return None
        traj_steps_detail = [self._serialize_step_for_memory(step) for step in trajectory.steps]

        # Select steps
        skill_selected = self._select_skill_memory_steps(trajectory) if self.enable_skill_memory else []
        executor_selected: Dict[str, List[int]] = {}
        if self.enable_executor_memory:
            for store in self.executor_memories.values():
                executor_selected[store.agent_name] = self._select_executor_memory_steps(trajectory, store.agent_name)

        # Gather metadata fields. Skill-specific fixed fields are provided per
        # selected step below, so the common list must not include every skill's
        # dedicated field.
        skill_fields = [
            field
            for field in list(self.skill_memory.metadata_fields)
            if field not in self.skill_memory.fixed_field_names
        ]
        skill_fixed_fields = self._skill_memory_fixed_fields_by_step(trajectory, skill_selected)
        exec_fields: Dict[str, List[str]] = {}
        for name, store in self.executor_memories.items():
            exec_fields[name] = store.metadata_fields

        # Select failure steps
        failure_selections: Dict[str, int | None] = {}
        if self.enable_executor_memory:
            failure_selections = self._select_failure_memory_steps(trajectory)

        failure_fields_by_agent: Dict[str, List[str]] = {}
        for name, store in self.executor_memories.items():
            failure_fields_by_agent[name] = store.failure_metadata_fields

        # One combined LLM call
        skill_llm_meta, exec_llm_meta, fail_llm_meta = self._call_combined_memory_update(
            trajectory=trajectory,
            traj_steps_detail=traj_steps_detail,
            skill_selected=skill_selected,
            skill_fields=skill_fields,
            skill_fixed_fields=skill_fixed_fields,
            executor_selected=executor_selected,
            exec_fields=exec_fields,
            failure_selections=failure_selections,
            failure_fields_by_agent=failure_fields_by_agent,
        )
        fail_llm_meta = self._repair_missing_failure_metadata(
            trajectory=trajectory,
            traj_steps_detail=traj_steps_detail,
            failure_selections=failure_selections,
            failure_fields_by_agent=failure_fields_by_agent,
            fail_llm_meta=fail_llm_meta,
        )
        return {
            "skill_selected": skill_selected,
            "executor_selected": executor_selected,
            "failure_selections": failure_selections,
            "skill_fixed_fields": skill_fixed_fields,
            "skill_llm_meta": skill_llm_meta,
            "exec_llm_meta": exec_llm_meta,
            "fail_llm_meta": fail_llm_meta,
        }

    def _apply_memory_update_from_trajectory(self, trajectory: TrajectoryRecord, memory_update: Dict[str, Any] | None) -> None:
        if not trajectory.steps or not memory_update:
            return
        skill_selected = list(memory_update.get("skill_selected", []) or [])
        executor_selected = dict(memory_update.get("executor_selected", {}) or {})
        failure_selections = dict(memory_update.get("failure_selections", {}) or {})
        skill_fixed_fields = {
            int(k): str(v)
            for k, v in dict(memory_update.get("skill_fixed_fields", {}) or {}).items()
            if str(v).strip()
        }
        skill_llm_meta = dict(memory_update.get("skill_llm_meta", {}) or {})
        exec_llm_meta = dict(memory_update.get("exec_llm_meta", {}) or {})
        fail_llm_meta = dict(memory_update.get("fail_llm_meta", {}) or {})

        # Apply skill memory
        if self.enable_skill_memory and skill_selected:
            skill_update = self.skill_memory.update_from_trajectory(
                trajectory,
                selected_steps=skill_selected,
                llm_metadata=skill_llm_meta,
                fixed_fields_by_step=skill_fixed_fields,
            )
            self.skill_memory.persist_incremental(
                touched_entry_ids=list(skill_update.get("touched_entry_ids", [])),
                removed_entry_ids=list(skill_update.get("removed_entry_ids", [])),
                reason=f"task_update:{trajectory.trajectory_id}",
            )
            self._update_skill_golden_examples()

        # Apply executor memory
        if self.enable_executor_memory:
            for store in self.executor_memories.values():
                agent_steps = executor_selected.get(store.agent_name, [])
                agent_meta = dict(exec_llm_meta.get(store.agent_name, {}) or {})
                for step_idx in agent_steps:
                    step = next((s for s in trajectory.steps if s.step_index == step_idx), None)
                    if step is None:
                        continue
                    payload = dict(step.executor_outputs.get(store.agent_name, {}))
                    existing = dict(agent_meta.get(step_idx, {}) or {})
                    normalized = {
                        key: str(existing.get(key, "")).strip()
                        for key in store.metadata_fields
                    }
                    if self._metadata_is_empty_or_na(normalized):
                        agent_meta[step_idx] = self._fallback_executor_metadata(store, step, payload)
                store.update_from_trajectory(trajectory, selected_steps=agent_steps, llm_metadata=agent_meta)

            # Add failure memories (reward=-1 stages)
            for agent_name, step_idx in failure_selections.items():
                store = self.executor_memories.get(agent_name)
                if not store or step_idx is None:
                    continue
                step = next((s for s in trajectory.steps if s.step_index == step_idx), None)
                if step is None:
                    continue
                payload = dict(step.executor_outputs.get(agent_name, {}))
                raw_meta = dict(fail_llm_meta.get(agent_name, {}).get(step_idx, {}))
                expected_fail_fields = store.failure_metadata_fields
                meta = {
                    key: str(raw_meta.get(key, "")).strip() or "N/A"
                    for key in expected_fail_fields
                }
                if self._metadata_has_missing_or_na(meta, expected_fail_fields):
                    continue
                from uuid import uuid4
                store.add_failure_entry(ExecutorMemoryEntry(
                    entry_id=str(uuid4()),
                    agent_name=agent_name,
                    task_description=trajectory.task.description,
                    subgoal=step.decision.subgoal,
                    stage_subgoal=str(step.decision.stage_subgoals.get(str(payload.get("stage_name", "")), "")),
                    state_trace_summary=str(step.metadata.get("state_trace_summary", "")),
                    recent_environment_feedback=str(step.metadata.get("environment_feedback", "")),
                    memory_metadata=meta,
                    action_summary=str(payload.get("proposed_action") or payload.get("output") or ""),
                    assigned_part=ExecutorMemoryStore._compose_assigned_part(
                        stage_name=str(payload.get("stage_name", "")),
                        stage_role=str(payload.get("stage_role", "")),
                    ),
                    workflow_name=step.decision.skill_name,
                    success=False,
                    reward=float(step.reward),
                    weight=0.0,
                ))

            # Persist executor memories
            for store in self.executor_memories.values():
                store.persist_full_rewrite(reason=f"task_update:{trajectory.trajectory_id}")
            if self._active_round_output_dir is not None:
                self._persist_executor_memory_snapshots(self._active_round_output_dir)

        if self.enable_profiles:
            self.profile_store.update_from_trajectory(trajectory)
            self.profile_store.persist(reason=f"task_update:{trajectory.trajectory_id}")

    @staticmethod
    def _metadata_is_empty_or_na(metadata: Dict[str, Any]) -> bool:
        if not metadata:
            return True
        for value in metadata.values():
            text = str(value).strip()
            if text and text.lower() not in {"n/a", "na", "none", "null"}:
                return False
        return True

    @classmethod
    def _metadata_has_missing_or_na(cls, metadata: Dict[str, Any], expected_fields: List[str]) -> bool:
        for field in expected_fields:
            value = str(dict(metadata or {}).get(field, "")).strip()
            if not value or value.lower() in {"n/a", "na", "none", "null", "unknown"}:
                return True
        return False

    @staticmethod
    def _fallback_executor_metadata(store: ExecutorMemoryStore, step, payload: Dict[str, Any]) -> Dict[str, str]:
        action = str(payload.get("proposed_action") or "").strip() or "<no direct action>"
        stage_name = str(payload.get("stage_name", "")).strip() or "stage"
        stage_subgoal = str(step.decision.stage_subgoals.get(stage_name, "")).strip()
        feedback = str(step.metadata.get("environment_feedback", "")).strip() or "No environment feedback recorded."
        state = str(step.metadata.get("state_trace_summary", "")).strip()
        generic = (
            f"{stage_name}: attempted {action}; feedback was '{feedback[:180]}'. "
            f"Use this with subgoal context '{stage_subgoal[:160]}' before repeating the pattern."
        )
        result: Dict[str, str] = {}
        for field in store.metadata_fields:
            lower = field.lower()
            if "action" in lower:
                value = f"Next action evidence: {action}; verify it matches visible state and subgoal before execution."
            elif "location" in lower or "searched" in lower or "route" in lower:
                value = f"Location/route evidence from state: {state[:220] or feedback[:220]}"
            elif "precondition" in lower or "assumption" in lower or "risk" in lower or "guard" in lower:
                value = f"Check preconditions against feedback before repeating: {feedback[:220]}"
            elif "progress" in lower or "delta" in lower or "handoff" in lower or "stop" in lower:
                value = f"Progress signal after this stage: reward={float(step.reward):.3f}, success={bool(step.success)}, feedback={feedback[:180]}"
            else:
                value = generic
            result[field] = value
        return result

    def _repair_missing_failure_metadata(
        self,
        trajectory: TrajectoryRecord,
        traj_steps_detail: List[Dict[str, Any]],
        failure_selections: Dict[str, int | None],
        failure_fields_by_agent: Dict[str, List[str]],
        fail_llm_meta: Dict[str, Dict[int, Dict[str, Any]]],
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        failure_requests: Dict[str, Dict[int, List[str]]] = {}
        for agent_name, step_idx in (failure_selections or {}).items():
            if step_idx is None:
                continue
            expected_fields = list(failure_fields_by_agent.get(agent_name, ["lesson", "error_analysis"]))
            existing = dict((fail_llm_meta or {}).get(agent_name, {}).get(step_idx, {}) or {})
            missing_fields = [
                field
                for field in expected_fields
                if self._metadata_has_missing_or_na({field: existing.get(field, "")}, [field])
            ]
            if missing_fields:
                failure_requests.setdefault(agent_name, {})[int(step_idx)] = missing_fields

        if not failure_requests:
            return fail_llm_meta

        repaired = self._call_failure_memory_repair(
            trajectory=trajectory,
            traj_steps_detail=traj_steps_detail,
            failure_requests=failure_requests,
        )
        merged: Dict[str, Dict[int, Dict[str, Any]]] = {
            agent_name: {
                int(step_idx): dict(meta or {})
                for step_idx, meta in dict(by_step or {}).items()
            }
            for agent_name, by_step in dict(fail_llm_meta or {}).items()
        }
        for agent_name, by_step in repaired.items():
            for step_idx, meta in by_step.items():
                expected_fields = failure_requests.get(agent_name, {}).get(int(step_idx), [])
                if not expected_fields:
                    continue
                current = dict(merged.setdefault(agent_name, {}).get(int(step_idx), {}) or {})
                for field in expected_fields:
                    value = str(dict(meta or {}).get(field, "")).strip()
                    if value and value.lower() not in {"n/a", "na", "none", "null", "unknown"}:
                        current[field] = value
                merged[agent_name][int(step_idx)] = current
        return merged

    @staticmethod
    def _memory_update_has_content(memory_update: Dict[str, Any] | None) -> bool:
        if not memory_update:
            return False
        if memory_update.get("skill_selected"):
            return True
        if any(v for v in dict(memory_update.get("executor_selected", {}) or {}).values()):
            return True
        if any(v is not None for v in dict(memory_update.get("failure_selections", {}) or {}).values()):
            return True
        return False

    def _mark_deferred_memory_update(self, trajectory: TrajectoryRecord, memory_update: Dict[str, Any] | None) -> None:
        updated = self._memory_update_has_content(memory_update)
        detail_lines = list(dict(trajectory.metadata or {}).get("detail_log_lines", []) or [])
        for idx, line in enumerate(detail_lines):
            if "memory_updated=False" in line and line.startswith(f"Task {trajectory.task.task_id} finished."):
                detail_lines[idx] = line.replace("memory_updated=False", f"memory_updated={updated}")
        skill_steps = list((memory_update or {}).get("skill_selected", []) or [])
        executor_steps = dict((memory_update or {}).get("executor_selected", {}) or {})
        failure_steps = {
            name: step_idx
            for name, step_idx in dict((memory_update or {}).get("failure_selections", {}) or {}).items()
            if step_idx is not None
        }
        detail_lines.append(
            "Deferred memory update applied after task execution "
            f"(memory_updated={updated} skill_steps={skill_steps} "
            f"executor_steps={executor_steps} failure_steps={failure_steps})"
        )
        trajectory.metadata["memory_updated"] = updated
        trajectory.metadata["detail_log_lines"] = detail_lines

    def _update_memories_from_trajectory(self, trajectory: TrajectoryRecord) -> None:
        memory_update = self._prepare_memory_update_from_trajectory(trajectory)
        self._apply_memory_update_from_trajectory(trajectory, memory_update)

    def _select_skill_memory_steps(self, trajectory: TrajectoryRecord) -> List[int]:
        """One step per skill: action-requiring → latest, non-action → earliest. Exploration score >= 0."""
        scores = compute_step_scores(
            trajectory.steps,
            trivial_feedbacks=self.task_manager.get_trivial_feedbacks(),
            failure_reward_threshold=self.task_manager.get_failure_reward_threshold(),
        )
        skill_steps: Dict[str, List] = {}
        for step, score in zip(trajectory.steps, scores):
            skill_name = step.decision.skill_name
            has_action = any(
                str(p.get("proposed_action", "")).strip()
                for p in step.executor_outputs.values() if isinstance(p, dict)
            )
            skill_steps.setdefault(skill_name, []).append({
                "step_index": step.step_index,
                "score": score,
                "has_action": has_action,
            })
        selected: List[int] = []
        for entries in skill_steps.values():
            eligible = [e for e in entries if e["score"] >= 0]
            if not eligible:
                eligible = [max(entries, key=lambda e: e["score"])]
            action_entries = [e for e in eligible if e["has_action"]]
            no_action_entries = [e for e in eligible if not e["has_action"]]
            if action_entries:
                selected.append(max(action_entries, key=lambda e: (e["score"], e["step_index"]))["step_index"])
            elif no_action_entries:
                selected.append(min(no_action_entries, key=lambda e: e["step_index"])["step_index"])
            else:
                selected.append(eligible[0]["step_index"])
        return sorted(selected)

    def _select_executor_memory_steps(self, trajectory: TrajectoryRecord, agent_name: str) -> List[int]:
        """Two steps per executor: one multi-agent + one solo, exploration score >= 0, prefer later."""
        scores = compute_step_scores(
            trajectory.steps,
            trivial_feedbacks=self.task_manager.get_trivial_feedbacks(),
            failure_reward_threshold=self.task_manager.get_failure_reward_threshold(),
        )
        candidates = []
        for step, score in zip(trajectory.steps, scores):
            if agent_name not in step.executor_outputs:
                continue
            if score < 0:
                continue
            participant_count = len(step.decision.participants)
            candidates.append({
                "step_index": step.step_index,
                "score": score,
                "is_multi": participant_count > 1,
                "is_solo": participant_count == 1,
            })
        if not candidates:
            # Fallback: pick the single best-scoring step this agent participated in
            all_agent_steps = [(step, sc) for step, sc in zip(trajectory.steps, scores) if agent_name in step.executor_outputs]
            if all_agent_steps:
                best = max(all_agent_steps, key=lambda x: x[1])
                return [best[0].step_index]
            return []
        multi = [c for c in candidates if c["is_multi"]]
        solo = [c for c in candidates if c["is_solo"]]
        selected: List[int] = []
        if multi:
            selected.append(max(multi, key=lambda c: (c["score"], c["step_index"]))["step_index"])
        if solo:
            selected.append(max(solo, key=lambda c: (c["score"], c["step_index"]))["step_index"])
        if not selected:
            selected.append(max(candidates, key=lambda c: (c["score"], c["step_index"]))["step_index"])
        if len(selected) < 2 and len(candidates) >= 2:
            remaining = [c for c in candidates if c["step_index"] not in selected]
            if remaining:
                selected.append(max(remaining, key=lambda c: (c["score"], c["step_index"]))["step_index"])
        return sorted(set(selected))

    def _select_failure_memory_steps(self, trajectory: TrajectoryRecord) -> Dict[str, int | None]:
        """Select one failure step per agent: find steps below failure threshold, pair with prev stage.
        Action stages → earliest, non-action → latest. One per agent."""
        agent_candidates: Dict[str, List[Dict]] = {agent.name: [] for agent in self.executors}
        effective_scores = compute_step_scores(
            trajectory.steps,
            trivial_feedbacks=self.task_manager.get_trivial_feedbacks(),
            failure_reward_threshold=self.task_manager.get_failure_reward_threshold(),
        )

        for i, (step, effective_score) in enumerate(zip(trajectory.steps, effective_scores)):
            if effective_score >= 0:
                continue
            # This step has reward below failure threshold
            for agent_name in step.executor_outputs:
                if agent_name not in agent_candidates:
                    continue
                has_action = bool(str(step.executor_outputs[agent_name].get("proposed_action", "")).strip())
                agent_candidates[agent_name].append({
                    "step_index": step.step_index,
                    "has_action": has_action,
                    "is_failure_step": True,
                })
                # Also add previous step as candidate
                if i > 0:
                    prev = trajectory.steps[i - 1]
                    if agent_name in prev.executor_outputs:
                        prev_has_action = bool(str(prev.executor_outputs[agent_name].get("proposed_action", "")).strip())
                        agent_candidates[agent_name].append({
                            "step_index": prev.step_index,
                            "has_action": prev_has_action,
                            "is_failure_step": False,
                        })

        result: Dict[str, int | None] = {}
        for agent_name, candidates in agent_candidates.items():
            if not candidates:
                result[agent_name] = None
                continue
            # Deduplicate by step_index
            seen = {}
            for c in candidates:
                idx = c["step_index"]
                if idx not in seen:
                    seen[idx] = c
            candidates = list(seen.values())
            # Action stages → earliest, non-action → latest
            action_cands = [c for c in candidates if c["has_action"]]
            noact_cands = [c for c in candidates if not c["has_action"]]
            if action_cands:
                result[agent_name] = min(action_cands, key=lambda c: c["step_index"])["step_index"]
            elif noact_cands:
                result[agent_name] = max(noact_cands, key=lambda c: c["step_index"])["step_index"]
            else:
                result[agent_name] = candidates[0]["step_index"]
        return result

    def _call_combined_memory_update(
        self,
        trajectory: TrajectoryRecord,
        traj_steps_detail: List[Dict[str, Any]],
        skill_selected: List[int],
        skill_fields: List[str],
        skill_fixed_fields: Dict[int, str],
        executor_selected: Dict[str, List[int]],
        exec_fields: Dict[str, List[str]],
        failure_selections: Dict[str, int | None] | None = None,
        failure_fields_by_agent: Dict[str, List[str]] | None = None,
    ) -> tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[int, Dict[str, Any]]], Dict[str, Dict[int, Dict[str, Any]]]]:
        """One LLM call for all memory updates. Returns (skill_metadata, executor_metadata, failure_metadata)."""
        has_any = skill_selected or any(v for v in executor_selected.values()) or any(v for v in (failure_selections or {}).values() if v is not None)
        if not has_any or not self.orchestrator:
            return {}, {}, {}
        try:
            system_prompt, user_prompt = build_combined_memory_update_prompt(
                task_description=trajectory.task.description,
                task_goal=trajectory.task.task_goal,
                task_success=trajectory.success,
                trajectory_steps=traj_steps_detail,
                skill_selected_indices=skill_selected,
                skill_metadata_fields=skill_fields,
                skill_fixed_fields=skill_fixed_fields,
                executor_selections=executor_selected,
                executor_metadata_fields=exec_fields,
                executor_failure_selections=failure_selections,
                executor_failure_metadata_fields=failure_fields_by_agent,
            )
            from ..llm.json_utils import parse_json_response
            response = self.orchestrator.client.chat.completions.create(
                model=self.orchestrator.config.model.name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=4000,
            )
            payload = parse_json_response(response.choices[0].message.content or "{}")

            def _fill_fields(meta: dict, expected_fields: list) -> dict:
                """Ensure all expected fields exist, fill missing with N/A."""
                result = {}
                for f in expected_fields:
                    result[f] = str(meta.get(f, "")).strip() or "N/A"
                return result

            skill_meta: Dict[int, Dict[str, Any]] = {}
            for entry in payload.get("skill_memory", []):
                if isinstance(entry, dict) and "step_index" in entry:
                    raw = dict(entry.get("memory_metadata", entry))
                    raw.pop("step_index", None)
                    step_index = int(entry["step_index"])
                    expected_fields = list(skill_fields)
                    fixed_field = str(skill_fixed_fields.get(step_index, "")).strip()
                    if fixed_field:
                        expected_fields.append(fixed_field)
                    skill_meta[step_index] = _fill_fields(raw, expected_fields)

            exec_meta: Dict[str, Dict[int, Dict[str, Any]]] = {}
            for agent_name, entries in (payload.get("executor_memory", {}) or {}).items():
                if not isinstance(entries, list):
                    # Maybe single dict instead of list
                    if isinstance(entries, dict) and "step_index" in entries:
                        entries = [entries]
                    else:
                        continue
                agent_map: Dict[int, Dict[str, Any]] = {}
                for entry in entries:
                    if isinstance(entry, dict) and "step_index" in entry:
                        raw = dict(entry.get("memory_metadata", entry))
                        raw.pop("step_index", None)
                        agent_fields = exec_fields.get(agent_name, ["lesson"])
                        agent_map[int(entry["step_index"])] = _fill_fields(raw, agent_fields)
                exec_meta[agent_name] = agent_map

            fail_meta: Dict[str, Dict[int, Dict[str, Any]]] = {}
            fail_payload = payload.get("failure_memory", {}) or {}
            for agent_name, entry in fail_payload.items():
                agent_failure_fields = (failure_fields_by_agent or {}).get(agent_name, ["lesson", "error_analysis"])
                if isinstance(entry, dict) and "step_index" in entry:
                    raw = dict(entry.get("memory_metadata", entry))
                    raw.pop("step_index", None)
                    fail_meta[agent_name] = {int(entry["step_index"]): _fill_fields(raw, agent_failure_fields)}
                elif isinstance(entry, list) and entry:
                    # Sometimes LLM returns list instead of dict
                    e = entry[0] if isinstance(entry[0], dict) else {}
                    if "step_index" in e:
                        raw = dict(e.get("memory_metadata", e))
                        raw.pop("step_index", None)
                        fail_meta[agent_name] = {int(e["step_index"]): _fill_fields(raw, agent_failure_fields)}

            return skill_meta, exec_meta, fail_meta
        except Exception as e:
            import sys
            print(f"[WARN] _call_combined_memory_update failed: {type(e).__name__}: {e}", file=sys.stderr)
            return {}, {}, {}

    def _call_failure_memory_repair(
        self,
        trajectory: TrajectoryRecord,
        traj_steps_detail: List[Dict[str, Any]],
        failure_requests: Dict[str, Dict[int, List[str]]],
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        if not failure_requests or not self.orchestrator:
            return {}
        try:
            system_prompt, user_prompt = build_failure_memory_repair_prompt(
                task_description=trajectory.task.description,
                task_goal=trajectory.task.task_goal,
                trajectory_steps=traj_steps_detail,
                failure_requests=failure_requests,
            )
            from ..llm.json_utils import parse_json_response
            response = self.orchestrator.client.chat.completions.create(
                model=self.orchestrator.config.model.name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1200,
            )
            payload = parse_json_response(response.choices[0].message.content or "{}")
            repaired: Dict[str, Dict[int, Dict[str, Any]]] = {}
            failure_payload = dict(payload.get("failure_memory", {}) or {})
            for agent_name, entries in failure_payload.items():
                if isinstance(entries, dict) and "step_index" in entries:
                    entries = [entries]
                if not isinstance(entries, list):
                    continue
                by_step: Dict[int, Dict[str, Any]] = {}
                for entry in entries:
                    if not isinstance(entry, dict) or "step_index" not in entry:
                        continue
                    step_idx = int(entry["step_index"])
                    if step_idx not in failure_requests.get(agent_name, {}):
                        continue
                    raw = dict(entry.get("memory_metadata", entry) or {})
                    raw.pop("step_index", None)
                    expected_fields = failure_requests.get(agent_name, {}).get(step_idx, [])
                    cleaned = {
                        field: str(raw.get(field, "")).strip()
                        for field in expected_fields
                        if str(raw.get(field, "")).strip()
                        and str(raw.get(field, "")).strip().lower() not in {"n/a", "na", "none", "null", "unknown"}
                    }
                    if cleaned:
                        by_step[step_idx] = cleaned
                if by_step:
                    repaired[str(agent_name)] = by_step
            return repaired
        except Exception as e:
            import sys
            print(f"[WARN] _call_failure_memory_repair failed: {type(e).__name__}: {e}", file=sys.stderr)
            return {}

    @staticmethod
    def _skill_memory_query_metadata(skill_spec: dict[str, object] | None, skill_name: str) -> Dict[str, str]:
        metadata = dict((skill_spec or {}).get("metadata", {}))
        selection_metadata = dict((skill_spec or {}).get("selection_metadata", {}))
        fixed_field = str(metadata.get("fixed_skill_memory_field", "")).strip()
        if not fixed_field:
            return {}
        fixed_value = str(selection_metadata.get("description", "")).strip() or str(skill_name).strip()
        return {fixed_field: fixed_value}

    def _skill_memory_fixed_fields_by_step(self, trajectory: TrajectoryRecord, step_indexes: List[int]) -> Dict[int, str]:
        result: Dict[int, str] = {}
        by_index = {step.step_index: step for step in trajectory.steps}
        for step_index in step_indexes:
            step = by_index.get(step_index)
            if step is None:
                continue
            spec = self.skill_library.get(step.decision.skill_name)
            if spec is None:
                continue
            fixed_field = str(dict(spec.metadata or {}).get("fixed_skill_memory_field", "")).strip()
            if fixed_field:
                self.skill_memory.register_fixed_field(fixed_field)
                result[step_index] = fixed_field
        return result

    @staticmethod
    def _serialize_step_for_memory(step) -> Dict[str, Any]:
        stages = []
        for agent_name, payload in step.executor_outputs.items():
            if not isinstance(payload, dict):
                continue
            stage_name = str(payload.get("stage_name", ""))
            stages.append({
                "stage_name": stage_name,
                "agent": agent_name,
                "output": str(payload.get("output", "")).strip()[:200],
                "proposed_action": str(payload.get("proposed_action", "")).strip(),
                "special_notes": [str(n).strip() for n in (payload.get("special_notes", []) or []) if str(n).strip()],
                "stage_subgoal": str(step.decision.stage_subgoals.get(stage_name, "")),
            })
        return {
            "skill_name": step.decision.skill_name,
            "subgoal": step.decision.subgoal,
            "assignment": dict(step.decision.assignment),
            "stages": stages,
            "environment_feedback": str(step.metadata.get("environment_feedback", "No environment action in this step.")),
        }

    def _update_skill_golden_examples(self) -> None:
        best_ids = self.skill_memory.best_entry_ids_by_skill()
        for skill_name, spec in self.skill_library.skills.items():
            best_id = best_ids.get(skill_name, "")
            if best_id:
                spec.selection_metadata["golden_example_id"] = best_id

    def _clear_memories_for_next_round(self, persist: bool = True) -> None:
        if self.enable_skill_memory:
            self.skill_memory.clear_entries_keep_structure()
            if persist:
                self.skill_memory.persist_full_rewrite(reason="clear_for_next_round")
        if self.enable_executor_memory:
            for store in self.executor_memories.values():
                store.clear_entries_keep_structure()
                if persist:
                    store.persist_full_rewrite(reason="clear_for_next_round")

    def _simple_write(self, text: str = "") -> None:
        if self.simple_logger is not None:
            self.simple_logger.write_line(text)

    def _detail_write(self, text: str = "") -> None:
        if self.detail_logger is not None:
            self.detail_logger.write_line(text)

    def _write_blank(self) -> None:
        # Atomic write to both loggers to prevent interleaving
        simple_lines = [""]
        detail_lines = [""]
        if self.simple_logger is not None:
            self.simple_logger.write_lines(simple_lines)
        if self.detail_logger is not None:
            self.detail_logger.write_lines(detail_lines)

    def _write_pair(self, simple: str, detail: str | None = None) -> None:
        # Atomic multi-line write to both loggers to prevent interleaving from concurrent tasks
        simple_lines = [simple]
        detail_text = "" if detail is None else detail
        detail_lines = [f"{simple} ({detail_text})"]
        
        if self.simple_logger is not None:
            self.simple_logger.write_lines(simple_lines)
        if self.detail_logger is not None:
            self.detail_logger.write_lines(detail_lines)

    def _write_block(self, simple_lines: list[str], detail_lines: list[str] | None = None) -> None:
        """Atomically write a block of lines to prevent interleaving from concurrent tasks.
        
        This ensures that all lines in a task step are written together without
        other tasks' logs appearing in between.
        """
        if detail_lines is None:
            detail_lines = simple_lines
        if self.simple_logger is not None:
            self.simple_logger.write_lines(simple_lines)
        if self.detail_logger is not None:
            self.detail_logger.write_lines(detail_lines)

    def _active_task_detail_lines(self) -> list[str]:
        stats_local = getattr(self, "_task_stats_local", None)
        active = getattr(stats_local, "active", None)
        if isinstance(active, dict):
            return list(active.get("detail_lines", []))
        task_log_local = getattr(self, "_task_log_local", None)
        active = getattr(task_log_local, "active", None)
        if isinstance(active, dict):
            return list(active.get("detail_lines", []))
        return []

    def _append_task_detail_lines(self, detail_lines: list[str]) -> None:
        stats_local = getattr(self, "_task_stats_local", None)
        active = getattr(stats_local, "active", None)
        if isinstance(active, dict):
            active.setdefault("detail_lines", []).extend(detail_lines)
            return
        task_log_local = getattr(self, "_task_log_local", None)
        active = getattr(task_log_local, "active", None)
        if isinstance(active, dict):
            active.setdefault("detail_lines", []).extend(detail_lines)

    def _prepare_round_partial_files(self, round_dir: Path) -> None:
        for path in [
            round_dir / "round_data.json",
            round_dir / "sampled_tasks.json",
            round_dir / "progress.partial.json",
            round_dir / "trajectories.partial.json",
            round_dir / "task_trajectories.jsonl",
            round_dir / "counterfactual_records.fresh.json",
            round_dir / "counterfactual_records.fresh.jsonl",
            round_dir / "counterfactual_records.selected.json",
            round_dir / "counterfactual_records.selected.jsonl",
        ]:
            if path.exists():
                path.unlink()
        for path in round_dir.glob("executor_memory_*.json"):
            path.unlink()
        for path in round_dir.glob("executor_memory_structure_*.json"):
            path.unlink()
        tasks_dir = ensure_dir(round_dir / "tasks")
        counterfactuals_dir = ensure_dir(round_dir / "counterfactuals")
        for target_dir in [tasks_dir, counterfactuals_dir]:
            for path in target_dir.glob("*.json"):
                path.unlink()

    def _persist_executor_memory_snapshots(self, output_dir: Path) -> None:
        lock = getattr(self, "_memory_snapshot_lock", None)
        if lock is None:
            self._persist_executor_memory_snapshots_unlocked(output_dir)
            return
        with lock:
            self._persist_executor_memory_snapshots_unlocked(output_dir)

    def _persist_executor_memory_snapshots_unlocked(self, output_dir: Path) -> None:
        for name, store in self.executor_memories.items():
            dump_json(output_dir / f"executor_memory_{name}.json", store.content_snapshot())
            dump_json(output_dir / f"executor_memory_structure_{name}.json", store.structure_snapshot())

    def _restore_executor_memories_from_output_dir(self, output_dir: Path) -> None:
        for name, store in self.executor_memories.items():
            snapshot = load_json(output_dir / f"executor_memory_{name}.json", default=None)
            structure_snapshot = load_json(output_dir / f"executor_memory_structure_{name}.json", default=None)
            if snapshot is not None and structure_snapshot is not None:
                merged_snapshot = dict(snapshot)
                merged_snapshot.update(dict(structure_snapshot))
                store.load_snapshot(merged_snapshot)
                continue
            store.load_snapshot(snapshot)

    def _load_runtime_memories_from_storage(self) -> None:
        if self.enable_skill_memory:
            self.skill_memory.load_from_storage()
        if self.enable_executor_memory:
            for store in self.executor_memories.values():
                store.load_from_storage()

    def _persist_round_data(
        self,
        target_path: Path,
        sampled_tasks: List[TaskRecord],
        trajectories: List[TrajectoryRecord],
        total_tasks: int,
    ) -> None:
        completed_task_ids = [trajectory.task.task_id for trajectory in trajectories]
        last_trajectory = trajectories[-1] if trajectories else None
        payload = {
            "sampled_tasks": [asdict(task) for task in sampled_tasks],
            "trajectories": [asdict(trajectory) for trajectory in trajectories],
            "completed_task_ids": completed_task_ids,
            "progress": {
                "completed_tasks": len(trajectories),
                "total_tasks": total_tasks,
                "last_trajectory_id": last_trajectory.trajectory_id if last_trajectory else "",
                "last_task_id": last_trajectory.task.task_id if last_trajectory else "",
                "last_task_success": bool(last_trajectory.success) if last_trajectory else False,
                "last_task_avg_reward": float(last_trajectory.average_reward) if last_trajectory else 0.0,
                "last_task_metric_label": self._metric_label(),
                "last_task_metric": float(last_trajectory.average_reward) if last_trajectory else 0.0,
            },
        }
        dump_json(target_path, payload)

    def _persist_round_memory_snapshots(self, round_dir: Path) -> None:
        dump_json(round_dir / "skill_library.json", self.skill_library.snapshot())
        dump_json(round_dir / "skill_memory.json", self.skill_memory.content_snapshot())
        dump_json(round_dir / "skill_memory_structure.json", self.skill_memory.structure_snapshot())
        self._persist_executor_memory_snapshots(round_dir)
        dump_json(round_dir / "profiles.json", self.profile_store.snapshot())

    _jsonl_lock = __import__("threading").Lock()

    @staticmethod
    def _append_jsonl(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if is_dataclass(payload):
            obj = asdict(payload)
        else:
            obj = payload
        with RoundRunner._jsonl_lock:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(obj, ensure_ascii=False))
                handle.write("\n")

    def _persist_evolution_outputs(self, round_dir: Path, artifacts: List[EvolutionArtifact]) -> None:
        """Write dedicated evolution output files so the next round can load them."""
        for artifact in artifacts:
            if not artifact.applied:
                continue
            if artifact.name == "skill_evolution":
                added = artifact.payload.get("added_skills", [])
                if added:
                    dump_json(round_dir / "evolved_skills.json", {
                        "round_dir": str(round_dir),
                        "added_skills": added,
                        "skill_library_snapshot": self.skill_library.snapshot(),
                    })
            elif artifact.name == "skill_memory_evolution":
                dump_json(round_dir / "evolved_skill_memory_structure.json", {
                    "round_dir": str(round_dir),
                    "new_fields": artifact.payload.get("new_fields", []),
                    "field_descriptions": artifact.payload.get("field_descriptions", {}),
                    "rationale": artifact.payload.get("rationale", ""),
                    "changed_entries": artifact.payload.get("changed_entries", 0),
                    "skill_memory_snapshot": self.skill_memory.snapshot(),
                })
            elif artifact.name == "executor_memory_evolution":
                dump_json(round_dir / "evolved_executor_memory_structure.json", {
                    "round_dir": str(round_dir),
                    "new_fields": artifact.payload.get("new_fields", []),
                    "field_descriptions": artifact.payload.get("field_descriptions", {}),
                    "rationale": artifact.payload.get("rationale", ""),
                    "changed_entries": artifact.payload.get("changed_entries", 0),
                    "executor_memories_snapshot": {
                        name: store.snapshot() for name, store in self.executor_memories.items()
                    },
                })

    def _persist_task_record_file(
        self,
        target_dir: Path,
        file_stem: str,
        trajectory: TrajectoryRecord,
        sequence_index: int,
        total_tasks: int,
        kind: str = "task",
    ) -> None:
        payload = {
            "task_id": trajectory.task.task_id,
            "trajectory_id": trajectory.trajectory_id,
            "success": trajectory.success,
            "step_count": len(trajectory.steps),
            "stage_count": self._trajectory_stage_count(trajectory),
            "elapsed_seconds": float(dict(trajectory.metadata or {}).get("elapsed_seconds", 0.0)),
            "prompt_tokens": int(dict(trajectory.metadata or {}).get("prompt_tokens", 0)),
            "completion_tokens": int(dict(trajectory.metadata or {}).get("completion_tokens", 0)),
            "detail_log_lines": list(
                dict(trajectory.metadata or {}).get("detail_log_lines", []) or self._active_task_detail_lines()
            ),
        }
        dump_json(target_dir / f"{file_stem}.json", payload)

    def _trajectory_stage_count(self, trajectory: TrajectoryRecord) -> int:
        total = 0
        for step in trajectory.steps:
            outputs = step.executor_outputs if isinstance(step.executor_outputs, dict) else {}
            if outputs:
                total += len(outputs)
                continue
            spec = getattr(self, "skill_library", None).get(step.decision.skill_name) if hasattr(getattr(self, "skill_library", None), "get") else None
            if spec is not None:
                total += max(1, int(spec.detail.get("stage_count", len(spec.detail.get("stages", [1])))))
            else:
                total += 1
        return total

    def _persist_task_trajectory(
        self,
        round_dir: Path,
        sampled_tasks: List[TaskRecord],
        trajectory: TrajectoryRecord,
        trajectories: List[TrajectoryRecord],
        task_index: int,
        total_tasks: int,
    ) -> None:
        self._persist_round_data(
            target_path=round_dir / "round_data.json",
            sampled_tasks=sampled_tasks,
            trajectories=trajectories,
            total_tasks=total_tasks,
        )
        self._persist_task_record_file(
            target_dir=round_dir / "tasks",
            file_stem=str(task_index - 1),
            trajectory=trajectory,
            sequence_index=task_index,
            total_tasks=total_tasks,
            kind="task",
        )
        self._write_pair(
            f"Task checkpoint saved: {task_index}/{total_tasks}",
            f"trajectory_id={trajectory.trajectory_id} task_id={trajectory.task.task_id}",
        )

    def _on_counterfactual_generated(self, round_dir: Path, record: CounterfactualRecord) -> None:
        dump_json(
            round_dir / "counterfactuals" / f"{record.task_id}__{record.removed_agent}_{record.removal_start_step}.json",
            record,
        )
        self._write_pair(
            "Counterfactual generated: "
            f"task={record.task_id} removed={record.removed_agent}@{record.removal_start_step} "
            f"success={record.success} {self._metric_label()}={record.average_reward:.3f}",
            f"source_trajectory={record.source_trajectory_id} "
            f"counterfactual_trajectory={record.counterfactual_trajectory_id} "
            f"status={record.metadata.get('status', '')}",
        )

    def _persist_selected_counterfactuals(
        self,
        round_dir: Path,
        counterfactual_records: List[CounterfactualRecord],
        counterfactual_source: str,
    ) -> None:
        self._write_pair(
            f"Counterfactual selected set: {len(counterfactual_records)}",
            f"source={counterfactual_source}",
        )

    @staticmethod
    def _console(message: str) -> None:
        print(f"[CoEvolve] {message}", flush=True)

    def _metric_label(self) -> str:
        return str(self.task_manager.get_metric_label()).strip() or "final_metric"

    def _install_usage_callback(self, client: Any) -> None:
        if hasattr(client, "usage_callback"):
            client.usage_callback = self._record_llm_usage

    def _record_llm_usage(self, response: Any) -> None:
        """Record token usage from LLM API response."""
        try:
            if hasattr(response, 'usage'):
                usage = response.usage
                completion_tokens = int(getattr(usage, 'completion_tokens', 0))
                prompt_tokens = int(getattr(usage, 'prompt_tokens', 0))
                
                with self._stats_lock:
                    self.total_completion_tokens += completion_tokens
                    self.total_prompt_tokens += prompt_tokens
                task_stats = getattr(self._task_stats_local, "active", None)
                if isinstance(task_stats, dict):
                    task_stats["completion_tokens"] = int(task_stats.get("completion_tokens", 0)) + completion_tokens
                    task_stats["prompt_tokens"] = int(task_stats.get("prompt_tokens", 0)) + prompt_tokens
        except Exception:
            pass  # Silently ignore if extraction fails

    @staticmethod
    def _short_text(value: object, limit: int = 160) -> str:
        text = str(value or "").replace("\n", " ").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _log_task_start(
        self,
        task: TaskRecord,
        round_id: int,
        trajectory_id: str,
        state_summary: str,
        removed_agent: str | None,
        removal_start_step: int | None,
        run_context: str | None = None,
    ) -> None:
        # Collect all log lines to ensure atomic write (no interleaving from concurrent tasks)
        simple_lines = []
        detail_lines = []
        
        # Helper to append a pair of lines
        def append_pair(simple: str, detail: str | None = None):
            simple_lines.append(simple)
            detail_text = "" if detail is None else detail
            detail_lines.append(f"{simple} ({detail_text})")
        
        context_text = run_context or f"round={round_id}"
        append_pair(
            f"Task {task.task_id} started.",
            f"{context_text} trajectory={trajectory_id} removed_agent={removed_agent or '<none>'} "
            f"removal_start_step={removal_start_step or '<none>'}",
        )
        append_pair(
            f"Task description: {task.description}",
            f"task_id={task.task_id} task_goal={task.task_goal}",
        )
        few_shot_key = str(task.metadata.get("few_shot_key", "")).strip()
        few_shot_count = len(task.metadata.get("few_shots", []) or [])
        append_pair(
            f"Few-shot key: {few_shot_key or '<none>'} count={few_shot_count}",
            f"few_shots={task.metadata.get('few_shots', [])}",
        )
        append_pair(f"Initial state: {self._short_text(state_summary, 240)}", f"state_full={self._short_text(state_summary, 800)}")
        simple_lines.append("")
        detail_lines.append("")
        
        # Write all lines atomically to prevent interleaving
        self._append_task_detail_lines(detail_lines)
        self._write_block(simple_lines, detail_lines)

    def _log_task_step(
        self,
        task: TaskRecord,
        step_index: int,
        state_summary: str,
        last_observation: str,
        selection: dict,
        decision: SkillDecision,
        skill_spec: dict,
        skill_memory_hits: list,
        assignment_result: dict,
        executor_outputs: dict,
        executor_memory_hits: dict,
        env_result: dict,
    ) -> None:
        # Collect all log lines to ensure atomic write (no interleaving from concurrent tasks)
        simple_lines = []
        detail_lines = []
        
        # Helper to append a pair of lines
        def append_pair(simple: str, detail: str | None = None):
            simple_lines.append(simple)
            detail_text = "" if detail is None else detail
            detail_lines.append(f"{simple} ({detail_text})")
        
        append_pair(f"Task {task.task_id} step {step_index}", f"task_id={task.task_id} step={step_index}")
        append_pair(
            f"State: {self._short_text(state_summary, 240)}",
            f"state_full={self._short_text(state_summary, 800)} last_observation={self._short_text(last_observation, 800)}",
        )
        append_pair(
            f"Skill: {decision.skill_name}",
            f"participants={decision.participants}",
        )
        append_pair(
            f"Skill description: {self._short_text(str(skill_spec.get('selection_metadata', {}).get('description', '')), 240)}",
            f"note={self._short_text(str(skill_spec.get('selection_metadata', {}).get('note', '')), 400)}",
        )
        append_pair(
            f"Skill memory: {self._memory_ids(skill_memory_hits)}",
            self._memory_details(skill_memory_hits),
        )
        append_pair(f"Participants: {decision.participants}", f"assignment_status={assignment_result.get('model_status', 'unknown')}")
        append_pair(
            f"Assignment: {decision.assignment}",
            f"subgoal={self._short_text(decision.subgoal, 400)} stage_subgoals={self._short_text(decision.stage_subgoals, 800)}",
        )

        for agent_name, payload in executor_outputs.items():
            stage_name = payload.get("stage_name", "")
            summary = self._short_text(payload.get("output") or "", 240)
            append_pair(
                f"Executor output [{agent_name}/{stage_name}]: {summary}",
                f"model_status={payload.get('model_status', 'unknown')} proposed_action={payload.get('proposed_action', '')} "
                f"special_notes={payload.get('special_notes', [])} "
                f"output_full={self._short_text(payload.get('output') or '', 800)}",
            )
            if payload.get("proposed_action", ""):
                append_pair(
                    f"Executor action [{agent_name}]: {payload.get('proposed_action', '')}",
                    f"stage={stage_name}",
                )
            mem_hits = executor_memory_hits.get(agent_name, [])
            append_pair(
                f"Executor memory [{agent_name}]: {self._memory_ids(mem_hits)}",
                self._memory_details(mem_hits),
            )

        append_pair(
            "Environment result: "
            f"executed_action={env_result.get('executed_action', '') or '<none>'}, "
            f"reward={float(env_result.get('reward', 0.0)):.3f}, "
            f"success={bool(env_result.get('success', False))}, "
            f"done={bool(env_result.get('done', False))}, "
            f"interactions={int(env_result.get('interaction_count', 0))}",
            f"state_summary={self._short_text(env_result.get('state_summary', ''), 400)} "
            f"last_observation={self._short_text(env_result.get('last_observation', ''), 400)}",
        )
        append_pair(f"Observation: {self._short_text(env_result.get('last_observation', ''), 240)}", "env_observation")
        simple_lines.append("")
        detail_lines.append("")
        
        # Write all lines atomically to prevent interleaving
        self._append_task_detail_lines(detail_lines)
        self._write_block(simple_lines, detail_lines)

    def _log_task_end(self, task: TaskRecord, trajectory: TrajectoryRecord, memory_updated: bool) -> None:
        # Collect all log lines to ensure atomic write (no interleaving from concurrent tasks)
        simple_lines = []
        detail_lines = []
        metric_label = self._metric_label()
        elapsed = float(dict(trajectory.metadata or {}).get("elapsed_seconds", 0.0))
        prompt_tokens = int(dict(trajectory.metadata or {}).get("prompt_tokens", 0))
        completion_tokens = int(dict(trajectory.metadata or {}).get("completion_tokens", 0))
        
        simple_lines.append(
            f"Task {task.task_id} finished. success={trajectory.success} "
            f"{metric_label}={trajectory.average_reward:.3f} steps={len(trajectory.steps)} "
            f"time={elapsed:.2f}s prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}"
        )
        detail_lines.append(
            f"Task {task.task_id} finished. success={trajectory.success} "
            f"{metric_label}={trajectory.average_reward:.3f} steps={len(trajectory.steps)} "
            f"time={elapsed:.2f}s prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
            f"(trajectory_id={trajectory.trajectory_id} memory_updated={memory_updated})"
        )
        simple_lines.append("=" * 80)
        detail_lines.append("=" * 80 + " (separator)")
        simple_lines.append("")
        detail_lines.append("")
        
        # Write all lines atomically to prevent interleaving
        self._append_task_detail_lines(detail_lines)
        self._write_block(simple_lines, detail_lines)

    @staticmethod
    def _task_text(task: TaskRecord) -> str:
        return task.full_text()

    @staticmethod
    def _memory_ids(hits: list) -> str:
        ids = [str(item.get("entry_id", "")) for item in hits if item.get("entry_id")]
        return f"[{', '.join(ids)}]" if ids else "[]"

    @staticmethod
    def _participants_from_assignment(assignment: Dict[str, List[str]]) -> List[str]:
        seen: List[str] = []
        for names in assignment.values():
            for name in names:
                candidate = str(name).strip()
                if candidate and candidate not in seen:
                    seen.append(candidate)
        return seen

    @staticmethod
    def _assigned_stage_subgoal_text(decision: SkillDecision, agent_name: str) -> str:
        parts: List[str] = []
        for stage_name, names in decision.assignment.items():
            if agent_name not in names:
                continue
            stage_subgoal = str(decision.stage_subgoals.get(stage_name, "")).strip()
            if stage_subgoal and stage_subgoal not in parts:
                parts.append(stage_subgoal)
        return "\n".join(parts)

    @staticmethod
    def _fmt_float(value: object, default: float = 0.0) -> str:
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return f"{default:.3f}"

    def _memory_details(self, hits: list) -> str:
        if not hits:
            return "none"
        details = []
        for item in hits:
            meta = dict(item.get("_retrieval_meta", item.get("metadata", {})))
            entry_id = str(item.get("entry_id", ""))
            memory_score = meta.get("memory_score", meta.get("bias"))
            memory_score_scaled = meta.get("memory_score_scaled", meta.get("bias_scaled"))
            details.append(
                "id={entry_id} score={score} sim={sim} memory_score={memory_score} memory_score_scaled={memory_score_scaled} exploration_bonus={exploration_bonus} source={source}".format(
                    entry_id=entry_id or "<none>",
                    score=self._fmt_float(meta.get("score")),
                    sim=self._fmt_float(meta.get("similarity")),
                    memory_score=self._fmt_float(memory_score),
                    memory_score_scaled=self._fmt_float(memory_score_scaled),
                    exploration_bonus=self._fmt_float(meta.get("exploration_bonus")),
                    source=str(meta.get("source", "")) or "memory",
                )
            )
        return "; ".join(details)

    @staticmethod
    def _previous_step_summary(buffer: dict) -> dict:
        steps = list(buffer.get("steps", []))
        if not steps:
            return {"history": []}
        history = []
        latest_notes: List[str] = []
        def _short(text: object, limit: int = 200) -> str:
            raw = str(text or "").strip()
            if len(raw) <= limit:
                return raw
            return raw[: max(0, limit - 3)] + "..."
        for step in steps:
            step_summaries: List[dict] = []
            for payload in step.executor_outputs.values():
                for item in payload.get("special_notes", []) or []:
                    note = str(item).strip()
                    if note:
                        latest_notes.append(note)
                step_summaries.append(
                    {
                        "stage_name": str(payload.get("stage_name", "")),
                        "output": _short(payload.get("output", ""), 160),
                        "action": _short(payload.get("proposed_action", ""), 120),
                        "special_notes": [
                            _short(item, 120)
                            for item in (payload.get("special_notes", []) or [])
                            if str(item).strip()
                        ],
                    }
                )
            history.append(
                {
                    "step": step.step_index,
                    "skill_name": step.decision.skill_name,
                    "stage_outputs": step_summaries,
                    "environment_feedback": str(
                        step.metadata.get("environment_feedback", "No environment action in this step.")
                    ),
                    "env_interaction_count": int(step.metadata.get("env_interaction_count", 0)),
                    "success": bool(step.success),
                    "reward": float(step.reward),
                }
            )
        return {
            "history": history[-10:],
            "special_notes": latest_notes[-5:],
        }

    @staticmethod
    def _build_prompt_step_context(
        buffer: dict,
        state_summary: str,
        last_observation: str,
        current_step_index: int,
        max_steps: int,
        remaining_steps: int | None = None,
    ) -> PromptStepContext:
        previous_step_summary = RoundRunner._previous_step_summary(buffer)
        state_trace_summary = build_selection_state_trace_summary(
            current_state_summary=state_summary,
            recent_environment_feedback=last_observation,
            history_summary=previous_step_summary,
        )
        return PromptStepContext(
            current_state_summary=str(state_summary or "").strip(),
            recent_environment_feedback=str(last_observation or state_summary or "").strip(),
            state_trace_summary=state_trace_summary,
            history=list(previous_step_summary.get("history", [])),
            special_notes=list(previous_step_summary.get("special_notes", [])),
            current_step_index=current_step_index,
            max_steps=max_steps,
            remaining_steps=(
                max(0, int(remaining_steps))
                if remaining_steps is not None
                else (max(0, max_steps - current_step_index + 1) if max_steps > 0 and current_step_index > 0 else 0)
            ),
        )

    @staticmethod
    def _promote_execute_when_stuck(
        selection: dict,
        prompt_context: PromptStepContext,
    ) -> dict:
        if selection.get("skill_name") == "execute":
            return selection
        history = list(prompt_context.history)
        if not history:
            return selection
        last_skill = str(history[-1].get("skill_name", ""))
        last_interactions = int(history[-1].get("env_interaction_count", 0))
        if last_skill == "plan" and last_interactions == 0:
            updated = dict(selection)
            updated["skill_name"] = "execute"
            return updated
        return selection

    def _restore_state(self, best_state: BestRoundState) -> None:
        skill_library_payload = self.state_manager.load_round_payload(best_state.round_dir, "skill_library.json", default=None)
        skill_payload = self.state_manager.load_round_payload(best_state.round_dir, "skill_memory.json", default=None)
        skill_structure_payload = self.state_manager.load_round_payload(best_state.round_dir, "skill_memory_structure.json", default=None)
        executor_payload = self.state_manager.load_round_payload(best_state.round_dir, "executor_memories.json", default=None)
        profiles_payload = self.state_manager.load_round_payload(best_state.round_dir, "profiles.json", default=None)
        if skill_library_payload:
            self.skill_library = SkillLibraryStore.from_payload(skill_library_payload, fallback_skill_names=["plan", "execute", "reflect"])
        if (skill_payload or skill_structure_payload) and self.enable_skill_memory:
            merged_skill_payload = dict(skill_payload or {})
            if skill_structure_payload:
                merged_skill_payload.update(dict(skill_structure_payload))
            self.skill_memory.load_snapshot(merged_skill_payload)
        for name, store in self.executor_memories.items():
            snapshot = executor_payload.get(name) if isinstance(executor_payload, dict) else None
            if snapshot is None:
                snapshot = self.state_manager.load_round_payload(best_state.round_dir, f"executor_memory_{name}.json", default=None)
            structure_snapshot = self.state_manager.load_round_payload(
                best_state.round_dir,
                f"executor_memory_structure_{name}.json",
                default=None,
            )
            if snapshot is not None or structure_snapshot is not None:
                merged_snapshot = dict(snapshot or {})
                if structure_snapshot is not None:
                    merged_snapshot.update(dict(structure_snapshot))
                store.load_snapshot(merged_snapshot)
            else:
                store.load_snapshot(snapshot)
        if profiles_payload:
            self.profile_store.load_snapshot(profiles_payload)
        self.state_manager.best_counterfactual_records = self.state_manager.load_best_counterfactuals(best_state)

    def _rerun_counterfactual(self, source_trajectory: TrajectoryRecord, removed_agent: str, removal_start_step: int) -> TrajectoryRecord:
        self._write_pair(
            f"Counterfactual: rerunning task {source_trajectory.task.task_id} without {removed_agent} from step {removal_start_step}",
            f"source={source_trajectory.trajectory_id}",
        )
        # Each counterfactual gets its own fresh task manager (independent env instance)
        local_task_manager = self._create_fresh_task_manager()
        episode = local_task_manager.start_task(source_trajectory.task)
        initial_state = local_task_manager.build_initial_state(source_trajectory.task)
        buffer = self.recorder.start(task=source_trajectory.task, round_id=int(source_trajectory.metadata.get("round_id", 0)))
        buffer["trajectory_id"] = f"{buffer['trajectory_id']}__cf_{removed_agent}_{removal_start_step}"
        state_summary = episode.get("state_summary", initial_state["summary"])
        last_observation = episode.get("last_observation", state_summary)
        final_success = False

        cf_tag = f"[CF task={source_trajectory.task.task_id} -{removed_agent}@{removal_start_step}]"

        # Replay prefix: push env state forward with actual actions, record all steps in buffer.
        prefix_steps = source_trajectory.steps[: max(0, removal_start_step - 1)]
        for original_step in prefix_steps:
            chosen_action = str(original_step.metadata.get("chosen_action", "")).strip()
            # Always record the original step in the trajectory buffer
            replay_step = TrajectoryStep(
                step_index=original_step.step_index,
                decision=original_step.decision,
                executor_memory_hits=original_step.executor_memory_hits,
                executor_outputs=original_step.executor_outputs,
                reward=float(original_step.reward),
                success=bool(original_step.success),
                metadata={
                    **dict(original_step.metadata),
                    "counterfactual_replay": True,
                    "replayed_prefix": True,
                },
            )
            self.recorder.append_step(buffer, replay_step)

            if not chosen_action:
                # No environment interaction (plan/reflect only), just record
                self._write_pair(
                    f"{cf_tag} replay step {original_step.step_index}: <no action>",
                    f"skill={original_step.decision.skill_name}",
                )
                continue
            # Replay the action to advance environment state
            step_result = local_task_manager.step_task(episode, chosen_action)
            episode.update(step_result)
            state_summary = step_result["state_summary"]
            last_observation = step_result["last_observation"]
            final_success = bool(step_result["success"])
            self._write_pair(
                f"{cf_tag} replay step {original_step.step_index}: action={chosen_action} reward={step_result['reward']:.3f} done={step_result['done']}",
                f"observation={self._short_text(step_result.get('last_observation', ''), 400)}",
            )
            if step_result["done"]:
                trajectory = self.recorder.finalize(
                    buffer=buffer,
                    success=final_success,
                    average_reward=self.task_manager.compute_average_reward(buffer["steps"]),
                )
                trajectory.metadata["counterfactual_removed_agent"] = removed_agent
                trajectory.metadata["counterfactual_removal_start_step"] = removal_start_step
                self._write_pair(
                    f"{cf_tag} finished during replay at step {original_step.step_index}. success={final_success}",
                    f"trajectory_id={trajectory.trajectory_id}",
                )
                return trajectory

        replayed_actions = sum(1 for s in prefix_steps if str(s.metadata.get("chosen_action", "")).strip())
        self._write_pair(
            f"{cf_tag} prefix replay done ({len(prefix_steps)} steps, {replayed_actions} actions). Starting new execution from step {removal_start_step}.",
            f"available_executors={[a.name for a in self.executors if a.name != removed_agent]}",
        )
        total_steps = max(1, len(source_trajectory.steps))
        task_few_shots = list(source_trajectory.task.metadata.get("few_shots", []))
        total_stages_used = sum(
            max(1, len(getattr(step, "metadata", {}).get("workflow_trace", {}).get("stage_runs", [])))
            for step in buffer.get("steps", [])
        )
        for step_index in range(removal_start_step, total_steps + 1):
            available_executor_names = [
                agent.name
                for agent in self.executors
                if agent.name != removed_agent
            ]
            profile_snapshots = self.profile_store.all() if self.enable_profiles else []
            prompt_context = self._build_prompt_step_context(
                buffer=buffer,
                state_summary=state_summary,
                last_observation=last_observation,
                current_step_index=total_stages_used + 1,
                max_steps=total_steps,
                remaining_steps=max(0, total_steps - total_stages_used),
            )
            selection = self.orchestrator.select_skill(
                task_description=source_trajectory.task.description,
                task_goal=source_trajectory.task.task_goal,
                prompt_context=prompt_context,
                available_skills=self._available_skill_names(buffer=buffer),
                available_skill_metadata=self._available_skill_metadata_with_golden(buffer=buffer),
                few_shots=task_few_shots,
            )
            selection = self._promote_execute_when_stuck(
                selection=selection,
                prompt_context=prompt_context,
            )
            skill_spec = self._skill_spec_payload(selection["skill_name"]) or {}
            golden_example_id = str(dict(skill_spec.get("selection_metadata", {})).get("golden_example_id", ""))
            skill_detail_memory_hits = self.skill_memory.retrieve_for_skill(
                skill_name=selection["skill_name"],
                task_description=source_trajectory.task.description,
                task_goal=source_trajectory.task.task_goal,
                state_trace_summary=prompt_context.state_trace_summary,
                memory_metadata=self._skill_memory_query_metadata(skill_spec, selection["skill_name"]),
                recent_environment_feedback=prompt_context.recent_environment_feedback,
                subgoal=selection["subgoal"],
                top_k=self._skill_memory_top_k(skill_spec),
                golden_example_id=golden_example_id,
            ) if self.enable_skill_memory else []
            assignment_result = self.orchestrator.assign_skill(
                task_description=source_trajectory.task.description,
                task_goal=source_trajectory.task.task_goal,
                skill_subgoal=selection["subgoal"],
                prompt_context=prompt_context,
                selected_skill_name=selection["skill_name"],
                selected_skill_detail=dict(skill_spec.get("detail", {})),
                skill_memory_hits=skill_detail_memory_hits,
                executor_names=available_executor_names,
                profile_snapshots=profile_snapshots,
                executor_personas={agent.name: agent.config.profile.persona for agent in self.executors},
            )
            assignment = dict(assignment_result.get("assignment", {}))
            participants = self._participants_from_assignment(assignment)
            decision = SkillDecision(
                skill_name=selection["skill_name"],
                participants=participants,
                subgoal=selection["subgoal"],
                assignment=assignment,
                stage_subgoals=dict(assignment_result.get("stage_subgoals", {})),
            )

            selected = [agent for agent in self.executors if agent.name in decision.participants]
            stage_roles = {
                str(stage.get("stage_name", "")): str(stage.get("stage_role", ""))
                for stage in dict(skill_spec.get("detail", {})).get("stages", [])
            }
            assigned_parts: dict[str, List[str]] = {}
            for stage_name, names in decision.assignment.items():
                role = stage_roles.get(stage_name, "")
                for name in names:
                    assigned_parts.setdefault(name, []).append(f"{stage_name}:{role}" if role else stage_name)
            executor_memory_hits = {
                agent.name: self.executor_memories[agent.name].retrieve(
                    task_description=source_trajectory.task.description,
                    stage_subgoal=self._assigned_stage_subgoal_text(decision, agent.name),
                    subgoal=decision.subgoal,
                    state_trace_summary=prompt_context.state_trace_summary,
                    recent_environment_feedback=prompt_context.recent_environment_feedback,
                    memory_metadata={},
                    skill_name=decision.skill_name,
                    assigned_parts=assigned_parts.get(agent.name, []),
                    top_k=agent.config.memory.retrieval_top_k,
                ) if self.enable_executor_memory else []
                for agent in selected
            }
            executor_failure_hits = {
                agent.name: self.executor_memories[agent.name].retrieve_failure(
                    task_description=source_trajectory.task.description,
                    stage_subgoal=self._assigned_stage_subgoal_text(decision, agent.name),
                    subgoal=decision.subgoal,
                    state_trace_summary=prompt_context.state_trace_summary,
                    recent_environment_feedback=prompt_context.recent_environment_feedback,
                    memory_metadata={},
                ) if self.enable_executor_memory else None
                for agent in selected
            }
            for agent in selected:
                fail_hit = executor_failure_hits.get(agent.name)
                if fail_hit:
                    fail_hit["_is_failure"] = True
                    executor_memory_hits[agent.name].append(fail_hit)
            runtime_result = self.skill_runtime.run(
                decision=decision,
                task_description=source_trajectory.task.description,
                task_goal=source_trajectory.task.task_goal,
                prompt_context=prompt_context,
                selected_executors=selected,
                executor_memory_hits=executor_memory_hits,
                skill_spec=skill_spec,
                few_shots=task_few_shots,
                env_step_fn=lambda action, _tm=local_task_manager: _tm.step_task(episode, action),
                action_format_prompt=self.task_manager.get_action_format_prompt(),
                current_step_index=prompt_context.current_step_index,
                max_steps=total_steps,
                remaining_steps=prompt_context.remaining_steps,
            )
            step_result = runtime_result["env_result"]
            step = TrajectoryStep(
                step_index=step_index,
                decision=decision,
                executor_memory_hits=executor_memory_hits,
                executor_outputs=runtime_result["executor_outputs"],
                reward=float(step_result.get("total_reward", step_result["reward"])),
                success=bool(step_result["success"]),
                metadata={
                    "round_id": int(source_trajectory.metadata.get("round_id", 0)),
                    "chosen_action": runtime_result["chosen_action"],
                    "last_observation": last_observation,
                    "state_trace_summary": prompt_context.state_trace_summary,
                    "environment_feedback": (
                        str(step_result.get("last_observation", "")).strip()
                        if int(step_result.get("interaction_count", 0)) > 0
                        else "No environment action in this step."
                    ),
                    "score": float(step_result.get("score", (step_result.get("info", {}) or {}).get("score", 0.0))),
                    "raw_score": float(step_result.get("raw_score", (step_result.get("info", {}) or {}).get("raw_score", 0.0))),
                    "counterfactual_replay": True,
                    "replayed_prefix": False,
                    "env_interaction_count": step_result.get("interaction_count", 0),
                    "skill_detail_memory_hits": skill_detail_memory_hits,
                    "workflow_trace": runtime_result.get("workflow_trace", {}),
                    "removed_agent": removed_agent,
                    "removal_start_step": removal_start_step,
                },
            )
            self.recorder.append_step(buffer, step)
            # Step-by-step logging for counterfactual
            executor_outputs = runtime_result["executor_outputs"]
            self._write_pair(
                f"{cf_tag} step {step_index}: skill={selection['skill_name']} participants={decision.participants}",
                f"subgoal={self._short_text(decision.subgoal, 400)} assignment={decision.assignment}",
            )
            for agent_name, payload in executor_outputs.items():
                self._write_pair(
                    f"{cf_tag} step {step_index} [{agent_name}/{payload.get('stage_name', '')}]: "
                    f"action={payload.get('proposed_action', '') or '<none>'} "
                    f"output={self._short_text(payload.get('output') or '', 160)}",
                    f"model_status={payload.get('model_status', 'unknown')} special_notes={payload.get('special_notes', [])}",
                )
            self._write_pair(
                f"{cf_tag} step {step_index} env: "
                f"action={runtime_result.get('chosen_action', '') or '<none>'} "
                f"reward={float(step_result.get('reward', 0.0)):.3f} "
                f"success={bool(step_result.get('success', False))} "
                f"done={bool(step_result.get('done', False))}",
                f"observation={self._short_text(step_result.get('last_observation', ''), 400)}",
            )
            episode.update(
                {
                    "state_summary": step_result["state_summary"],
                    "last_observation": step_result["last_observation"],
                    "done": step_result["done"],
                    "score": step_result.get("score") or (step_result.get("info", {}) or {}).get("score", 0.0),
                    "raw_score": step_result.get("raw_score") or (step_result.get("info", {}) or {}).get("raw_score", 0.0),
                    "step_count": (step_result.get("info", {}) or {}).get("step_count", episode.get("step_count", 0)),
                    "stop_interaction": bool(step_result.get("stop_interaction", False)),
                }
            )
            state_summary = step_result["state_summary"]
            last_observation = step_result["last_observation"]
            final_success = bool(step_result["success"])
            total_stages_used += max(1, len(runtime_result.get("workflow_trace", {}).get("stage_runs", executor_outputs)))
            if step_result["done"] or step_result.get("stop_interaction", False):
                break

        average_reward = self.task_manager.compute_average_reward(buffer["steps"])
        trajectory = self.recorder.finalize(
            buffer=buffer,
            success=final_success,
            average_reward=average_reward,
        )
        trajectory.metadata["counterfactual_removed_agent"] = removed_agent
        trajectory.metadata["counterfactual_removal_start_step"] = removal_start_step
        self._write_pair(
            f"Counterfactual done: task={source_trajectory.task.task_id} removed={removed_agent}@{removal_start_step} success={final_success} steps={len(buffer['steps'])}",
            f"trajectory_id={trajectory.trajectory_id} {self._metric_label()}={average_reward:.3f}",
        )
        return trajectory

    def _run_test(
        self,
        tasks: List[TaskRecord],
        sample_size: int,
        max_steps_per_task: int,
        base_seed: int,
        best_state: BestRoundState,
        resume: bool = False,
    ) -> BestRoundState:
        test_dir = ensure_dir(self.output_dir / "test")
        test_data_path = test_dir / "test_data.json"
        ensure_dir(test_dir / "tasks")

        # Resume: load already-completed test trajectories
        trajectories: List[TrajectoryRecord] = []
        completed_task_ids: set = set()
        test_data = load_json(test_data_path, default=None) if resume else None
        is_resuming = resume and test_data is not None
        if is_resuming:
            partial = list(test_data.get("trajectories", []) or [])
            trajectories = [self.state_manager._build_trajectory_record(item) for item in partial]
            completed_task_ids = {item.task.task_id for item in trajectories}
            if trajectories:
                self._console(f"Test resume: {len(trajectories)} tasks already completed")

        self.detail_logger = PlainTextLogger(self.log_dir / "test.detail.log", overwrite=not is_resuming)
        self.simple_logger = PlainTextLogger(self.log_dir / "test.simple.log", overwrite=not is_resuming)

        if is_resuming:
            # _restore_state loads the best evolved structures before entering
            # test mode. Resume must then reload the live test memories that
            # were updated after each completed task.
            self._load_runtime_memories_from_storage()
        else:
            # Fresh test starts from evolved skills + best memory structures,
            # but not old memory entries.
            self._clear_memories_for_next_round(persist=False)
            if self.enable_skill_memory:
                self._console(f"Test: cleared skill memory entries, structure={self.skill_memory.metadata_fields}")
            for store in self.executor_memories.values():
                self._console(
                    f"Test: cleared executor memory for {store.agent_name}, "
                    f"success_structure={store.metadata_fields}, failure_structure={store.failure_metadata_fields}"
                )

        self._write_pair(
            f"Test {'resumed' if is_resuming else 'started'}. best_round={best_state.round_id} best_success_rate={best_state.average_success_rate:.3f}",
            f"sample_size={sample_size} max_steps={max_steps_per_task} seed={base_seed} "
            f"skill_library_size={len(self.skill_library)} skill_memory_entries={len(self.skill_memory.entries)} "
            f"completed={len(completed_task_ids)}",
        )
        self.task_manager = self._create_fresh_task_manager()
        if is_resuming and test_data is not None:
            sampled_tasks = [TaskRecord(**item) for item in list(test_data.get("sampled_tasks", []) or [])]
        else:
            sampled_tasks = self.task_manager.sample_tasks(tasks, sample_size=sample_size, seed=base_seed)
            self._persist_round_data(
                target_path=test_data_path,
                sampled_tasks=sampled_tasks,
                trajectories=[],
                total_tasks=len(sampled_tasks),
            )

        remaining_tasks = [t for t in sampled_tasks if t.task_id not in completed_task_ids]
        total_tasks = len(sampled_tasks)
        source_best_round_id = int(best_state.round_id or 0)
        test_run_metadata = {"source_best_round_id": source_best_round_id}

        if self.max_workers > 1 and len(remaining_tasks) > 1:
            import threading
            from concurrent.futures import ThreadPoolExecutor, as_completed
            memory_lock = threading.Lock()

            def run_and_update(task):
                trajectory = self._execute_task(
                    task=task,
                    round_id=source_best_round_id,
                    max_steps_per_task=max_steps_per_task,
                    update_memory=False,
                    run_label="test",
                    run_metadata=test_run_metadata,
                )
                memory_update = self._prepare_memory_update_from_trajectory(trajectory)
                with memory_lock:
                    self._apply_memory_update_from_trajectory(trajectory, memory_update)
                    self._mark_deferred_memory_update(trajectory, memory_update)
                    trajectories.append(trajectory)
                    self._persist_round_data(
                        target_path=test_data_path,
                        sampled_tasks=sampled_tasks,
                        trajectories=trajectories,
                        total_tasks=total_tasks,
                    )
                    self._persist_task_record_file(
                        target_dir=test_dir / "tasks",
                        file_stem=str(len(trajectories) - 1),
                        trajectory=trajectory,
                        sequence_index=len(trajectories),
                        total_tasks=total_tasks,
                        kind="test_task",
                    )
                    self._write_pair(
                        f"Test checkpoint: {len(trajectories)}/{total_tasks} task={task.task_id} success={trajectory.success}",
                        f"{self._metric_label()}={trajectory.average_reward:.3f} steps={len(trajectory.steps)}",
                    )
                return trajectory

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                task_iter = iter(remaining_tasks)
                futures = {}

                # seed initial batch
                for _ in range(min(self.max_workers, len(remaining_tasks))):
                    item = next(task_iter, None)
                    if item is None:
                        break
                    fut = pool.submit(run_and_update, item)
                    futures[fut] = item

                while futures:
                    for future in as_completed(list(futures.keys())):
                        future.result()
                        futures.pop(future, None)
                        next_item = next(task_iter, None)
                        if next_item is not None:
                            newf = pool.submit(run_and_update, next_item)
                            futures[newf] = next_item
        else:
            for task in remaining_tasks:
                trajectory = self._execute_task(
                    task=task,
                    round_id=source_best_round_id,
                    max_steps_per_task=max_steps_per_task,
                    update_memory=False,
                    run_label="test",
                    run_metadata=test_run_metadata,
                )
                memory_update = self._prepare_memory_update_from_trajectory(trajectory)
                self._apply_memory_update_from_trajectory(trajectory, memory_update)
                self._mark_deferred_memory_update(trajectory, memory_update)
                trajectories.append(trajectory)
                self._persist_round_data(
                    target_path=test_data_path,
                    sampled_tasks=sampled_tasks,
                    trajectories=trajectories,
                    total_tasks=total_tasks,
                )
                self._persist_task_record_file(
                    target_dir=test_dir / "tasks",
                    file_stem=str(len(trajectories) - 1),
                    trajectory=trajectory,
                    sequence_index=len(trajectories),
                    total_tasks=total_tasks,
                    kind="test_task",
                )
                self._write_pair(
                    f"Test checkpoint: {len(trajectories)}/{total_tasks} task={task.task_id} success={trajectory.success}",
                    f"{self._metric_label()}={trajectory.average_reward:.3f} steps={len(trajectory.steps)}",
                )

        metrics = self._build_round_metrics(round_id=0, trajectories=trajectories, best_state=best_state)
        metrics.metadata.update({"mode": "test", "source_best_round_id": source_best_round_id})
        dump_json(test_dir / "test_metrics.json", metrics)
        dump_json(
            test_dir / "summary.json",
            {
                "mode": "test",
                "source_best_round_id": source_best_round_id,
                "task_count": metrics.task_count,
                "success_count": metrics.success_count,
                "success_rate": metrics.average_success_rate,
                "metric_label": self._metric_label(),
                "metric_value": metrics.average_reward,
            },
        )
        success_count = sum(1 for t in trajectories if t.success)
        self._write_pair(
            f"Test finished. {success_count}/{len(trajectories)} success ({metrics.average_success_rate*100:.1f}%) {self._metric_label()}={metrics.average_reward:.3f}",
            f"metrics={metrics}",
        )
        self._console(f"Test finished. {success_count}/{len(trajectories)} success ({metrics.average_success_rate*100:.1f}%)")
        return best_state

    def _available_skill_names(self, buffer: dict | None = None) -> List[str]:
        names = [
            name
            for name in self.skill_library.names()
            if self._skill_available_for_selection(name, buffer=buffer)
        ]
        if names:
            return names
        return [name for name in self.skill_library.names() if self._skill_lifecycle_status(name) != "killed"] or ["plan"]

    def _available_skill_metadata(self, buffer: dict | None = None) -> List[dict[str, object]]:
        allowed = set(self._available_skill_names(buffer=buffer))
        specs = []
        for spec in self.skill_library.selection_specs():
            if str(spec.get("skill_name", "")) not in allowed:
                continue
            lifecycle = dict(spec.get("lifecycle", {}) or {})
            if lifecycle:
                lifecycle["current_task_uses"] = self._skill_use_count_in_buffer(
                    str(spec.get("skill_name", "")),
                    buffer=buffer,
                )
                lifecycle.setdefault("max_task_uses", self.skill_trial_max_uses_per_task)
                spec = dict(spec)
                spec["lifecycle"] = lifecycle
            specs.append(spec)
        return specs

    def _available_skill_metadata_with_golden(self, buffer: dict | None = None) -> List[dict[str, object]]:
        specs = self._available_skill_metadata(buffer=buffer)
        if not self.enable_skill_memory:
            return specs
        enriched: List[dict[str, object]] = []
        for spec in specs:
            payload = dict(spec)
            golden_id = str(payload.get("golden_example_id", ""))
            golden_entry = self.skill_memory.get_entry_by_id(golden_id) if golden_id else None
            payload["golden_example"] = golden_entry or {}
            enriched.append(payload)
        return enriched

    def _skill_lifecycle(self, skill_name: str) -> Dict[str, object]:
        spec = self.skill_library.get(skill_name)
        if spec is None:
            return {}
        lifecycle = dict(spec.metadata.get("lifecycle", {}) or {})
        if bool(spec.metadata.get("primitive", False)) or spec.source == "primitive":
            lifecycle.setdefault("status", "active")
            spec.metadata["lifecycle"] = lifecycle
            return lifecycle
        if not lifecycle:
            lifecycle = {
                "status": "probation",
                "trial_rounds": 0,
                "failed_promotion_rounds": 0,
                "last_task_count": 0,
                "last_success_count": 0,
                "last_success_rate": "N/A",
                "last_average_reward": "N/A",
            }
            spec.metadata["lifecycle"] = lifecycle
        return lifecycle

    def _skill_lifecycle_status(self, skill_name: str) -> str:
        lifecycle = self._skill_lifecycle(skill_name)
        if not lifecycle:
            return "unknown"
        return str(lifecycle.get("status", "active")).strip() or "active"

    def _skill_available_for_selection(self, skill_name: str, buffer: dict | None = None) -> bool:
        status = self._skill_lifecycle_status(skill_name)
        if status == "killed":
            return False
        if status != "probation":
            return True
        return self._skill_use_count_in_buffer(skill_name, buffer=buffer) < self.skill_trial_max_uses_per_task

    @staticmethod
    def _skill_use_count_in_buffer(skill_name: str, buffer: dict | None = None) -> int:
        steps = list((buffer or {}).get("steps", []) or [])
        return sum(1 for step in steps if getattr(getattr(step, "decision", None), "skill_name", "") == skill_name)

    def _skill_spec_payload(self, skill_name: str) -> dict[str, object] | None:
        spec = self.skill_library.get(skill_name)
        return asdict(spec) if spec else None

    @staticmethod
    def _skill_memory_top_k(skill_spec: dict[str, object] | None) -> int:
        detail = dict((skill_spec or {}).get("detail", {}))
        return int(detail.get("memory_top_k", 5))

    def _load_initial_skill_library(self, skills_path: str, fallback_skill_names: List[str]) -> SkillLibraryStore:
        if skills_path:
            payload = load_json(skills_path, default=None)
            if isinstance(payload, dict) and isinstance(payload.get("task_skills"), dict):
                task_payload = dict(payload["task_skills"]).get(self.task_manager_name)
                if not task_payload:
                    raise ValueError(
                        f"No initial skill library configured for task manager: {self.task_manager_name}"
                    )
                return SkillLibraryStore.from_payload(task_payload, fallback_skill_names=fallback_skill_names)
            return SkillLibraryStore.from_payload(payload, fallback_skill_names=fallback_skill_names)
        return SkillLibraryStore.from_payload(payload=None, fallback_skill_names=fallback_skill_names)

    def _evaluate_skill_trials(
        self,
        round_id: int,
        trajectories: List[TrajectoryRecord],
    ) -> EvolutionArtifact:
        probation_names = [
            name
            for name in self.skill_library.names()
            if self._skill_lifecycle_status(name) == "probation"
        ]
        if not probation_names:
            return EvolutionArtifact(
                name="skill_trial_evaluation",
                applied=False,
                summary="No probation skills to evaluate.",
                payload={},
            )

        decisions: Dict[str, Dict[str, object]] = {}
        removed: List[str] = []
        promoted: List[str] = []
        continued: List[str] = []

        for skill_name in probation_names:
            spec = self.skill_library.get(skill_name)
            if spec is None:
                continue
            matching = [
                trajectory
                for trajectory in trajectories
                if any(step.decision.skill_name == skill_name for step in trajectory.steps)
            ]
            task_count = len(matching)
            success_count = sum(1 for trajectory in matching if trajectory.success)
            step_count = sum(
                1
                for trajectory in matching
                for step in trajectory.steps
                if step.decision.skill_name == skill_name
            )
            average_reward = (
                sum(float(trajectory.average_reward) for trajectory in matching) / task_count
                if task_count
                else 0.0
            )
            success_rate = success_count / task_count if task_count else 0.0

            lifecycle = self._skill_lifecycle(skill_name)
            lifecycle["trial_rounds"] = int(lifecycle.get("trial_rounds", 0) or 0) + (1 if task_count else 0)
            lifecycle["last_task_count"] = task_count
            lifecycle["last_success_count"] = success_count
            lifecycle["last_success_rate"] = round(success_rate, 4) if task_count else "N/A"
            lifecycle["last_average_reward"] = round(average_reward, 4) if task_count else "N/A"
            lifecycle["last_step_count"] = step_count
            lifecycle["last_evaluated_round"] = round_id

            if task_count >= self.skill_trial_min_tasks and success_rate >= self.skill_trial_promote_success_rate:
                lifecycle["status"] = "active"
                lifecycle["promoted_round"] = round_id
                promoted.append(skill_name)
                decision = "promoted"
            elif task_count > 0:
                failed_rounds = int(lifecycle.get("failed_promotion_rounds", 0) or 0) + 1
                lifecycle["failed_promotion_rounds"] = failed_rounds
                if failed_rounds >= self.skill_trial_max_failed_rounds:
                    lifecycle["status"] = "killed"
                    lifecycle["killed_round"] = round_id
                    removed.append(skill_name)
                    decision = "killed"
                else:
                    lifecycle["status"] = "probation"
                    continued.append(skill_name)
                    decision = "continued"
            else:
                lifecycle["status"] = "probation"
                continued.append(skill_name)
                decision = "continued_no_usage"

            spec.metadata["lifecycle"] = lifecycle
            decisions[skill_name] = {
                "decision": decision,
                "task_count": task_count,
                "success_count": success_count,
                "success_rate": round(success_rate, 4) if task_count else "N/A",
                "average_reward": round(average_reward, 4) if task_count else "N/A",
                "step_count": step_count,
                "failed_promotion_rounds": lifecycle.get("failed_promotion_rounds", 0),
            }

        summary = (
            f"Skill trial evaluation: promoted={len(promoted)}, continued={len(continued)}, killed={len(removed)}."
        )
        self._write_pair("Skill trial evaluation finished.", summary)
        return EvolutionArtifact(
            name="skill_trial_evaluation",
            applied=bool(promoted or removed or continued),
            summary=summary,
            payload={
                "round_id": round_id,
                "trial_max_uses_per_task": self.skill_trial_max_uses_per_task,
                "min_tasks": self.skill_trial_min_tasks,
                "promote_success_rate": self.skill_trial_promote_success_rate,
                "max_failed_rounds": self.skill_trial_max_failed_rounds,
                "promoted": promoted,
                "continued": continued,
                "killed": removed,
                "decisions": decisions,
            },
        )

    def _run_evolution(
        self,
        success_trajectories: List[TrajectoryRecord],
        failed_trajectories: List[TrajectoryRecord],
        counterfactual_records: List,
    ) -> List[EvolutionArtifact]:
        artifacts: List[EvolutionArtifact] = []
        self._write_pair(
            f"Evolution started: {len(success_trajectories)} success, {len(failed_trajectories)} failed, {len(counterfactual_records)} counterfactual",
            f"skill_evo={self.enable_skill_evolution} skill_mem_evo={self.enable_skill_memory_evolution} exec_mem_evo={self.enable_executor_memory_evolution}",
        )

        # 1. Skill evolution
        self._write_pair("Evolution [skill_evolution] started.", "")
        if self.enable_skill_evolution:
            artifact = self.skill_evolution_engine.evolve(
                success_trajectories=success_trajectories,
                failed_trajectories=failed_trajectories,
                counterfactual_records=counterfactual_records,
                skill_library=self.skill_library.skills,
            )
            artifacts.append(artifact)
            self._write_pair(
                f"Evolution [skill_evolution] finished. applied={artifact.applied}",
                f"summary={self._short_text(artifact.summary, 400)}",
            )
        else:
            artifacts.append(self._disabled_artifact("skill_evolution"))
            self._write_pair("Evolution [skill_evolution] finished. disabled.", "")

        # 2. Skill memory evolution
        self._write_pair("Evolution [skill_memory_evolution] started.", "")
        if self.enable_skill_memory and self.enable_skill_memory_evolution:
            skill_memory_artifact = (
                self.skill_memory_evolution_engine.evolve(
                    success_trajectories=success_trajectories,
                    failed_trajectories=failed_trajectories,
                    counterfactual_records=counterfactual_records,
                    skill_memory_store=self.skill_memory,
                    skill_library=self.skill_library.skills,
                )
            )
            artifacts.append(skill_memory_artifact)
            if skill_memory_artifact.applied:
                self.skill_memory.persist_full_rewrite(
                    reason=f"skill_memory_structure_evolution_round:{len(success_trajectories)}:{len(failed_trajectories)}"
                )
            self._write_pair(
                f"Evolution [skill_memory_evolution] finished. applied={skill_memory_artifact.applied}",
                f"summary={self._short_text(skill_memory_artifact.summary, 400)}",
            )
        else:
            artifacts.append(self._disabled_artifact("skill_memory_evolution"))
            self._write_pair("Evolution [skill_memory_evolution] finished. disabled.", "")

        # 3. Executor memory evolution
        self._write_pair("Evolution [executor_memory_evolution] started.", "")
        if self.enable_executor_memory and self.enable_executor_memory_evolution:
            executor_memory_artifact = self.executor_memory_evolution_engine.evolve(
                success_trajectories=success_trajectories,
                failed_trajectories=failed_trajectories,
                counterfactual_records=counterfactual_records,
                executor_memories=self.executor_memories,
                skill_library=self.skill_library.skills,
            )
            artifacts.append(executor_memory_artifact)
            if executor_memory_artifact.applied:
                for name, store in self.executor_memories.items():
                    store.persist_full_rewrite(
                        reason=f"executor_memory_structure_evolution_round:{len(success_trajectories)}:{len(failed_trajectories)}"
                    )
            self._write_pair(
                f"Evolution [executor_memory_evolution] finished. applied={executor_memory_artifact.applied}",
                f"summary={self._short_text(executor_memory_artifact.summary, 400)}",
            )
        else:
            artifacts.append(self._disabled_artifact("executor_memory_evolution"))
            self._write_pair("Evolution [executor_memory_evolution] finished. disabled.", "")

        self._write_pair(f"Evolution all done. {sum(1 for a in artifacts if a.applied)}/{len(artifacts)} applied.", "")
        return artifacts

    @staticmethod
    def _disabled_artifact(name: str) -> EvolutionArtifact:
        return EvolutionArtifact(
            name=name,
            applied=False,
            summary=f"{name} is disabled by runtime configuration.",
            payload={},
        )

    def _resolve_evolution_context(
        self,
        best_state: BestRoundState,
        metrics: RoundMetrics,
        current_trajectories: List[TrajectoryRecord],
        fresh_counterfactuals: List,
    ) -> tuple[List[TrajectoryRecord], List[TrajectoryRecord], List, str, dict[str, object]]:
        if metrics.best_so_far or not best_state.round_dir:
            self.state_manager.best_counterfactual_records = list(fresh_counterfactuals)
            return (
                [item for item in current_trajectories if item.success],
                [item for item in current_trajectories if not item.success],
                list(fresh_counterfactuals),
                "current_round",
                {},
            )

        self._restore_state(best_state)
        best_trajectories = self.state_manager.load_best_trajectories(best_state)
        best_counterfactuals = self.state_manager.load_best_counterfactuals(best_state)
        return (
            [item for item in best_trajectories if item.success],
            [item for item in best_trajectories if not item.success],
            best_counterfactuals,
            "best_round",
            {
                "evolution_reference_round": best_state.round_id,
                "non_best_round_reverted": True,
                "non_best_round_success_rate": metrics.average_success_rate,
                "non_best_round_reward": metrics.average_reward,
            },
        )
