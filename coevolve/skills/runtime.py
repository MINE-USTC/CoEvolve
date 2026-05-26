from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from ..agents import ExecutorAgent
from ..schema import PromptStepContext, SkillDecision


class SkillRuntime:
    def run(
        self,
        decision: SkillDecision,
        task_description: str,
        task_goal: str,
        prompt_context: PromptStepContext,
        selected_executors: List[ExecutorAgent],
        executor_memory_hits: Dict[str, List[Dict[str, Any]]],
        skill_spec: Dict[str, Any] | None = None,
        few_shots: List[str] | None = None,
        env_step_fn: Callable[[str], Dict[str, Any]] | None = None,
        action_format_prompt: str = "",
        current_step_index: int = 0,
        max_steps: int = 0,
        remaining_steps: int | None = None,
    ) -> Dict[str, Any]:
        detail = dict((skill_spec or {}).get("detail", {}))
        selection_metadata = dict((skill_spec or {}).get("selection_metadata", {}))
        template_type = str(detail.get("template_type", "solo"))
        stages = list(detail.get("stages", []))
        executor_index = {agent.name: agent for agent in selected_executors}
        base_state_summary = str(prompt_context.current_state_summary or "")
        state_trace_summary = str(prompt_context.state_trace_summary or "")
        recent_environment_feedback = str(prompt_context.recent_environment_feedback or "")
        previous_skill_special_notes = list(prompt_context.special_notes)

        if not stages:
            return self._empty_result(
                skill_name=decision.skill_name,
                template_type=template_type,
                state_summary=base_state_summary,
                recent_environment_feedback=recent_environment_feedback,
            )

        if template_type == "solo" or template_type.startswith("sequential_"):
            stage_runs, recent_environment_feedback, env_interactions = self._run_sequential(
                decision=decision,
                task_description=task_description,
                task_goal=task_goal,
                stages=stages,
                selection_metadata=selection_metadata,
                executor_index=executor_index,
                executor_memory_hits=executor_memory_hits,
                state_trace_summary=state_trace_summary,
                recent_environment_feedback=recent_environment_feedback,
                few_shots=few_shots or [],
                previous_skill_special_notes=previous_skill_special_notes,
                env_step_fn=env_step_fn,
                action_format_prompt=action_format_prompt,
                current_step_index=current_step_index,
                max_steps=max_steps,
                remaining_steps=remaining_steps,
            )
        elif template_type.startswith("parallel_"):
            stage_runs, recent_environment_feedback, env_interactions = self._run_parallel(
                decision=decision,
                task_description=task_description,
                task_goal=task_goal,
                stages=stages,
                selection_metadata=selection_metadata,
                executor_index=executor_index,
                executor_memory_hits=executor_memory_hits,
                state_trace_summary=state_trace_summary,
                recent_environment_feedback=recent_environment_feedback,
                few_shots=few_shots or [],
                previous_skill_special_notes=previous_skill_special_notes,
                env_step_fn=env_step_fn,
                action_format_prompt=action_format_prompt,
                current_step_index=current_step_index,
                max_steps=max_steps,
                remaining_steps=remaining_steps,
            )
        else:
            raise ValueError(f"Unsupported skill template_type: {template_type}")

        executor_outputs = {
            run["agent_name"]: run["output"]
            for run in stage_runs
            if run.get("agent_name")
        }
        final_env = self._final_env_payload(
            state_summary=base_state_summary,
            recent_environment_feedback=recent_environment_feedback,
            env_interactions=env_interactions,
        )
        return {
            "executor_outputs": executor_outputs,
            "chosen_action": final_env.get("executed_action", ""),
            "skill_name": decision.skill_name,
            "env_result": final_env,
            "workflow_trace": {
                "skill_name": decision.skill_name,
                "base_skill": str((skill_spec or {}).get("base_skill", decision.skill_name)),
                "skill_source": str((skill_spec or {}).get("source", "primitive")),
                "template_type": template_type,
                "selection_metadata": selection_metadata,
                "stage_runs": stage_runs,
                "env_interactions": env_interactions,
            },
        }

    def _run_sequential(
        self,
        decision: SkillDecision,
        task_description: str,
        task_goal: str,
        stages: List[Dict[str, Any]],
        selection_metadata: Dict[str, Any],
        executor_index: Dict[str, ExecutorAgent],
        executor_memory_hits: Dict[str, List[Dict[str, Any]]],
        state_trace_summary: str,
        recent_environment_feedback: str,
        few_shots: List[str],
        previous_skill_special_notes: List[str],
        env_step_fn: Callable[[str], Dict[str, Any]] | None,
        action_format_prompt: str = "",
        current_step_index: int = 0,
        max_steps: int = 0,
        remaining_steps: int | None = None,
    ) -> tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
        stage_runs: List[Dict[str, Any]] = []
        env_interactions: List[Dict[str, Any]] = []
        previous_special_notes: List[str] = []
        current_feedback = str(recent_environment_feedback or "")
        incoming_special_notes = [
            str(item).strip()
            for item in previous_skill_special_notes
            if str(item).strip()
        ]
        total_stage_count = len(stages)
        base_remaining_stages = (
            max(0, int(remaining_steps))
            if remaining_steps is not None
            else (max(0, max_steps - current_step_index + 1) if max_steps > 0 and current_step_index > 0 else 0)
        )
        for stage_position, stage in enumerate(stages, start=1):
            stage_name = str(stage.get("stage_name", ""))
            stage_role = str(stage.get("stage_role", ""))
            stage_requires_action = self._stage_requires_action(stage)
            assigned_names = list(decision.assignment.get(stage_name, []))
            agent = executor_index.get(assigned_names[0]) if assigned_names else None
            if agent is None:
                continue
            stage_input = {
                "task_description": task_description,
                "task_goal": task_goal,
                "subgoal": decision.subgoal,
                "stage_subgoal": str(decision.stage_subgoals.get(stage_name, "")),
                "state_trace_summary": state_trace_summary,
                "recent_environment_feedback": current_feedback,
                "current_step_index": current_step_index + stage_position - 1 if current_step_index > 0 else 0,
                "max_steps": max_steps,
                "remaining_steps": max(0, base_remaining_stages - stage_position + 1),
                "stage_position": stage_position,
                "skill_stage_count": total_stage_count,
                "few_shots": few_shots,
                "previous_skill_special_notes": incoming_special_notes,
                "action_format_prompt": action_format_prompt,
            }
            output = agent.act(
                skill_name=decision.skill_name,
                stage_name=stage_name,
                stage_role=stage_role,
                stage_requires_action=stage_requires_action,
                skill_description=str(selection_metadata.get("description", "")),
                stage_prompt=str(stage.get("prompt_template", "")),
                stage_input=stage_input,
                memory_hits=executor_memory_hits.get(agent.name, []),
                previous_special_notes=list(previous_special_notes),
                few_shots=few_shots,
                action_format_prompt=action_format_prompt,
            )
            env_feedback, current_feedback = self._maybe_step_environment(
                stage_requires_action=stage_requires_action,
                output=output,
                current_feedback=current_feedback,
                env_step_fn=env_step_fn,
            )
            if env_feedback:
                env_interactions.append(env_feedback)
            stage_runs.append(
                {
                    "stage_name": stage_name,
                    "stage_role": stage_role,
                    "stage_requires_action": stage_requires_action,
                    "agent_name": agent.name,
                    "input": stage_input,
                    "output": output,
                    "env_feedback": env_feedback,
                }
            )
            previous_special_notes = [
                str(item).strip()
                for item in (output.get("special_notes", []) or [])
                if str(item).strip()
            ]
            # Update state_trace_summary so the next stage sees this stage's result
            state_trace_summary = self._append_stage_to_trace(
                state_trace_summary,
                stage_name=stage_name,
                output=output,
                env_feedback=env_feedback,
            )
            if env_feedback and env_feedback.get("done"):
                break
        return stage_runs, current_feedback, env_interactions

    def _run_parallel(
        self,
        decision: SkillDecision,
        task_description: str,
        task_goal: str,
        stages: List[Dict[str, Any]],
        selection_metadata: Dict[str, Any],
        executor_index: Dict[str, ExecutorAgent],
        executor_memory_hits: Dict[str, List[Dict[str, Any]]],
        state_trace_summary: str,
        recent_environment_feedback: str,
        few_shots: List[str],
        previous_skill_special_notes: List[str],
        env_step_fn: Callable[[str], Dict[str, Any]] | None,
        action_format_prompt: str = "",
        current_step_index: int = 0,
        max_steps: int = 0,
        remaining_steps: int | None = None,
    ) -> tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
        stage_runs: List[Dict[str, Any]] = []
        env_interactions: List[Dict[str, Any]] = []
        current_feedback = str(recent_environment_feedback or "")
        previous_special_notes: List[str] = []
        incoming_special_notes = [
            str(item).strip()
            for item in previous_skill_special_notes
            if str(item).strip()
        ]
        batch_feedback = current_feedback
        total_stage_count = len(stages)
        base_remaining_stages = (
            max(0, int(remaining_steps))
            if remaining_steps is not None
            else (max(0, max_steps - current_step_index + 1) if max_steps > 0 and current_step_index > 0 else 0)
        )
        for stage_position, stage in enumerate(stages, start=1):
            stage_name = str(stage.get("stage_name", ""))
            stage_role = str(stage.get("stage_role", ""))
            stage_requires_action = self._stage_requires_action(stage)
            assigned_names = list(decision.assignment.get(stage_name, []))
            agent = executor_index.get(assigned_names[0]) if assigned_names else None
            if agent is None:
                continue
            stage_input = {
                "task_description": task_description,
                "task_goal": task_goal,
                "subgoal": decision.subgoal,
                "stage_subgoal": str(decision.stage_subgoals.get(stage_name, "")),
                "state_trace_summary": state_trace_summary,
                "recent_environment_feedback": batch_feedback,
                "current_step_index": current_step_index + stage_position - 1 if current_step_index > 0 else 0,
                "max_steps": max_steps,
                "remaining_steps": max(0, base_remaining_stages - stage_position + 1),
                "stage_position": stage_position,
                "skill_stage_count": total_stage_count,
                "few_shots": few_shots,
                "discussion_mode": True,
                "previous_skill_special_notes": incoming_special_notes,
                "action_format_prompt": action_format_prompt,
            }
            output = agent.act(
                skill_name=decision.skill_name,
                stage_name=stage_name,
                stage_role=stage_role,
                stage_requires_action=stage_requires_action,
                skill_description=str(selection_metadata.get("description", "")),
                stage_prompt=str(stage.get("prompt_template", "")),
                stage_input=stage_input,
                memory_hits=executor_memory_hits.get(agent.name, []),
                previous_special_notes=list(previous_special_notes),
                few_shots=few_shots,
                action_format_prompt=action_format_prompt,
            )
            env_feedback, current_feedback = self._maybe_step_environment(
                stage_requires_action=stage_requires_action,
                output=output,
                current_feedback=current_feedback,
                env_step_fn=env_step_fn,
            )
            if env_feedback:
                env_interactions.append(env_feedback)
            stage_runs.append(
                {
                    "stage_name": stage_name,
                    "stage_role": stage_role,
                    "stage_requires_action": stage_requires_action,
                    "agent_name": agent.name,
                    "input": stage_input,
                    "output": output,
                    "env_feedback": env_feedback,
                }
            )
            previous_special_notes = [
                str(item).strip()
                for item in (output.get("special_notes", []) or [])
                if str(item).strip()
            ]
        # Parallel: append all stages to trace after all finish
        for run in stage_runs:
            state_trace_summary = self._append_stage_to_trace(
                state_trace_summary,
                stage_name=run["stage_name"],
                output=run["output"],
                env_feedback=run["env_feedback"],
            )
        return stage_runs, current_feedback, env_interactions

    @staticmethod
    def _append_stage_to_trace(
        state_trace_summary: str,
        stage_name: str,
        output: Dict[str, Any],
        env_feedback: Dict[str, Any] | None,
    ) -> str:
        parts = [f"stage={stage_name}"]
        out_text = str(output.get("output", "")).strip()
        if out_text:
            if len(out_text) > 180:
                out_text = out_text[:177] + "..."
            parts.append(f"output={out_text}")
        action = str(output.get("proposed_action", "")).strip()
        if action:
            parts.append(f"action={action}")
        notes = [str(n).strip() for n in (output.get("special_notes", []) or []) if str(n).strip()]
        if notes:
            parts.append(f"special_notes={json.dumps(notes, ensure_ascii=False)}")
        stage_line = "  - " + "; ".join(parts)
        feedback_line = ""
        if env_feedback:
            obs = str(env_feedback.get("last_observation", "")).strip()
            feedback_line = f"\n  environment_feedback: {obs}" if obs else ""
        else:
            feedback_line = "\n  environment_feedback: No environment action in this stage."
        return state_trace_summary + "\n" + stage_line + feedback_line

    @staticmethod
    def _maybe_step_environment(
        stage_requires_action: bool,
        output: Dict[str, Any],
        current_feedback: str,
        env_step_fn: Callable[[str], Dict[str, Any]] | None,
    ) -> tuple[Dict[str, Any] | None, str]:
        action = str(output.get("proposed_action", "")).strip()
        if not stage_requires_action or not action or env_step_fn is None:
            return None, current_feedback
        step_result = env_step_fn(action)
        updated_feedback = str(step_result["last_observation"] or "")
        env_feedback = {
            "executed_action": action,
            "reward": float(step_result["reward"]),
            "success": bool(step_result["success"]),
            "done": bool(step_result["done"]),
            "stop_interaction": bool(
                step_result.get("stop_interaction")
                or (step_result.get("info", {}) or {}).get("stop_interaction", False)
            ),
            "state_summary": step_result["state_summary"],
            "last_observation": step_result["last_observation"],
            "score": float((step_result.get("info", {}) or {}).get("score", 0.0)),
            "raw_score": float((step_result.get("info", {}) or {}).get("raw_score", 0.0)),
        }
        return env_feedback, updated_feedback

    @staticmethod
    def _stage_requires_action(stage: Dict[str, Any]) -> bool:
        if "requires_action" in stage:
            return bool(stage.get("requires_action"))
        stage_role = str(stage.get("stage_role", "")).strip().lower()
        if stage_role == "execute":
            return True
        prompt = str(stage.get("prompt_template", "")).lower()
        if "requires taking an environment action" in prompt:
            return True
        if "proposed_action must" in prompt and "empty string" not in prompt:
            return True
        return False

    @staticmethod
    def _final_env_payload(
        state_summary: str,
        recent_environment_feedback: str,
        env_interactions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if env_interactions:
            last = dict(env_interactions[-1])
            last["total_reward"] = sum(item["reward"] for item in env_interactions)
            last["interaction_count"] = len(env_interactions)
            return last
        return {
            "executed_action": "",
            "reward": 0.0,
            "total_reward": 0.0,
            "success": False,
            "done": False,
            "stop_interaction": False,
            "state_summary": state_summary,
            "last_observation": recent_environment_feedback,
            "interaction_count": 0,
        }

    @staticmethod
    def _empty_result(
        skill_name: str,
        template_type: str,
        state_summary: str,
        recent_environment_feedback: str,
    ) -> Dict[str, Any]:
        env_result = {
            "executed_action": "",
            "reward": 0.0,
            "total_reward": 0.0,
            "success": False,
            "done": False,
            "stop_interaction": False,
            "state_summary": state_summary,
            "last_observation": recent_environment_feedback,
            "interaction_count": 0,
        }
        return {
            "executor_outputs": {},
            "chosen_action": "",
            "skill_name": skill_name,
            "env_result": env_result,
            "workflow_trace": {
                "skill_name": skill_name,
                "template_type": template_type,
                "stages": [],
                "env_interactions": [],
                "note": "No stages were defined.",
            },
        }
