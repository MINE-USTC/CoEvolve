from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List

from openai import OpenAI

from ..llm.json_utils import parse_json_response
from ..prompts import (
    build_executor_prompt,
    build_skill_assignment_prompt,
    build_skill_selection_prompt,
    build_task_experience_prompt,
)
from ..schema import AgentConfig, PromptStepContext, SkillDecision


@dataclass
class BaseAgent:
    config: AgentConfig
    client: OpenAI
    runtime_state: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.config.name


@dataclass
class OrchestratorAgent(BaseAgent):
    def select_skill(
        self,
        task_description: str,
        task_goal: str,
        prompt_context: PromptStepContext,
        available_skills: List[str],
        available_skill_metadata: List[Dict[str, Any]] | None = None,
        few_shots: List[str] | None = None,
    ) -> Dict[str, str]:
        fallback_skill = available_skills[0] if available_skills else "plan"
        prompt = build_skill_selection_prompt(
            task_description=task_description,
            task_goal=task_goal,
            state_trace_summary=prompt_context.state_trace_summary,
            recent_environment_feedback=prompt_context.recent_environment_feedback,
            current_step_index=prompt_context.current_step_index,
            max_steps=prompt_context.max_steps,
            remaining_steps=prompt_context.remaining_steps,
            available_skills=available_skills,
            available_skill_metadata=available_skill_metadata or [],
            few_shots=few_shots or [],
        )
        try:
            selection_persona = str(self.config.profile.metadata.get("persona_selection", self.config.profile.persona))
            response = self.client.chat.completions.create(
                model=self.config.model.name,
                messages=[
                    {"role": "system", "content": selection_persona},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens,
                top_p=self.config.model.top_p,
            )
            payload = parse_json_response(response.choices[0].message.content or "{}")
            skill_name = str(payload.get("skill_name", fallback_skill))
            if skill_name not in available_skills:
                skill_name = fallback_skill
            return {
                "skill_name": skill_name,
                "subgoal": str(
                    payload.get(
                        "subgoal",
                        f"Advance task toward goal: {task_goal or prompt_context.state_trace_summary or task_description}",
                    )
                ),
            }
        except Exception as exc:
            del exc
            return {
                "skill_name": fallback_skill,
                "subgoal": f"Advance task toward goal: {task_goal or prompt_context.state_trace_summary or task_description}",
            }

    def assign_skill(
        self,
        task_description: str,
        task_goal: str,
        skill_subgoal: str,
        prompt_context: PromptStepContext,
        selected_skill_name: str,
        selected_skill_detail: Dict[str, Any],
        executor_names: List[str],
        skill_memory_hits: List[Dict[str, Any]] | None = None,
        profile_snapshots: List[Dict[str, Any]] | None = None,
        executor_personas: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        stages = list(selected_skill_detail.get("stages", []))
        prompt = build_skill_assignment_prompt(
            task_description=task_description,
            task_goal=task_goal,
            skill_subgoal=skill_subgoal,
            state_trace_summary=prompt_context.state_trace_summary,
            recent_environment_feedback=prompt_context.recent_environment_feedback,
            current_step_index=prompt_context.current_step_index,
            max_steps=prompt_context.max_steps,
            remaining_steps=prompt_context.remaining_steps,
            selected_skill_name=selected_skill_name,
            selected_skill_detail=selected_skill_detail,
            skill_memory_hits=skill_memory_hits or [],
            executor_names=executor_names,
            profile_snapshots=profile_snapshots or [],
            executor_personas=executor_personas or {},
        )
        try:
            assignment_persona = str(self.config.profile.metadata.get("persona_assignment", self.config.profile.persona))
            response = self.client.chat.completions.create(
                model=self.config.model.name,
                messages=[
                    {"role": "system", "content": assignment_persona},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens,
                top_p=self.config.model.top_p,
            )
            payload = parse_json_response(response.choices[0].message.content or "{}")
            assignment = self._normalize_assignment(
                assignment_payload=payload.get("assignment", {}),
                stages=stages,
                executor_names=executor_names,
            )
            stage_subgoals = self._normalize_stage_subgoals(
                stage_subgoals_payload=payload.get("stage_subgoals", {}),
                stages=stages,
                fallback_subgoal=skill_subgoal,
            )
            if not self._participants_from_assignment(assignment):
                assignment = self._fallback_assignment(stages=stages, executor_names=executor_names)
                stage_subgoals = self._fallback_stage_subgoals(stages=stages, fallback_subgoal=skill_subgoal)
            return {
                "assignment": assignment,
                "stage_subgoals": stage_subgoals,
                "model_status": "ok",
                "raw_response": (response.choices[0].message.content or "")[:1000],
            }
        except Exception as exc:
            assignment = self._fallback_assignment(stages=stages, executor_names=executor_names)
            return {
                "assignment": assignment,
                "stage_subgoals": self._fallback_stage_subgoals(stages=stages, fallback_subgoal=skill_subgoal),
                "model_status": "fallback",
                "model_error": repr(exc),
                "raw_response": "",
            }

    def summarize_task_experiences(
        self,
        task_description: str,
        trajectory_summary: List[Dict[str, Any]],
        task_success: bool,
        average_reward: float,
        executor_names: List[str],
        max_experiences: int = 3,
    ) -> Dict[str, Any]:
        prompt = build_task_experience_prompt(
            task_description=task_description,
            trajectory_summary=trajectory_summary,
            task_success=task_success,
            average_reward=average_reward,
            executor_names=executor_names,
            max_experiences=max_experiences,
        )
        valid_names = set(executor_names)
        try:
            reflection_persona = str(self.config.profile.metadata.get("persona_reflection", self.config.profile.persona))
            response = self.client.chat.completions.create(
                model=self.config.model.name,
                messages=[
                    {"role": "system", "content": reflection_persona},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens,
                top_p=self.config.model.top_p,
            )
            payload = parse_json_response(response.choices[0].message.content or "{}")
            raw_items = payload.get("experiences", [])
            normalized: List[Dict[str, Any]] = []
            if isinstance(raw_items, list):
                for item in raw_items:
                    if not isinstance(item, dict):
                        continue
                    raw_agents = item.get("agent_names", [])
                    if isinstance(raw_agents, str):
                        raw_agents = [raw_agents]
                    agent_names = [name for name in raw_agents if isinstance(name, str) and name in valid_names]
                    if not agent_names:
                        continue
                    raw_steps = item.get("related_step_indexes", [])
                    if isinstance(raw_steps, (int, float, str)):
                        raw_steps = [raw_steps]
                    related_steps: List[int] = []
                    if isinstance(raw_steps, list):
                        for value in raw_steps:
                            try:
                                idx = int(value)
                            except (TypeError, ValueError):
                                continue
                            if idx > 0 and idx not in related_steps:
                                related_steps.append(idx)
                    snippet = str(item.get("trajectory_snippet", "")).strip()
                    experience = str(item.get("experience", "")).strip()
                    if not snippet and not experience:
                        continue
                    normalized.append(
                        {
                            "agent_names": agent_names,
                            "related_step_indexes": related_steps,
                            "trajectory_snippet": snippet,
                            "experience": experience,
                        }
                    )
            return {
                "experiences": normalized[: max(0, int(max_experiences))],
                "model_status": "ok",
                "raw_response": (response.choices[0].message.content or "")[:2000],
            }
        except Exception as exc:
            return {
                "experiences": [],
                "model_status": "fallback",
                "model_error": repr(exc),
                "raw_response": "",
            }

    @staticmethod
    def _normalize_assignment(
        assignment_payload: Any,
        stages: List[Dict[str, Any]],
        executor_names: List[str],
    ) -> Dict[str, List[str]]:
        normalized: Dict[str, List[str]] = {}
        valid_names = set(executor_names)
        if not isinstance(assignment_payload, dict):
            return normalized
        for stage in stages:
            stage_name = str(stage.get("stage_name", ""))
            raw = assignment_payload.get(stage_name, [])
            if isinstance(raw, str):
                raw = [raw]
            if isinstance(raw, list):
                filtered = [name for name in raw if name in valid_names]
                if filtered:
                    normalized[stage_name] = filtered[:1]
        return normalized

    @staticmethod
    def _fallback_assignment(stages: List[Dict[str, Any]], executor_names: List[str]) -> Dict[str, List[str]]:
        if not executor_names:
            return {}
        assignment: Dict[str, List[str]] = {}
        for index, stage in enumerate(stages):
            stage_name = str(stage.get("stage_name", f"stage_{index + 1}"))
            assignment[stage_name] = [executor_names[index % len(executor_names)]]
        return assignment

    @staticmethod
    def _normalize_stage_subgoals(
        stage_subgoals_payload: Any,
        stages: List[Dict[str, Any]],
        fallback_subgoal: str,
    ) -> Dict[str, str]:
        if not isinstance(stage_subgoals_payload, dict):
            return OrchestratorAgent._fallback_stage_subgoals(stages=stages, fallback_subgoal=fallback_subgoal)
        normalized: Dict[str, str] = {}
        fallback_map = OrchestratorAgent._fallback_stage_subgoals(stages=stages, fallback_subgoal=fallback_subgoal)
        for stage in stages:
            stage_name = str(stage.get("stage_name", "")).strip()
            if not stage_name:
                continue
            value = str(stage_subgoals_payload.get(stage_name, "")).strip()
            normalized[stage_name] = value or fallback_map.get(stage_name, "")
        return normalized

    @staticmethod
    def _fallback_stage_subgoals(stages: List[Dict[str, Any]], fallback_subgoal: str) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        base = str(fallback_subgoal or "").strip() or "Advance the selected skill toward the task goal."
        for index, stage in enumerate(stages, start=1):
            stage_name = str(stage.get("stage_name", f"stage_{index}")).strip() or f"stage_{index}"
            stage_role = str(stage.get("stage_role", "")).strip()
            if stage_role:
                normalized[stage_name] = f"Advance the {stage_role} stage toward: {base}"
            else:
                normalized[stage_name] = f"Advance {stage_name} toward: {base}"
        return normalized

    @staticmethod
    def _participants_from_assignment(assignment: Dict[str, List[str]]) -> List[str]:
        seen: List[str] = []
        for names in assignment.values():
            for name in names:
                if name not in seen:
                    seen.append(name)
        return seen


@dataclass
class ExecutorAgent(BaseAgent):
    @staticmethod
    def _recover_payload_from_text(content: str, stage_requires_action: bool) -> Dict[str, Any] | None:
        """Best-effort recovery when a model returns almost-JSON with one broken field."""
        text = content.strip()
        if not text:
            return None

        proposed_action = ""
        match = re.search(r'"proposed_action"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
        if match:
            try:
                proposed_action = str(json.loads(f'"{match.group(1)}"')).strip()
            except Exception:
                proposed_action = match.group(1).strip()
        elif stage_requires_action:
            loose = re.search(r"proposed_action\s*[:：]\s*([^\n,}]+)", text, flags=re.IGNORECASE)
            if loose:
                proposed_action = loose.group(1).strip().strip('"').strip("'")

        if stage_requires_action and not proposed_action:
            return None

        return {
            "output": text[:800],
            "proposed_action": proposed_action,
            "special_notes": [],
        }

    def act(
        self,
        skill_name: str,
        stage_name: str,
        stage_role: str,
        stage_requires_action: bool,
        skill_description: str,
        stage_prompt: str,
        stage_input: Dict[str, Any],
        memory_hits: List[Dict[str, Any]] | None = None,
        previous_special_notes: List[str] | None = None,
        few_shots: List[str] | None = None,
        action_format_prompt: str = "",
    ) -> Dict[str, Any]:
        prompt = build_executor_prompt(
            agent_name=self.name,
            collaboration_style=self.config.profile.collaboration_style,
            skill_name=skill_name,
            stage_name=stage_name,
            stage_role=stage_role,
            stage_requires_action=stage_requires_action,
            skill_description=skill_description,
            stage_prompt=stage_prompt,
            stage_input=stage_input,
            memory_hits=memory_hits or [],
            previous_special_notes=previous_special_notes or [],
            few_shots=few_shots or [],
            action_format_prompt=action_format_prompt,
        )
        executor_system_prompt = (
            "You are an executor in a multi-agent system. At each step, the system's coordinator "
            "and planner assigns you one stage of a workflow skill. Your job is to faithfully "
            "complete the stage goal assigned by the planner. This may involve interacting with "
            "the environment or producing analysis that benefits the downstream stages. To keep "
            "the whole task in view, you may refer to the task description, the task goal, and "
            "the current stage goal, but you must stay focused on the current assigned stage. "
            f"{self.config.profile.persona}"
        )
        content = ""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model.name,
                messages=[
                    {"role": "system", "content": executor_system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens,
                top_p=self.config.model.top_p,
            )
            content = response.choices[0].message.content or "{}"
            model_status = "ok"
            model_error = ""
            try:
                payload = parse_json_response(content)
            except Exception as parse_exc:
                recovered = self._recover_payload_from_text(content, stage_requires_action)
                if recovered is None:
                    raise parse_exc
                payload = recovered
                model_status = "parse_recovered"
                model_error = repr(parse_exc)
            proposed_action = str(payload.get("proposed_action", "")).strip()
            if not stage_requires_action:
                proposed_action = ""
            result = {
                "agent_name": self.name,
                "stage_name": stage_name,
                "stage_role": stage_role,
                "output": str(payload.get("output", "")),
                "proposed_action": proposed_action,
                "special_notes": list(payload.get("special_notes", [])),
                "model_status": model_status,
                "raw_response": content[:1000],
            }
            if model_error:
                result["model_error"] = model_error
            return result
        except Exception as exc:
            import sys
            print(f"[EXECUTOR FALLBACK] {self.name}/{stage_name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return {
                "agent_name": self.name,
                "stage_name": stage_name,
                "stage_role": stage_role,
                "output": f"{self.name} fallback output for {stage_name}",
                "proposed_action": "",
                "special_notes": [],
                "model_status": "fallback",
                "model_error": repr(exc),
                "raw_response": content[:1000],
            }


def build_agents(
    orchestrator_config: AgentConfig,
    executor_configs: List[AgentConfig],
    client: OpenAI,
) -> tuple[OrchestratorAgent, List[ExecutorAgent]]:
    orchestrator = OrchestratorAgent(config=orchestrator_config, client=client)
    executors = [ExecutorAgent(config=item, client=client) for item in executor_configs]
    return orchestrator, executors
