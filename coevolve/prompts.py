from __future__ import annotations

import json
from typing import Any, Dict, List


def _short_text(value: Any, limit: int = 240) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _trim_text_preserving_lines(value: Any, limit: int = 520) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    kept: List[str] = []
    total = 0
    for line in text.splitlines():
        fragment = line.strip()
        if not fragment:
            continue
        projected = total + len(fragment) + (1 if kept else 0)
        if projected > limit and kept:
            break
        kept.append(fragment)
        total = projected
    if not kept:
        return _short_text(text, limit)
    joined = "\n".join(kept)
    if len(joined) < len(text):
        joined += "\n..."
    return joined


def _format_budget_status(
    current_step_index: int,
    max_steps: int,
    remaining_steps: int,
    stage_position: int | None = None,
    stage_count: int | None = None,
) -> str:
    lines = [
        f"- Stage budget: next stage {current_step_index}/{max_steps}; remaining stages: {remaining_steps}"
        if current_step_index > 0 and max_steps > 0
        else "- Stage budget: unavailable"
    ]
    if stage_position and stage_count:
        lines.append(f"- Current skill stage: {stage_position}/{stage_count}")
    if remaining_steps > 0 and stage_count and stage_count > remaining_steps:
        lines.append("- Warning: this skill has more stages than the remaining budget; choose/execute only if it can finish or make decisive progress.")
    return "\n".join(lines)


def _optional_section(title: str, content: str) -> str:
    text = str(content or "").strip()
    if not text:
        return ""
    return f"\n{title}:\n{text}\n"


def _format_few_shots(few_shots: List[str]) -> str:
    if not few_shots:
        return "No few-shot successful trajectories available."
    blocks: List[str] = []
    for index, item in enumerate(few_shots, start=1):
        blocks.append(f"【Few-shot {index}】\n{str(item)}")
    return "\n\n".join(blocks)


def _format_recent_feedback(feedback: str) -> str:
    text = _trim_text_preserving_lines(feedback, 520)
    return text or "No recent environment feedback available."


def _format_skill_memory_entry(
    entry: Dict[str, Any] | None,
    include_selected_skill: bool = False,
    include_state_trace_summary: bool = False,
) -> str:
    item = dict(entry or {})
    memory_metadata = dict(item.get("memory_metadata", {}) or {})
    del item, include_selected_skill, include_state_trace_summary
    return f"- Semantic metadata: {json.dumps(memory_metadata, ensure_ascii=False)}"


def build_selection_state_trace_summary(
    current_state_summary: str,
    recent_environment_feedback: str,
    history_summary: Dict[str, Any] | None,
) -> str:
    history = list((history_summary or {}).get("history", []))
    if not history:
        blocks = ["Step 0 {", f"  state: {_short_text(current_state_summary, 280) or 'No current state summary.'}"]
        observation = _short_text(recent_environment_feedback, 280)
        if observation:
            blocks.append(f"  observation: {observation}")
        blocks.append("}")
        return "\n".join(blocks)
    blocks: List[str] = []
    for item in history[-10:]:
        step_index = item.get("step", item.get("step_index", "?"))
        skill_name = str(item.get("skill_name", "")).strip() or "Unknown skill"
        lines = [f"Step {step_index} {{", f"  skill: {skill_name}"]
        for output in item.get("stage_outputs", []) or []:
            if not isinstance(output, dict):
                continue
            stage_name = str(output.get("stage_name", "")).strip() or "unknown_stage"
            stage_output = _short_text(output.get("output", ""), 180)
            action = _short_text(output.get("action", ""), 120)
            notes = [_short_text(note, 120) for note in output.get("special_notes", []) or [] if _short_text(note, 120)]
            entry_parts = [f"stage={stage_name}"]
            if stage_output:
                entry_parts.append(f"output={stage_output}")
            if action:
                entry_parts.append(f"action={action}")
            if notes:
                entry_parts.append(f"special_notes={json.dumps(notes, ensure_ascii=False)}")
            lines.append("  - " + "; ".join(entry_parts))
        env_feedback = _short_text(item.get("environment_feedback", ""), 220)
        if env_feedback:
            lines.append(f"  environment_feedback: {env_feedback}")
        lines.append("}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)

def _format_golden_example(example: Dict[str, Any] | None) -> str:
    if not example:
        return "No golden example available."
    return _format_skill_memory_entry(
        example,
        include_selected_skill=True,
        include_state_trace_summary=True,
    )


def _format_executor_memory_entry(entry: Dict[str, Any] | None) -> str:
    item = dict(entry or {})
    memory_metadata = dict(item.get("memory_metadata", {}) or {})
    del item
    return f"- Semantic metadata: {json.dumps(memory_metadata, ensure_ascii=False)}"


def _format_skill_cards(available_skill_metadata: List[Dict[str, Any]]) -> str:
    if not available_skill_metadata:
        return "No available skills."
    blocks: List[str] = []
    for item in available_skill_metadata:
        name = str(item.get("skill_name", "")).strip() or "Unknown skill"
        description = str(item.get("description", "")).strip() or "No description."
        note = str(item.get("note", "")).strip() or "No additional notes."
        tags = ", ".join(str(tag) for tag in item.get("selection_tags", []) if str(tag).strip()) or "No tags"
        agent_count = str(item.get("agent_count_hint", "")).strip() or "Unknown"
        stage_count = str(item.get("stage_count", "1")).strip()
        lifecycle = dict(item.get("lifecycle", {}) or {})
        lifecycle_lines: List[str] = []
        if lifecycle:
            status = str(lifecycle.get("status", "active")).strip() or "active"
            task_uses = int(lifecycle.get("current_task_uses", 0) or 0)
            max_task_uses = int(lifecycle.get("max_task_uses", 0) or 0)
            failed_rounds = int(lifecycle.get("failed_promotion_rounds", 0) or 0)
            trial_rounds = int(lifecycle.get("trial_rounds", 0) or 0)
            last_success_rate = lifecycle.get("last_success_rate", "N/A")
            last_task_count = lifecycle.get("last_task_count", "N/A")
            lifecycle_lines = [
                f"Lifecycle: {status}",
                f"Trial stats: current_task_uses={task_uses}/{max_task_uses or 'unlimited'}, "
                f"trial_rounds={trial_rounds}, failed_promotion_rounds={failed_rounds}, "
                f"last_success_rate={last_success_rate}, last_task_count={last_task_count}",
            ]
        golden = _format_golden_example(item.get("golden_example") if isinstance(item.get("golden_example"), dict) else {})
        blocks.append(
            "\n".join(
                [
                    f"[{name}]",
                    f"Description: {description}",
                    f"Note: {note}",
                    f"Stage count: {stage_count}",
                    f"Participant count hint: {agent_count}",
                    f"Tags: {tags}",
                    *lifecycle_lines,
                    "Golden example:",
                    golden,
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_existing_skill_metadata(existing_skill_metadata: List[Dict[str, Any]] | None) -> str:
    if not existing_skill_metadata:
        return "No existing skill metadata available."
    blocks: List[str] = []
    for item in existing_skill_metadata:
        name = str(item.get("skill_name", "")).strip() or "Unknown skill"
        base_skill = str(item.get("base_skill", "")).strip()
        description = _short_text(item.get("description", ""), 220) or "No description."
        note = _short_text(item.get("note", ""), 220) or "No additional notes."
        template_type = str(item.get("template_type", "")).strip() or "Unknown"
        stage_count = str(item.get("stage_count", "")).strip()
        fixed_field = str(item.get("fixed_skill_memory_field", "")).strip()
        phases = [str(phase).strip() for phase in item.get("phases", []) or [] if str(phase).strip()]
        stages = list(item.get("stages", []) or [])

        lines = [
            f"[{name}]",
            f"Base skill: {base_skill or 'N/A'}",
            f"Description: {description}",
            f"Note: {note}",
            f"Template type: {template_type}",
            f"Stage count: {stage_count or str(len(stages) or 1)}",
            f"Phases: {', '.join(phases) if phases else 'N/A'}",
            f"Dedicated skill-memory field: {fixed_field or 'N/A'}",
            "Stages:",
        ]
        if not stages:
            lines.append("  - No stage definition available.")
        for index, stage in enumerate(stages, start=1):
            if not isinstance(stage, dict):
                continue
            stage_name = str(stage.get("stage_name", "")).strip() or f"stage_{index}"
            stage_role = str(stage.get("stage_role", "")).strip() or "unknown_role"
            requires_action = stage.get("requires_action", "unknown")
            prompt_template = _short_text(stage.get("prompt_template", ""), 180) or "No prompt."
            lines.append(
                f"  - Stage {index}: name={stage_name}; role={stage_role}; "
                f"requires_action={requires_action}; prompt={prompt_template}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_selected_skill_detail(selected_skill_name: str, selected_skill_detail: Dict[str, Any]) -> str:
    detail = dict(selected_skill_detail or {})
    template_type = str(detail.get("template_type", "")).strip() or "Unknown"
    stages = list(detail.get("stages", []) or [])
    participant_count = int(detail.get("participant_count_hint", 0) or 0)
    if participant_count <= 0:
        participant_count = max(1, len(stages))
    lines = [
        f"- Skill name: {selected_skill_name}",
        f"- Participant count: {participant_count}",
        f"- Template type: {template_type}",
        "- Stage prompts:",
    ]
    if not stages:
        lines.append("  - No stage definition available.")
        return "\n".join(lines)
    for index, stage in enumerate(stages, start=1):
        stage_name = str(stage.get("stage_name", "")).strip() or f"stage_{index}"
        stage_role = str(stage.get("stage_role", "")).strip() or "unknown_role"
        prompt_template = _short_text(stage.get("prompt_template", ""), 280) or "No prompt."
        lines.append(f"  - Stage {index}: name={stage_name}; role={stage_role}; prompt={prompt_template}")
    return "\n".join(lines)


def _format_skill_memory_hits_for_assignment(skill_memory_hits: List[Dict[str, Any]]) -> str:
    if not skill_memory_hits:
        return "No relevant skill memories available."
    blocks: List[str] = []
    for index, item in enumerate(skill_memory_hits, start=1):
        metadata = dict(item.get("metadata", {}))
        title = f"【Memory {index}】"
        if str(metadata.get("source", "")).strip() == "golden_example":
            title = f"【Memory {index} / Golden Example】"
        blocks.append(
            "\n".join(
                [
                    title,
                    _format_skill_memory_entry(
                        item,
                        include_selected_skill=False,
                        include_state_trace_summary=False,
                    ),
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_skill_memory_hits_for_execution(skill_memory_hits: List[Dict[str, Any]]) -> str:
    if not skill_memory_hits:
        return "No selected skill memories available."
    blocks: List[str] = []
    for index, item in enumerate(skill_memory_hits, start=1):
        metadata = dict(item.get("metadata", {}))
        title = f"【Skill memory {index}】"
        if str(metadata.get("source", "")).strip() == "golden_example":
            title = f"【Skill memory {index} / Golden Example】"
        blocks.append(
            "\n".join(
                [
                    title,
                    _format_skill_memory_entry(
                        item,
                        include_selected_skill=False,
                        include_state_trace_summary=False,
                    ),
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_executor_memory_hits_for_execution(memory_hits: List[Dict[str, Any]]) -> str:
    if not memory_hits:
        return "No relevant executor memories available."
    blocks: List[str] = []
    for index, item in enumerate(memory_hits, start=1):
        is_failure = bool(item.get("_is_failure"))
        if is_failure:
            title = f"【FAILURE EXAMPLE {index} - Analyze what went wrong and AVOID repeating this mistake】"
        else:
            title = f"【Executor memory {index}】"
        blocks.append(
            "\n".join(
                [
                    title,
                    _format_executor_memory_entry(item),
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_executor_profiles_for_assignment(
    profile_snapshots: List[Dict[str, Any]],
    executor_personas: Dict[str, str] | None = None,
) -> str:
    personas = {str(name): str(text) for name, text in dict(executor_personas or {}).items()}
    snapshot_index = {
        str(item.get("agent_name", "")).strip(): dict(item)
        for item in profile_snapshots
        if str(item.get("agent_name", "")).strip()
    }
    executor_names = sorted(set(snapshot_index) | set(personas))
    if not executor_names:
        return "No executor profile information available."
    blocks: List[str] = []
    for name in executor_names:
        snapshot = snapshot_index.get(name, {})
        completed_tasks = int(snapshot.get("completed_tasks", 0) or 0)
        successful_tasks = int(snapshot.get("successful_tasks", 0) or 0)
        success_rate = successful_tasks / completed_tasks if completed_tasks else 0.0
        preferred_skill_stages = dict(snapshot.get("preferred_skill_stages", {}) or {})
        best_skill_stages = [
            pair for pair, _ in sorted(preferred_skill_stages.items(), key=lambda item: item[1], reverse=True)[:3]
        ]
        blocks.append(
            "\n".join(
                [
                    f"【{name}】",
                    f"Persona: {_short_text(personas.get(name, ''), 220) or 'N/A'}",
                    f"Collaboration style: {_short_text(snapshot.get('collaboration_style', ''), 220) or 'N/A'}",
                    f"Success rate: {success_rate:.3f} ({successful_tasks}/{completed_tasks})",
                    f"Tasks completed: {int(snapshot.get('completed_tasks', 0) or 0)}",
                    f"Best skill-stage pairs: {', '.join(best_skill_stages) if best_skill_stages else 'N/A'}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_skill_selection_prompt(
    task_description: str,
    task_goal: str,
    state_trace_summary: str,
    recent_environment_feedback: str,
    current_step_index: int,
    max_steps: int,
    remaining_steps: int,
    available_skills: List[str],
    available_skill_metadata: List[Dict[str, Any]],
    few_shots: List[str] | None = None,
) -> str:
    available_skill_names = ", ".join(available_skills) if available_skills else "No available skills"
    formatted_few_shots = _format_few_shots(few_shots or [])
    formatted_skills = _format_skill_cards(available_skill_metadata)
    budget_status = _format_budget_status(current_step_index, max_steps, remaining_steps)
    budget_section = _optional_section("Progress budget", budget_status)
    return f"""
Your job at this step is to choose the single most appropriate skill so that the task can keep making progress and eventually be completed.

You must make this decision by jointly considering the task description, the task goal, optional examples from similar tasks, the chronological state summary of prior skill usage, and the metadata of the available skills.

What is the current task:
{task_description}

What is the current goal:
{task_goal}

Examples from similar tasks (few-shot, optional reference):
{formatted_few_shots}

Current state summary / prior skill trace:
{state_trace_summary}

Current situation from the most recent environment feedback:
{_format_recent_feedback(recent_environment_feedback)}
{budget_section}

The currently available skills are:
{available_skill_names}

Below is the metadata for each available skill. You should rely especially on description, note, and golden example to judge when each skill should be used:
{formatted_skills}

Your responsibilities:
1. Select exactly one skill that is best for the next step.
2. Focus on the current task goal and the state-summary trace, not the full end-to-end plan.
3. Decide only which skill should be invoked next and what that skill should accomplish right now.
4. Do not assign executors at this step.
5. The selected subgoal may be moderately high-level when appropriate, as long as it clearly guides the next skill invocation toward task progress.

Return JSON with exactly these two keys:
- skill_name
- subgoal

Output requirements:
- `skill_name` must exactly match one available skill name.
- `subgoal` must be concrete, actionable, and scoped to the current single skill invocation.
- Return valid JSON only.
- Do not output markdown.
- Do not include extra keys.
""".strip()


def build_skill_assignment_prompt(
    task_description: str,
    task_goal: str,
    skill_subgoal: str,
    state_trace_summary: str,
    recent_environment_feedback: str,
    current_step_index: int,
    max_steps: int,
    remaining_steps: int,
    selected_skill_name: str,
    selected_skill_detail: Dict[str, Any],
    skill_memory_hits: List[Dict[str, Any]],
    executor_names: List[str],
    profile_snapshots: List[Dict[str, Any]],
    executor_personas: Dict[str, str] | None = None,
) -> str:
    formatted_skill_detail = _format_selected_skill_detail(selected_skill_name, selected_skill_detail)
    formatted_skill_memories = _format_skill_memory_hits_for_assignment(skill_memory_hits)
    formatted_executor_profiles = _format_executor_profiles_for_assignment(profile_snapshots, executor_personas)
    budget_status = _format_budget_status(current_step_index, max_steps, remaining_steps)
    budget_section = _optional_section("Progress budget", budget_status)
    return f"""
Your job at this step is to choose the most suitable executor for each stage of the already selected skill and define the stage-level subgoal for each stage so that the skill can be executed smoothly, effectively, and toward the intended outcome.

What is the current task:
{task_description}

What is the current goal:
{task_goal}

Current skill-level subgoal:
{skill_subgoal}

Prior skill trace / previous actions:
{state_trace_summary}

Current situation from the most recent environment feedback:
{_format_recent_feedback(recent_environment_feedback)}
{budget_section}

Selected skill:
{formatted_skill_detail}

Relevant skill memories:
{formatted_skill_memories}

Available executors:
{json.dumps(executor_names, ensure_ascii=False)}

Executor agent profiles:
{formatted_executor_profiles}

---

Output requirements:

Return JSON with keys:
- assignment
- stage_subgoals

Where:
- assignment: dict mapping each stage_name → list of executor names
- stage_subgoals: dict mapping each stage_name → one concrete stage-level subgoal string

Rules:
- Do not modify the selected skill.
- Do not introduce new stages.
- Do not include executors not in the provided list.
- Every stage in the selected skill must appear in both `assignment` and `stage_subgoals`.
- Return valid JSON only (no markdown, no extra keys).
""".strip()


def build_executor_prompt(
    agent_name: str,
    collaboration_style: str,
    skill_name: str,
    stage_name: str,
    stage_role: str,
    stage_requires_action: bool,
    skill_description: str,
    stage_prompt: str,
    stage_input: Dict[str, Any],
    memory_hits: List[Dict[str, Any]],
    previous_special_notes: List[str],
    few_shots: List[str] | None = None,
    action_format_prompt: str = "",
) -> str:
    task_description = str(stage_input.get("task_description", "")).strip() or "N/A"
    task_goal = str(stage_input.get("task_goal", "")).strip() or "N/A"
    skill_subgoal = str(stage_input.get("subgoal", "")).strip() or "N/A"
    stage_subgoal = str(stage_input.get("stage_subgoal", "")).strip() or "N/A"
    state_trace_summary = str(stage_input.get("state_trace_summary", "")).strip() or "No prior trace."
    recent_env_feedback = str(stage_input.get("recent_environment_feedback", "")).strip() or "No recent feedback."
    current_step_index = int(stage_input.get("current_step_index", 0) or 0)
    max_steps = int(stage_input.get("max_steps", 0) or 0)
    remaining_steps = int(stage_input.get("remaining_steps", 0) or 0)
    stage_position = int(stage_input.get("stage_position", 0) or 0)
    stage_count = int(stage_input.get("skill_stage_count", 0) or 0)
    prev_skill_notes = stage_input.get("previous_skill_special_notes", []) or []

    all_notes = list(previous_special_notes or []) + [str(n).strip() for n in prev_skill_notes if str(n).strip()]
    latest_special_note = all_notes[-1] if all_notes else "None"

    formatted_memories = _format_executor_memory_hits_for_execution(memory_hits)
    metadata_highlights = _extract_semantic_metadata_highlights(memory_hits)
    formatted_few_shots = _format_few_shots(few_shots or [])
    budget_status = _format_budget_status(
        current_step_index=current_step_index,
        max_steps=max_steps,
        remaining_steps=remaining_steps,
        stage_position=stage_position,
        stage_count=stage_count,
    )
    budget_section = _optional_section("Progress budget", budget_status)

    return f"""
You are executor "{agent_name}". The system planner has selected you to execute the [{stage_name}] stage of skill [{skill_name}].
This stage requires: {stage_prompt}

Task: {task_description}
Goal: {task_goal}
Skill-level subgoal: {skill_subgoal}
Your immediate focus: {stage_subgoal}
Requires action: {stage_requires_action}
{budget_section}

Current situation (state trace):
{state_trace_summary}

Previous executor context (pay close attention):
- Feedback after last action: {_format_recent_feedback(recent_env_feedback)}
- Last special note: {latest_special_note}
Look at the last step in the state trace above for the previous executor's output, action, and special notes. Use this to decide your next move.

Examples from similar tasks (few-shot, optional reference):
{formatted_few_shots}

Retrieved relevant memories:
{formatted_memories}
{metadata_highlights}
Pay attention to all non-empty semantic metadata fields above. Use them to avoid repeating mistakes and preserve successful patterns.

---

{action_format_prompt}

---

Notes:
- Follow the stage prompt faithfully as your primary instruction.
- Focus on your immediate subgoal.
- Use the state trace to avoid repeating failed actions. If stuck in a loop, propose a break strategy.
- Pay close attention to the previous executor's output, action, feedback, and special notes.
- Learn from retrieved memories, especially the non-empty semantic metadata fields.
- Use few-shot trajectories only as optional examples when they are relevant.
- If requires_action is true, proposed_action MUST be exactly one valid action (no period). If false, proposed_action MUST be "".

Return JSON with exactly these keys:
{{"output": "...", "proposed_action": "...", "special_notes": [...]}}

- output: your reasoning and operational summary for this stage.
- proposed_action: single valid action string (no period) or "".
- special_notes: critical constraints, repair clues, lessons from this step.
- Return valid JSON only. No markdown.
""".strip()


def _extract_semantic_metadata_highlights(memory_hits: List[Dict[str, Any]]) -> str:
    highlights: List[str] = []
    for index, item in enumerate(memory_hits or [], start=1):
        metadata = dict(item.get("memory_metadata", {}) or {})
        pairs = []
        for key, value in metadata.items():
            clean_key = str(key).strip()
            clean_value = _short_text(value, 160)
            if not clean_key or clean_key.startswith("_"):
                continue
            if not clean_value or clean_value.upper() == "N/A":
                continue
            pairs.append(f"{clean_key}={clean_value}")
        if pairs:
            highlights.append(f"  [{index}] " + "; ".join(pairs))
    if not highlights:
        return ""
    return "Key semantic metadata from memories:\n" + "\n".join(highlights)


def build_task_experience_prompt(
    task_description: str,
    trajectory_summary: List[Dict[str, Any]],
    task_success: bool,
    average_reward: float,
    executor_names: List[str],
    max_experiences: int = 3,
) -> str:
    return f"""
You are the post-task reflection orchestrator for a multi-agent system.

Your job is to read one completed task trajectory and extract a small set of high-value reusable experiences.

Task:
{task_description}

Task result:
- success: {task_success}
- average_reward: {average_reward:.3f}

Available executors:
{json.dumps(executor_names, ensure_ascii=False)}

Trajectory summary:
{json.dumps(trajectory_summary, ensure_ascii=False)}

Goal:
- Produce at most {max_experiences} experience items.
- Each item should be grounded in a short trajectory snippet plus a reusable lesson.
- Assign each item to the relevant executor(s), not to the whole team.

Requirements:
1. Focus on transferable lessons for similar states/tasks.
2. Keep each lesson concise, actionable, and specific.
3. Prefer evidence-backed lessons (cite step signals via snippet + related step indexes).
4. Avoid generic statements like "be careful" without concrete context.
5. Use only executor names in the provided list.

Return JSON with keys:
- experiences

Where `experiences` is a list of objects with keys:
- agent_names: list[str]
- related_step_indexes: list[int]
- trajectory_snippet: str
- experience: str

Hard rules:
- Maximum {max_experiences} items.
- Do not include markdown fences.
- Return valid JSON only.
- Do not include extra top-level keys.
""".strip()


def build_combined_memory_update_prompt(
    task_description: str,
    task_goal: str,
    task_success: bool,
    trajectory_steps: List[Dict[str, Any]],
    skill_selected_indices: List[int],
    skill_metadata_fields: List[str],
    skill_fixed_fields: Dict[int, str] | None,
    executor_selections: Dict[str, List[int]],
    executor_metadata_fields: Dict[str, List[str]],
    executor_failure_selections: Dict[str, int | None] | None = None,
    executor_failure_metadata_fields: Dict[str, List[str]] | None = None,
) -> tuple[str, str]:
    system_prompt = (
        "You are the memory writer for a multi-agent system. "
        "After a task is completed, you fill in memory_metadata for selected skill invocations, executor actions, and failure examples. "
        "Be specific and actionable. Ground every field value in what actually happened in the trajectory."
    )

    steps_text = ""
    for idx, step in enumerate(trajectory_steps, 1):
        steps_text += f"  [Step {idx}] " + _format_detail_steps([step]).replace("\n", "\n  ") + "\n"

    # Skill memory section
    skill_section = ""
    if skill_selected_indices:
        skill_fields_text = ", ".join(skill_metadata_fields)
        skill_indices_text = ", ".join(str(i) for i in skill_selected_indices)
        fixed_field_lines = []
        for idx in skill_selected_indices:
            fixed_field = str((skill_fixed_fields or {}).get(idx, "")).strip()
            if fixed_field:
                fixed_field_lines.append(f"  step {idx}: fixed skill-specific field = {fixed_field}")
        fixed_field_text = "\n".join(fixed_field_lines)
        skill_section = f"""
--- Skill Memory ---
Selected steps for skill memory: [{skill_indices_text}]
Common fields to fill for every selected step: [{skill_fields_text or "(none)"}]
Focus: For each step, what should the COORDINATOR remember about this skill invocation to make better skill selections in similar future states?
Skill-specific fields to fill:
{fixed_field_text or "  (none)"}
Important: for each selected step, fill ONLY its listed skill-specific field. Do not fill skill-specific fields that belong to other skills.
"""

    # Executor memory section (positive)
    exec_section = ""
    for agent_name, step_indices in executor_selections.items():
        if not step_indices:
            continue
        fields = executor_metadata_fields.get(agent_name, ["lesson"])
        fields_text = ", ".join(fields)
        indices_text = ", ".join(str(i) for i in step_indices)
        exec_section += f"""
--- Executor Memory: {agent_name} ---
Selected steps: [{indices_text}]
Fields to fill: [{fields_text}]
Focus: For each step, what should "{agent_name}" remember about their action/output to perform better in similar situations?
"""

    # Failure memory section
    failure_section = ""
    fail_selections = executor_failure_selections or {}
    fail_agents = [(name, idx) for name, idx in fail_selections.items() if idx is not None]
    if fail_agents:
        failure_section = """
--- Failure Memory (NEGATIVE EXAMPLES) ---
These are steps where something went wrong (the action produced a failure signal, e.g. low or negative reward). Fill in the requested fields to help agents avoid repeating these mistakes.
"""
        for name, idx in fail_agents:
            fail_fields = (executor_failure_metadata_fields or {}).get(name, ["lesson", "error_analysis"])
            fail_fields_text = ", ".join(fail_fields)
            failure_section += (
                f"  {name}: step {idx}\n"
                f"    Fields to fill: [{fail_fields_text}]\n"
                f"    Focus: summarize the failure, the incorrect assumption, and what should be done instead.\n"
            )

    user_prompt = f"""
A task has just been completed. Result: {"SUCCESS" if task_success else "FAILED"}.

Task description: {task_description}
Task goal: {task_goal}

Full trajectory:
{steps_text}
{skill_section}{exec_section}{failure_section}
Requirements:
1. For each selected step, provide concrete values for EVERY listed field. Do NOT leave any field as empty string.
2. Each field value: 1-2 sentences, grounded in what happened. Write "N/A" if genuinely not applicable.
3. Pay attention to environment feedback, special_notes, and action outcomes.
4. For FAILURE entries: analyze what went wrong, incorrect assumptions, and what should be done instead.
5. For FAILURE entries, avoid "N/A" whenever the action, feedback, subgoal, or state trace provides enough evidence to write a concrete warning.

Return JSON:
{{"skill_memory": [{{"step_index": <int>, "memory_metadata": {{...}}}}, ...], "executor_memory": {{"<agent_name>": [{{"step_index": <int>, "memory_metadata": {{...}}}}, ...], ...}}, "failure_memory": {{"<agent_name>": {{"step_index": <int>, "memory_metadata": {{...}}}}, ...}}}}

Hard rules:
- Return valid JSON only. No markdown.
""".strip()
    return system_prompt, user_prompt


def build_failure_memory_repair_prompt(
    task_description: str,
    task_goal: str,
    trajectory_steps: List[Dict[str, Any]],
    failure_requests: Dict[str, Dict[int, List[str]]],
) -> tuple[str, str]:
    system_prompt = (
        "You repair missing failure-memory metadata for a multi-agent system. "
        "Only write concrete negative-example metadata grounded in the trajectory. "
        "Do not invent facts and do not return generic placeholders."
    )

    steps_text = ""
    for idx, step in enumerate(trajectory_steps, 1):
        steps_text += f"  [Step {idx}] " + _format_detail_steps([step]).replace("\n", "\n  ") + "\n"

    request_lines = []
    for agent_name, by_step in failure_requests.items():
        for step_index, fields in by_step.items():
            request_lines.append(
                f"- {agent_name}, step {step_index}: fill fields [{', '.join(fields)}]"
            )

    user_prompt = f"""
Task description: {task_description}
Task goal: {task_goal}

Full trajectory:
{steps_text}

Missing failure-memory fields:
{chr(10).join(request_lines)}

Requirements:
1. Return one entry for every requested agent/step.
2. Fill every requested field with a concrete warning, diagnosis, or repair action grounded in the action, feedback, state trace, or subgoal.
3. Do not write "N/A", empty strings, "unknown", or placeholders. If a field is hard, infer the narrowest useful warning from the observed failure.
4. Return valid JSON only. No markdown.

Return JSON:
{{"failure_memory": {{"<agent_name>": [{{"step_index": <int>, "memory_metadata": {{...}}}}, ...]}}}}
""".strip()
    return system_prompt, user_prompt


def build_skill_generation_prompts(
    generation_mode: str,
    candidate: Dict[str, Any],
    existing_skill_names: List[str],
    existing_skill_metadata: List[Dict[str, Any]] | None = None,
) -> tuple[str, str]:
    if generation_mode == "pattern_reuse":
        return _build_pattern_reuse_prompts(candidate, existing_skill_names, existing_skill_metadata or [])
    return _build_failure_novel_prompts(candidate, existing_skill_names)


def _format_brief_steps(steps: List[Dict[str, Any]]) -> str:
    if not steps:
        return "  (no prior steps)"
    lines: List[str] = []
    for s in steps:
        parts = [f"skill={s.get('skill_name', '?')}"]
        assignment = s.get("assignment", {})
        if assignment:
            parts.append(f"assignment={json.dumps(assignment, ensure_ascii=False)}")
        for stage in s.get("stages", []):
            stage_parts = [f"{stage.get('stage_name', '?')}"]
            out = stage.get("output", "")
            if out:
                stage_parts.append(f"output={out[:120]}")
            act = stage.get("proposed_action", "")
            if act:
                stage_parts.append(f"action={act}")
            parts.append("stage[" + "; ".join(stage_parts) + "]")
        fb = s.get("environment_feedback", "")
        if fb and fb != "No environment action in this step.":
            parts.append(f"env_feedback={fb[:120]}")
        lines.append("  " + " | ".join(parts))
    return "\n".join(lines)


def _format_detail_steps(steps: List[Dict[str, Any]]) -> str:
    if not steps:
        return "  (empty)"
    lines: List[str] = []
    for s in steps:
        lines.append(f"  skill={s.get('skill_name', '?')}, subgoal={s.get('subgoal', 'N/A')}")
        assignment = s.get("assignment", {})
        if assignment:
            lines.append(f"    assignment: {json.dumps(assignment, ensure_ascii=False)}")
        for stage in s.get("stages", []):
            sn = stage.get("stage_name", "?")
            sg = stage.get("stage_subgoal", "")
            out = stage.get("output", "")
            act = stage.get("proposed_action", "")
            notes = stage.get("special_notes", [])
            line = f"    [{sn}] agent={stage.get('agent', '?')}"
            if sg:
                line += f" stage_subgoal={sg[:100]}"
            if out:
                line += f" output={out[:150]}"
            if act:
                line += f" action={act}"
            if notes:
                line += f" special_notes={json.dumps(notes, ensure_ascii=False)}"
            lines.append(line)
        fb = s.get("environment_feedback", "")
        if fb and fb != "No environment action in this step.":
            lines.append(f"    environment_feedback: {fb[:150]}")
    return "\n".join(lines)


def _build_pattern_reuse_prompts(
    candidate: Dict[str, Any],
    existing_skill_names: List[str],
    existing_skill_metadata: List[Dict[str, Any]],
) -> tuple[str, str]:
    system_prompt = (
        "You are a skill designer for a multi-agent system (MAS) that solves multi-step complex tasks. "
        "We have identified workflow patterns that contribute most to task success. "
        "Your job is to formalize these patterns into well-described reusable skills. "
        "You must fully design the skill: name, description, note, stage definitions, and each stage's prompt_template. "
        "You should only design skills that help task execution, not post-hoc analysis skills. "
        "Return strict JSON only."
    )

    phases = candidate.get("phases", [])
    snippets = candidate.get("snippets", [])

    examples_text = ""
    for idx, snippet in enumerate(snippets, 1):
        pre = _format_brief_steps(snippet.get("pre_window_steps", []))
        window = _format_detail_steps(snippet.get("window_steps", []))
        examples_text += f"""
【Example {idx}】
Task goal: {snippet.get('task_goal', 'N/A')}
Task description: {snippet.get('task_description', 'N/A')}

Trajectory before this workflow:
{pre}

What this workflow did:
{window}
"""

    existing_skills_text = ""
    if existing_skill_metadata:
        lines: List[str] = []
        for meta in existing_skill_metadata:
            name = meta.get("skill_name", "")
            desc = meta.get("description", "")
            note = meta.get("note", "")
            memory_field = meta.get("fixed_skill_memory_field", "")
            template_type = meta.get("template_type", "solo")
            stages = meta.get("stages", [])
            lines.append(
                f"- {name}: description={desc}; note={note}; memory_field={memory_field}; "
                f"template_type={template_type}; stages={json.dumps(stages, ensure_ascii=False)}"
            )
        existing_skills_text = "\n".join(lines)
    else:
        existing_skills_text = "\n".join(f"- {name}" for name in existing_skill_names)

    user_prompt = f"""
We need you to formalize the following workflow into a new skill.

Workflow sequence: {' -> '.join(phases)}
Suggested workflow length: {len(phases)}

This workflow sequence is evidence, not a fixed implementation. You MUST design the new skill's stage structure and prompt_template fields yourself, using the successful examples below as guidance. Do not simply copy an existing skill's stages.

Below are {len(snippets)} example(s) where this workflow was used successfully:
{examples_text}

Existing skills (your new skill must be clearly differentiated from these):
{existing_skills_text}

Your task:
1. Give this skill a clear, descriptive name (must not collide with existing names).
2. Write a concise description of when this skill should be used.
3. Write a note with usage conditions, caveats, and tips for the coordinator who selects skills.
4. Design the skill's template_type and stages. Each stage must have a concrete stage_name, stage_role, requires_action, and prompt_template.
5. In each stage's prompt_template, specify what the executor should do, whether proposed_action should be empty or a real action, and whether special_notes should be produced for downstream coordination.
6. Choose which example (1-{len(snippets)}) best represents the typical use case of this skill, as the golden example.
7. Design one dedicated skill-memory metadata field for this skill. This field will store compact evidence that helps the coordinator decide when to select this skill in future tasks.

Naming rules:
- The name must be meaningful and descriptive (e.g. "plan_then_execute_local", "navigate_inspect_verify").
- Do NOT use meaningless numbers, codes, or random suffixes (e.g. "skill_001", "workflow_abc123").
- The name can be 1-5 words connected by underscores.
- The description and note must be clearly different from existing skills listed above.

Return JSON with exactly these keys:
- skill_name: a new unique name for this skill
- description: when and why to use this skill (1-2 sentences)
- note: usage conditions, caveats, what to watch out for (1-3 sentences)
- template_type: "solo", "sequential_2", "sequential_3", "parallel_2", "parallel_3", or "sequential_chain"
- stages: list of 1-5 stage objects, each with:
  - stage_name: descriptive snake_case name
  - stage_role: role label
  - requires_action: true if this stage should produce an environment action, false otherwise
  - prompt_template: full instruction for the executor at this stage
- golden_example_index: integer (1-{len(snippets)}), which example to use as the golden example
- fixed_skill_memory_field: a concise snake_case field name dedicated to this skill, generated from what this skill needs to remember (do NOT use a generic skill_focus field)
- dedicated_field_description: one-line description of what this dedicated field stores

Hard rules:
- Return valid JSON only.
- Do not include markdown fences.
- Do not include extra keys.
- skill_name must not equal any existing skill name.
- Stage definitions must be newly written for this skill and should be as concrete as failure-novel skills.
""".strip()
    return system_prompt, user_prompt


def _build_failure_novel_prompts(
    candidate: Dict[str, Any],
    existing_skill_names: List[str],
) -> tuple[str, str]:
    system_prompt = (
        "You are a skill designer for a multi-agent system (MAS) that solves multi-step complex tasks. "
        "When you are called, it means the existing skills and their combinations have failed to solve the task. "
        "You are expected to design NEW solo skills (single agent, one stage) or parallel skills (2-3 agents working simultaneously on one step, no sequential dependency between them). "
        "You must fully design the skill: name, description, note, each stage definition, and each stage's prompt_template. "
        "Return strict JSON only."
    )

    trajectories = candidate.get("trajectories", [])

    # Format trajectories
    traj_text = ""
    for idx, traj in enumerate(trajectories, 1):
        steps_text = _format_brief_steps(traj.get("steps", []))
        traj_text += f"""
【Trajectory {idx}】
Task goal: {traj.get('task_goal', 'N/A')}
Task description: {traj.get('task_description', 'N/A')}

Steps:
{steps_text}
"""

    # Format existing skills with full detail
    existing_skills = candidate.get("existing_skill_details", [])
    solo_skills_text = ""
    parallel_skills_text = ""
    for skill in existing_skills:
        template_type = str(skill.get("template_type", "solo"))
        entry = f"""- {skill.get('skill_name', '?')}:
    description: {skill.get('description', 'N/A')}
    note: {skill.get('note', 'N/A')}
    memory_field: {skill.get('fixed_skill_memory_field', 'N/A')}
    stages: {json.dumps(skill.get('stages', []), ensure_ascii=False)}
"""
        if "parallel" in template_type:
            parallel_skills_text += entry
        else:
            solo_skills_text += entry

    if not solo_skills_text:
        solo_skills_text = "  (none)\n"
    if not parallel_skills_text:
        parallel_skills_text = "  (none)\n"

    user_prompt = f"""
Analyze the following {len(trajectories)} failed trajectories to identify what the existing skills cannot handle, then design a NEW skill to address the gap.

{traj_text}

You are allowed to design ONE of these skill types:
- solo: a single-agent, single-stage skill (one agent produces output alone)
- parallel: a 2-3 agent parallel skill (multiple agents produce output simultaneously in one step, NO sequential logic between them; choose "parallel" as template_type)

Existing solo skills:
{solo_skills_text}
Existing parallel skills:
{parallel_skills_text}

Design requirements:
1. Analyze what went wrong in the failed trajectories and what capability is missing.
2. Design a skill that would help in these situations.
3. For each stage, write a clear prompt_template that tells the executor exactly what to do.
4. In each stage's prompt_template, specify whether the executor should produce a special_note (for downstream coordination) and whether proposed_action should be empty or a real action.
5. The new skill must be clearly different from all existing skills.
6. Design one dedicated skill-memory metadata field for this skill. This field will store compact evidence that helps the coordinator decide when to select this skill in future tasks.

Naming rules:
- The name must be meaningful and descriptive (e.g. "diagnose_and_retry", "parallel_verify_action").
- Do NOT use meaningless numbers, codes, or random suffixes.
- The name can be 1-5 words connected by underscores.

Return JSON with exactly these keys:
- skill_name: unique descriptive name
- description: when and why to use this skill (1-2 sentences)
- note: usage conditions, caveats, tips for the coordinator (1-3 sentences)
- fixed_skill_memory_field: a concise snake_case field name dedicated to this skill, generated from what this skill needs to remember (do NOT use a generic skill_focus field)
- dedicated_field_description: one-line description of what this dedicated field stores
- template_type: "solo" or "parallel_2" or "parallel_3"
- stages: list of stage objects, each with:
  - stage_name: descriptive name
  - stage_role: role label
  - requires_action: true if this stage should produce an environment action, false otherwise
  - prompt_template: full instruction for the executor at this stage

Hard rules:
- Return valid JSON only.
- Do not include markdown fences.
- Do not include extra keys.
- skill_name must not equal any existing skill name.
- Stage count: 1 for solo, 2-3 for parallel.
""".strip()
    return system_prompt, user_prompt


def build_skill_memory_structure_evolution_prompts(
    current_memory_metadata_fields: List[str],
    contrast_cases: List[Dict[str, Any]],
    existing_skill_metadata: List[Dict[str, Any]] | None = None,
) -> tuple[str, str]:
    system_prompt = (
        "You are a memory architecture designer for a multi-agent system (MAS). "
        "Your job is to evolve the memory_metadata fields of skill memory entries. "
        "Skill memory helps the coordinator decide WHEN to use WHICH skill. "
        "You will be given pairs of failed trajectories and their counterfactual successful versions, "
        "especially the PIVOT POINT where the counterfactual started outperforming the original. "
        "The skill chosen at the pivot is critical. You must analyze what information distinguishes "
        "the successful choice from the failed one, and design memory_metadata fields to capture that. "
        "You ONLY modify fields inside memory_metadata dict. Do not touch other fields. "
        "Return strict JSON only."
    )

    cases_text = ""
    for idx, case in enumerate(contrast_cases, 1):
        task_goal = case.get("task_goal", "N/A")
        task_description = case.get("task_description", "N/A")
        shared_prefix = case.get("shared_prefix_trace", "(no shared prefix)")
        failed_to_pivot = _format_brief_steps(case.get("failed_steps_to_pivot", []))
        success_to_pivot = _format_brief_steps(case.get("success_steps_to_pivot", []))
        failed_pivot = _format_detail_steps(case.get("failed_pivot_step", []))
        success_pivot = _format_detail_steps(case.get("success_pivot_step", []))

        cases_text += f"""
【Contrast Pair {idx}】
Task goal: {task_goal}
Task description: {task_description}

Shared prefix (before the fork):
{shared_prefix}

Failed branch (fork to pivot):
{failed_to_pivot}

Successful branch (fork to pivot):
{success_to_pivot}

Failed pivot step (detailed):
{failed_pivot}

Successful pivot step (detailed):
{success_pivot}
"""

    existing_skills_text = _format_existing_skill_metadata(existing_skill_metadata)

    user_prompt = f"""
We need you to evolve the skill memory structure. You must ONLY modify fields inside the memory_metadata dict. Do NOT touch other fields (task_description, task_goal, subgoal, state_trace_summary, etc.).

Carefully compare the failed trajectories with the counterfactual successful trajectories, especially at the pivot point. Observe:
- What information did the successful branch use that the failed branch ignored?
- What patterns in the pivot step's skill choice, stage assignments, or outputs made the difference?
- What should be remembered so the coordinator can make better skill selections in similar future situations?

Summarize your findings as memory_metadata fields. These fields will be stored per skill-memory entry and used by the coordinator when selecting skills.

Existing skills available to the coordinator:
{existing_skills_text}

Below are the contrastive trajectory pairs:
{cases_text}

Current memory_metadata fields:
{json.dumps(current_memory_metadata_fields, ensure_ascii=False)}

Requirements:
1. You may ADD new fields, REMOVE useless fields, or RENAME existing fields inside memory_metadata.
2. Each field name must be meaningful and clear (e.g. "critical_object_relation", "action_prerequisite", "failure_pattern").
3. Fields must be semantically distinct from the outer entry fields (task_description, task_goal, subgoal, state_trace_summary, recent_environment_feedback, stage_assignments, stage_subgoals).
4. These fields will be stored for a SINGLE trajectory step, not for comparing two trajectories. Do NOT design fields like "comparison_result" or "trajectory_diff".
5. Fields should help the coordinator decide: in this state, should I pick skill A or skill B?
6. Use the existing skill metadata to understand what kinds of selection cues distinguish the available skills, but design general memory fields rather than fields tied to one hard-coded task.
7. Keep the total number of memory_metadata fields between 2-6.

Return JSON with exactly these keys:
- fields: list of new memory_metadata field names (list[str])
- field_descriptions: dict mapping each field name to a one-line description of what it stores
- rationale: brief explanation of why these fields help (1-3 sentences)

Hard rules:
- Return valid JSON only.
- Do not include markdown fences.
- Do not include extra keys.
""".strip()
    return system_prompt, user_prompt


def build_executor_memory_structure_evolution_prompts(
    agent_name: str,
    current_memory_metadata_fields: List[str],
    current_failure_metadata_fields: List[str],
    contrast_cases: List[Dict[str, Any]],
    existing_skill_metadata: List[Dict[str, Any]] | None = None,
) -> tuple[str, str]:
    system_prompt = (
        "You are a memory architecture designer for a multi-agent system (MAS). "
        f"Your job is to evolve BOTH the positive-memory fields and the failure-memory fields for agent \"{agent_name}\". "
        "Executor memory helps this agent decide HOW to act at a given stage: what action to take, "
        "what to pay attention to, and what mistakes to avoid. "
        f"You will be given trajectories where agent \"{agent_name}\" was present but the task FAILED, "
        f"paired with counterfactual trajectories where \"{agent_name}\" was REMOVED and the task SUCCEEDED. "
        f"This means \"{agent_name}\" was making decisions that hurt performance. "
        "Analyze what information this agent was missing or ignoring, and design two complementary schemas: "
        "one for successful/action-guiding executor memories, and one for failure memories that explain what went wrong and how to avoid repeating it. "
        "You ONLY modify fields inside memory_metadata dicts. Do not touch other fields. "
        "You should prefer ADDING useful fields over removing existing ones. "
        "Return strict JSON only."
    )

    cases_text = ""
    for idx, case in enumerate(contrast_cases, 1):
        task_goal = case.get("task_goal", "N/A")
        task_description = case.get("task_description", "N/A")
        shared_prefix = case.get("shared_prefix_trace", "(no shared prefix)")
        failed_to_pivot = _format_brief_steps(case.get("failed_steps_to_pivot", []))
        success_to_pivot = _format_brief_steps(case.get("success_steps_to_pivot", []))
        failed_pivot = _format_detail_steps(case.get("failed_pivot_step", []))
        success_pivot = _format_detail_steps(case.get("success_pivot_step", []))

        cases_text += f"""
【Contrast Pair {idx}】
Task goal: {task_goal}
Task description: {task_description}

Shared prefix (before the fork, {agent_name} was still present):
{shared_prefix}

Failed branch ({agent_name} present, task failed — fork to pivot):
{failed_to_pivot}

Successful branch ({agent_name} removed, task succeeded — fork to pivot):
{success_to_pivot}

Failed pivot step (detailed — {agent_name} was involved):
{failed_pivot}

Successful pivot step (detailed — {agent_name} was absent):
{success_pivot}
"""

    existing_skills_text = _format_existing_skill_metadata(existing_skill_metadata)

    user_prompt = f"""
We need you to evolve the executor memory structure for agent "{agent_name}". You must ONLY modify fields inside the memory_metadata dicts.

In the trajectories below, when "{agent_name}" was present the task failed, but when "{agent_name}" was removed the task succeeded. This means "{agent_name}" needs better memory to make better decisions.

Carefully compare what happened with and without "{agent_name}", especially at the pivot point:
- What action or decision did "{agent_name}" make (or influence) that led to failure?
- What contextual information was "{agent_name}" missing that the remaining agents utilized?
- What should "{agent_name}" remember so they can avoid these mistakes and make better choices?

Design:
- positive memory fields: what "{agent_name}" should remember when facing a similar stage in the future
- failure memory fields: what "{agent_name}" should remember specifically from failed actions, wrong assumptions, and repair strategies

These fields will be stored per executor-memory entry and retrieved when "{agent_name}" faces similar situations.

Existing skills and stages that executors may be assigned to:
{existing_skills_text}

Below are the contrastive trajectory pairs:
{cases_text}

Current positive-memory fields:
{json.dumps(current_memory_metadata_fields, ensure_ascii=False)}

Current failure-memory fields:
{json.dumps(current_failure_metadata_fields, ensure_ascii=False)}

Requirements:
1. You may ADD new fields or RENAME existing fields inside memory_metadata. Prefer adding over removing.
2. Each field name must be meaningful and clear (e.g. "action_prerequisite", "object_interaction_hint", "common_mistake", "environment_constraint").
3. Fields must be semantically distinct from the outer entry fields (task_description, subgoal, stage_subgoal, state_trace_summary, recent_environment_feedback, action_summary, assigned_part, workflow_name).
4. These fields will be stored for a SINGLE executor action, not for comparing trajectories.
5. Positive-memory fields should help "{agent_name}" decide: in this state + this stage, what action should I take and what should I watch out for?
6. Failure-memory fields should help "{agent_name}" remember: what failed, what assumption was wrong, and what correction or recovery should happen next.
7. Use the existing skill and stage metadata to understand the stage roles this agent may face, but design general memory fields rather than fields tied to one hard-coded task.
8. Keep the total number of positive-memory fields between 2-6.
9. Keep the total number of failure-memory fields between 2-6.

Return JSON with exactly these keys:
- success_fields: list of new positive-memory field names (list[str])
- success_field_descriptions: dict mapping each positive field name to a one-line description
- failure_fields: list of new failure-memory field names (list[str])
- failure_field_descriptions: dict mapping each failure field name to a one-line description
- rationale: brief explanation of why these fields help "{agent_name}" (1-3 sentences)

Hard rules:
- Return valid JSON only.
- Do not include markdown fences.
- Do not include extra keys.
""".strip()
    return system_prompt, user_prompt
