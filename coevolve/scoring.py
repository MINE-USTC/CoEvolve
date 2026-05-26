"""Step-level scoring based on exploration diversity, shared across pipeline and evolution."""
from __future__ import annotations

from typing import Any, Dict, List


_TRIVIAL_FEEDBACKS = {"nothing happens.", "no environment action in this step.", "no environment action in this stage.", ""}


def compute_step_scores(
    steps: list,
    skill_library: dict | None = None,
    trivial_feedbacks: set | None = None,
    failure_reward_threshold: float = 0.0,
) -> List[float]:
    """Compute the effective step signal used by exploration and evolution.

    Rule:
    - reward < 0: keep the negative reward as-is
    - reward > 0 and reward != 1: keep the positive reward as-is
    - reward == 0 or reward == 1: use the environment-feedback exploration score

    Exploration score:
    - distinct_ratio = unique meaningful feedbacks so far / stages so far
    - repeat_ratio = repeated meaningful feedbacks so far / action steps so far
    - score = max(0, distinct_ratio - 0.6 * repeat_ratio)
    """
    effective_trivials = trivial_feedbacks if trivial_feedbacks is not None else _TRIVIAL_FEEDBACKS
    seen_feedbacks: Dict[str, int] = {}
    action_steps = 0
    scores: List[float] = []

    for stage_count, step in enumerate(steps, start=1):
        reward = _get_reward(step)
        feedback = _get_feedback(step)
        has_action = _has_action(step)

        if has_action:
            action_steps += 1
            normalized = feedback.strip().lower()
            if normalized and normalized not in effective_trivials:
                seen_feedbacks[normalized] = seen_feedbacks.get(normalized, 0) + 1

        if reward < 0:
            scores.append(reward)
            continue

        if reward > 0 and reward != 1:
            scores.append(reward)
            continue

        n_distinct = len(seen_feedbacks)
        n_repeated = sum(max(0, count - 1) for count in seen_feedbacks.values())
        stage_denom = max(1, stage_count)
        action_denom = max(1, action_steps)
        score = n_distinct / stage_denom - 0.6 * n_repeated / action_denom
        scores.append(max(0.0, score))

    return scores


def _get_reward(step) -> float:
    if isinstance(step, dict):
        return float(step.get("reward", 0.0))
    return float(step.reward)


def _get_feedback(step) -> str:
    if isinstance(step, dict):
        meta = step.get("metadata", {})
        if isinstance(meta, dict):
            return str(meta.get("environment_feedback", ""))
        return ""
    return str(step.metadata.get("environment_feedback", ""))


def _has_action(step) -> bool:
    if isinstance(step, dict):
        outputs = step.get("executor_outputs", {})
    else:
        outputs = step.executor_outputs
    if not isinstance(outputs, dict):
        return False
    for payload in outputs.values():
        if isinstance(payload, dict) and str(payload.get("proposed_action", "")).strip():
            return True
    return False
