from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from ..schema import ProfileSnapshot, TrajectoryRecord
from .base import BaseMemoryStore


class ProfileStore(BaseMemoryStore):
    def __init__(self, storage_path: str, agent_names: List[str], collaboration_styles: Dict[str, str] | None = None) -> None:
        super().__init__(storage_path)
        styles = collaboration_styles or {}
        self.profiles: Dict[str, ProfileSnapshot] = {
            name: ProfileSnapshot(agent_name=name, collaboration_style=str(styles.get(name, ""))) for name in agent_names
        }

    def all(self) -> List[Dict[str, object]]:
        return [asdict(item) for item in self.profiles.values()]

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_text = str(query or "").lower().strip()
        tokens = [item for item in query_text.replace("\n", " ").split(" ") if item]
        scored: List[tuple[float, ProfileSnapshot]] = []
        for profile in self.profiles.values():
            score = 0.0
            style = str(profile.collaboration_style or "").lower()
            if query_text and query_text in style:
                score += 1.0
            if tokens:
                for token in tokens:
                    if token in style:
                        score += 0.2
                    if token in " ".join(profile.collaboration_tags).lower():
                        score += 0.15
            score += max(0.0, profile.average_reward) * 0.1
            score += min(1.0, profile.successful_tasks / max(1, profile.completed_tasks)) * 0.1
            scored.append((score, profile))

        scored.sort(key=lambda item: item[0], reverse=True)
        limited = scored[: max(1, int(top_k))]
        return [
            {
                "agent_name": item.agent_name,
                "collaboration_style": item.collaboration_style,
                "completed_tasks": item.completed_tasks,
                "successful_tasks": item.successful_tasks,
                "average_reward": item.average_reward,
                "preferred_skills": dict(item.preferred_skills),
                "preferred_stages": dict(item.preferred_stages),
                "preferred_skill_stages": dict(item.preferred_skill_stages),
                "collaboration_tags": list(item.collaboration_tags),
                "metadata": dict(item.metadata),
            }
            for _, item in limited
        ]

    def update_from_trajectory(self, trajectory: TrajectoryRecord) -> None:
        touched_agents = set()
        for step in trajectory.steps:
            for agent_name, output in step.executor_outputs.items():
                touched_agents.add(agent_name)
                profile = self.profiles[agent_name]
                strengths = output.get("strength_hints", [])
                failures = output.get("failure_hints", [])
                collab_tags = output.get("collaboration_tags", [])
                profile.recent_strengths = list((profile.recent_strengths + strengths)[-5:])
                profile.recent_failures = list((profile.recent_failures + failures)[-5:])
                profile.collaboration_tags = list((profile.collaboration_tags + collab_tags)[-8:])
                profile.preferred_skills[step.decision.skill_name] = profile.preferred_skills.get(step.decision.skill_name, 0.0) + step.reward
                stage_role = str(output.get("stage_role", "")).strip()
                stage_name = str(output.get("stage_name", "")).strip()
                stage_key = stage_role or stage_name
                if stage_key:
                    profile.preferred_stages[stage_key] = profile.preferred_stages.get(stage_key, 0.0) + step.reward
                    skill_stage_key = f"{step.decision.skill_name}/{stage_key}"
                    profile.preferred_skill_stages[skill_stage_key] = (
                        profile.preferred_skill_stages.get(skill_stage_key, 0.0) + step.reward
                    )
                for teammate in step.decision.participants:
                    if teammate == agent_name:
                        continue
                    profile.teammate_affinity[teammate] = profile.teammate_affinity.get(teammate, 0.0) + (0.5 if trajectory.success else -0.2)

        for agent_name in touched_agents:
            profile = self.profiles[agent_name]
            profile.completed_tasks += 1
            if trajectory.success:
                profile.successful_tasks += 1
            total = max(1, profile.completed_tasks)
            profile.average_reward += (float(trajectory.average_reward) - profile.average_reward) / total
            profile.metadata["last_task_id"] = trajectory.task.task_id
            profile.metadata["collaboration_profile"] = self._build_collaboration_profile(profile)

    def snapshot(self) -> Dict[str, object]:
        return {
            "profiles": {name: asdict(profile) for name, profile in self.profiles.items()}
        }

    def persist(self, reason: str = "") -> None:
        if not self.storage_dir:
            return
        payload = self.snapshot()
        payload["persist_reason"] = reason
        self.persist_snapshot("profiles.json", payload=payload)

    def load_snapshot(self, payload: Dict[str, object] | None) -> None:
        if not payload:
            return
        profiles = payload.get("profiles", {})
        for name, item in profiles.items():
            data = dict(item)
            if "collaboration_style" not in data:
                data["collaboration_style"] = self.profiles.get(name, ProfileSnapshot(agent_name=name)).collaboration_style
            self.profiles[name] = ProfileSnapshot(**data)

    @staticmethod
    def _build_collaboration_profile(profile: ProfileSnapshot) -> Dict[str, object]:
        sorted_skills = sorted(profile.preferred_skills.items(), key=lambda item: item[1], reverse=True)
        sorted_stages = sorted(profile.preferred_stages.items(), key=lambda item: item[1], reverse=True)
        sorted_skill_stages = sorted(profile.preferred_skill_stages.items(), key=lambda item: item[1], reverse=True)
        sorted_teammates = sorted(profile.teammate_affinity.items(), key=lambda item: item[1], reverse=True)
        return {
            "best_skills": [name for name, _ in sorted_skills[:2]],
            "best_stages": [name for name, _ in sorted_stages[:2]],
            "best_skill_stages": [name for name, _ in sorted_skill_stages[:3]],
            "best_teammates": [name for name, _ in sorted_teammates[:2]],
            "recent_strengths": profile.recent_strengths[-3:],
            "recent_failures": profile.recent_failures[-3:],
        }
