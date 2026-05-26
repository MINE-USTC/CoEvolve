from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
from typing import Dict, Iterable, List

from ..schema import SkillSpec
from ..storage.io import load_json


class SkillLibraryStore:
    def __init__(self, initial_skills: Iterable[SkillSpec] | None = None) -> None:
        self.skills: Dict[str, SkillSpec] = {}
        for spec in initial_skills or []:
            self.add_skill(spec)

    def add_skill(self, spec: SkillSpec) -> None:
        self._ensure_fixed_skill_field(spec)
        if spec.skill_name not in self.skills:
            self.skills[spec.skill_name] = spec

    def extend(self, specs: Iterable[SkillSpec]) -> None:
        for spec in specs:
            self.add_skill(spec)

    def get(self, skill_name: str) -> SkillSpec | None:
        return self.skills.get(skill_name)

    def names(self) -> List[str]:
        return list(self.skills.keys())

    def __len__(self) -> int:
        return len(self.skills)

    def selection_specs(self) -> List[dict[str, object]]:
        return [self._selection_spec(spec) for spec in self.skills.values()]

    def snapshot(self) -> dict[str, object]:
        return {
            "skills": [asdict(spec) for spec in self.skills.values()]
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object] | None, fallback_skill_names: List[str] | None = None) -> "SkillLibraryStore":
        if payload and payload.get("skills"):
            skills = [cls._coerce_skill_spec(item) for item in payload.get("skills", [])]
            return cls(initial_skills=skills)
        return cls(initial_skills=_primitive_skill_specs(fallback_skill_names or ["plan", "execute", "reflect"]))

    @classmethod
    def from_file(cls, path: str | Path, fallback_skill_names: List[str] | None = None) -> "SkillLibraryStore":
        payload = load_json(path, default=None)
        return cls.from_payload(payload, fallback_skill_names=fallback_skill_names)

    @staticmethod
    def _selection_spec(spec: SkillSpec) -> dict[str, object]:
        selection = dict(spec.selection_metadata or {})
        detail = dict(spec.detail or {})
        stage_count = int(detail.get("stage_count", len(detail.get("stages", [1]))))
        lifecycle = dict(spec.metadata.get("lifecycle", {}) or {})
        return {
            "skill_name": spec.skill_name,
            "base_skill": spec.base_skill,
            "source": spec.source,
            "description": selection.get("description", f"Workflow rooted in {spec.base_skill}."),
            "note": selection.get("note", ""),
            "golden_example_id": selection.get("golden_example_id", ""),
            "selection_tags": list(selection.get("selection_tags", [])),
            "agent_count_hint": selection.get("agent_count_hint", "1"),
            "stage_count": stage_count,
            "lifecycle": lifecycle,
        }

    @staticmethod
    def _coerce_skill_spec(payload: dict[str, object]) -> SkillSpec:
        spec = SkillSpec(
            skill_name=str(payload.get("skill_name", "")),
            base_skill=str(payload.get("base_skill", payload.get("skill_name", ""))),
            source=str(payload.get("source", "primitive")),
            selection_metadata=dict(payload.get("selection_metadata", {})),
            detail=dict(payload.get("detail", {})),
            phases=list(payload.get("phases", [])),
            metadata=dict(payload.get("metadata", {})),
        )
        SkillLibraryStore._ensure_fixed_skill_field(spec)
        return spec

    @staticmethod
    def _ensure_fixed_skill_field(spec: SkillSpec) -> None:
        metadata = dict(spec.metadata or {})
        if not str(metadata.get("fixed_skill_memory_field", "")).strip():
            metadata["fixed_skill_memory_field"] = _skill_fixed_field_name(spec.skill_name)
        spec.metadata = metadata


def _primitive_skill_specs(skill_names: List[str]) -> List[SkillSpec]:
    return [
        SkillSpec(
            skill_name=name,
            base_skill=name,
            source="primitive",
            selection_metadata={
                "description": f"Primitive {name} workflow.",
                "note": "",
                "golden_example_id": "",
                "selection_tags": [name],
                "agent_count_hint": "1",
            },
            detail={
                "template_type": "solo",
                "memory_top_k": 5,
                "stages": [
                    {
                        "stage_name": f"{name}_stage",
                        "stage_role": name,
                        "prompt_template": f"Run the primitive {name} stage for the current input.",
                    }
                ],
            },
            phases=[name],
            metadata={"primitive": True, "fixed_skill_memory_field": _skill_fixed_field_name(name)},
        )
        for name in skill_names
    ]


def _skill_fixed_field_name(skill_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(skill_name).strip().lower()).strip("_")
    return f"skill_focus__{normalized or 'skill'}"
