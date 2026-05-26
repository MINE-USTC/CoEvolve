from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import re
from typing import Any, Dict, Iterable, List, Tuple

from ..llm.json_utils import parse_json_response
from ..memory import ExecutorMemoryStore, SkillMemoryStore
from ..prompts import (
    build_executor_memory_structure_evolution_prompts,
    build_skill_generation_prompts,
    build_skill_memory_structure_evolution_prompts,
)
from ..schema import (
    CounterfactualRecord,
    EvolutionArtifact,
    SkillSpec,
    TrajectoryRecord,
)
from ..scoring import compute_step_scores


def _skill_metadata_for_prompt(skill_library: Dict[str, SkillSpec] | None) -> List[Dict[str, Any]]:
    metadata: List[Dict[str, Any]] = []
    for name, spec in (skill_library or {}).items():
        selection_meta = dict(spec.selection_metadata or {})
        detail = dict(spec.detail or {})
        spec_meta = dict(spec.metadata or {})
        stages = list(detail.get("stages", []) or [])
        metadata.append({
            "skill_name": name,
            "base_skill": spec.base_skill,
            "description": str(selection_meta.get("description", "")),
            "note": str(selection_meta.get("note", "")),
            "fixed_skill_memory_field": str(spec_meta.get("fixed_skill_memory_field", "")),
            "template_type": str(detail.get("template_type", "solo")),
            "stage_count": int(detail.get("stage_count", len(stages) or 1) or 1),
            "stages": stages,
            "phases": list(spec.phases or []),
        })
    return metadata


class SkillEvolutionEngine:
    def __init__(
        self,
        max_new_skills_per_round: int = 2,
        max_failure_summary_skills_per_round: int = 1,
        max_evidence_per_candidate: int = 5,
        llm_client: Any | None = None,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.2,
    ) -> None:
        self.max_reuse_skills_per_round = max_new_skills_per_round
        self.max_failure_summary_skills_per_round = max_failure_summary_skills_per_round
        self.max_evidence_per_candidate = max(1, max_evidence_per_candidate)
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature

    def evolve(
        self,
        success_trajectories: List[TrajectoryRecord],
        failed_trajectories: List[TrajectoryRecord],
        counterfactual_records: List[CounterfactualRecord],
        skill_library: Dict[str, SkillSpec],
    ) -> EvolutionArtifact:
        if not success_trajectories and not counterfactual_records:
            return EvolutionArtifact(
                name="skill_evolution",
                applied=False,
                summary="No successful or counterfactual traces were available for skill evolution.",
                payload={},
            )

        # 1. Collect success pattern candidates (top 2)
        success_patterns = self._collect_success_patterns(success_trajectories, skill_library)
        success_candidates = success_patterns[:2]

        # 2. Collect counterfactual success candidates (top 1)
        cf_success_candidates = self._collect_cf_success_patterns(
            failed_trajectories, counterfactual_records, skill_library,
        )[:1]

        # 3. Collect counterfactual failure candidates (top 1 from worst 3)
        failure_cases = self._collect_failed_counterfactual_cases(
            failed_trajectories=failed_trajectories,
            counterfactual_records=counterfactual_records,
            skill_library=skill_library,
        )[:1]

        all_candidates = success_candidates + cf_success_candidates + failure_cases
        if not all_candidates:
            return EvolutionArtifact(
                name="skill_evolution",
                applied=False,
                summary="No new reusable workflow pattern passed the conservative skill evolution filters.",
                payload={
                    "success_trajectory_count": len(success_trajectories),
                    "counterfactual_count": len(counterfactual_records),
                },
            )

        created: List[SkillSpec] = []
        for candidate in all_candidates:
            phases = list(candidate.get("phases", []))
            if len(phases) < 2:
                continue
            source = candidate.get("source", "")
            generation_mode = "failure_novel" if "failed" in source else "pattern_reuse"
            spec = self._build_candidate_skill_spec(
                candidate=candidate,
                skill_library=skill_library,
                generation_mode=generation_mode,
            )
            if spec is None:
                continue
            skill_library[spec.skill_name] = spec
            created.append(spec)

        return EvolutionArtifact(
            name="skill_evolution",
            applied=bool(created),
            summary=(
                f"Added {len(created)} evolved skill(s) from successful/counterfactual workflow motifs."
                if created
                else "Skill evolution ran but did not add any new skill."
            ),
            payload={
                "added_skills": [asdict(item) for item in created],
                "skill_library_size": len(skill_library),
                "success_candidates": len(success_candidates),
                "cf_success_candidates": len(cf_success_candidates),
                "failure_cases": len(failure_cases),
            },
        )

    def _collect_cf_success_patterns(
        self,
        failed_trajectories: List[TrajectoryRecord],
        counterfactual_records: List[CounterfactualRecord],
        skill_library: Dict[str, SkillSpec],
        top_k: int = 1,
        snippets_per_pattern: int = 3,
    ) -> List[Dict[str, Any]]:
        """From counterfactual successes, extract pivot windows with stage-aware reward."""
        failed_index = {t.trajectory_id: t for t in failed_trajectories}
        pattern_data: Dict[Tuple[str, ...], Dict[str, Any]] = defaultdict(
            lambda: {"score": 0.0, "occurrences": []}
        )
        for record in counterfactual_records:
            source = failed_index.get(record.source_trajectory_id)
            if not source or not _counterfactual_improved(source, record):
                continue
            cf_payload = _counterfactual_payload(record)
            cf_steps = cf_payload.get("steps", [])
            if not cf_steps:
                continue
            phases, pivot_meta = _pivot_window_from_counterfactual(source, record, cf_steps, skill_library)
            if len(phases) < 2:
                continue
            window = tuple(phases)

            pivot_step = pivot_meta.get("pivot_index", record.removal_start_step)
            cf_step_index = {int(s.get("step_index", -1)): s for s in cf_steps}
            total_cf_steps = len(cf_steps)

            # Compute exploration scores for cf steps
            cf_step_scores = compute_step_scores(cf_steps)
            cf_score_map = {int(s.get("step_index", -1)): sc for s, sc in zip(cf_steps, cf_step_scores)}

            window_scores = []
            total_stages = 0
            window_end_idx = pivot_step
            for idx in range(pivot_step, max(cf_step_index.keys(), default=pivot_step) + 1):
                s = cf_step_index.get(idx)
                if not s:
                    continue
                sc = _step_stage_count(s, skill_library)
                window_scores.append(cf_score_map.get(idx, 0.0))
                total_stages += sc
                window_end_idx = idx
                if total_stages >= 2:
                    break
            total_stages = max(1, total_stages)
            window_score = sum(window_scores) / total_stages
            early_ratio = 1.0 - (pivot_step / max(1, total_cf_steps))
            occ_score = 2.0 * window_score + 1.0 * early_ratio
            pattern_data[window]["score"] += window_score
            pattern_data[window]["occurrences"].append({
                "source_trajectory": source,
                "cf_steps": cf_steps,
                "pivot_step": pivot_step,
                "window_end_idx": window_end_idx,
                "early_ratio": early_ratio,
                "occ_score": occ_score,
            })

        results: List[Dict[str, Any]] = []
        for window, data in pattern_data.items():
            occurrences = data["occurrences"]
            raw_score = data["score"]
            avg_early = sum(o["early_ratio"] for o in occurrences) / max(1, len(occurrences))
            weighted = 2.0 * raw_score + 1.0 * avg_early
            results.append({
                "phases": list(window),
                "weighted_score": weighted,
                "source": "counterfactual_success_pattern",
                "raw_occurrences": occurrences,
            })
        results.sort(key=lambda x: x["weighted_score"], reverse=True)

        final: List[Dict[str, Any]] = []
        for pattern in results[:top_k]:
            occs = sorted(pattern["raw_occurrences"], key=lambda o: o["occ_score"], reverse=True)[:snippets_per_pattern]
            snippets: List[Dict[str, Any]] = []
            for occ in occs:
                cf_steps = occ["cf_steps"]
                pivot = occ["pivot_step"]
                wend = occ["window_end_idx"]
                pre_window = [_serialize_step_brief(s, skill_library) for s in cf_steps if int(s.get("step_index", -1)) < pivot]
                window_detail = [
                    _serialize_step_detail(s, skill_library)
                    for s in cf_steps
                    if pivot <= int(s.get("step_index", -1)) <= wend
                ]
                snippets.append({
                    "task_description": occ["source_trajectory"].task.description,
                    "task_goal": occ["source_trajectory"].task.task_goal,
                    "pre_window_steps": pre_window,
                    "window_steps": window_detail,
                })
            final.append({
                "phases": pattern["phases"],
                "source": pattern["source"],
                "snippets": snippets,
            })
        return final

    def _build_candidate_skill_spec(
        self,
        candidate: Dict[str, Any],
        skill_library: Dict[str, SkillSpec],
        generation_mode: str,
    ) -> SkillSpec | None:
        if self.llm_client is None:
            return None
        generated = self._generate_with_llm(
            candidate=candidate,
            generation_mode=generation_mode,
            existing_skill_names=list(skill_library.keys()),
            skill_library=skill_library,
        )
        if generated is None:
            return None
        if generation_mode == "pattern_reuse":
            return self._coerce_pattern_reuse_spec(
                payload=generated,
                candidate=candidate,
                skill_library=skill_library,
            )
        return self._coerce_failure_novel_spec(
            payload=generated,
            candidate=candidate,
            skill_library=skill_library,
        )

    def _generate_with_llm(
        self,
        candidate: Dict[str, Any],
        generation_mode: str,
        existing_skill_names: List[str],
        skill_library: Dict[str, SkillSpec] | None = None,
    ) -> Dict[str, Any] | None:
        existing_skill_metadata = _skill_metadata_for_prompt(skill_library)
        system_prompt, user_prompt = build_skill_generation_prompts(
            generation_mode=generation_mode,
            candidate=candidate,
            existing_skill_names=existing_skill_names,
            existing_skill_metadata=existing_skill_metadata,
        )
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.llm_temperature,
                max_tokens=1400,
                top_p=1.0,
            )
            content = response.choices[0].message.content or "{}"
            return parse_json_response(content)
        except Exception:
            return None

    def _coerce_pattern_reuse_spec(
        self,
        payload: Dict[str, Any],
        candidate: Dict[str, Any],
        skill_library: Dict[str, SkillSpec],
    ) -> SkillSpec | None:
        """For pattern_reuse: LLM designs the full reusable skill specification."""
        suggested_name = str(payload.get("skill_name", "")).strip()
        if not suggested_name:
            return None
        skill_name = _ensure_unique_generated_name(suggested_name, skill_library)
        phases = list(candidate.get("phases", []))
        if not phases:
            return None

        template_type = str(payload.get("template_type", _template_type_from_phases(phases, prefer_parallel=False))).strip()
        if not _is_supported_template(template_type):
            template_type = _template_type_from_phases(phases, prefer_parallel=False)

        stages_raw = list(payload.get("stages", []))
        stages: List[Dict[str, Any]] = []
        for index, stage in enumerate(stages_raw[:5]):
            if not isinstance(stage, dict):
                continue
            stage_name = str(stage.get("stage_name", f"stage_{index + 1}")).strip() or f"stage_{index + 1}"
            stage_role = str(stage.get("stage_role", "")).strip() or f"stage_{index + 1}"
            prompt_template = str(stage.get("prompt_template", "")).strip()
            if not prompt_template:
                continue
            stages.append({
                "stage_name": stage_name,
                "stage_role": _coerce_stage_role(stage_role, prompt_template),
                "requires_action": _coerce_requires_action(stage.get("requires_action"), stage_role, prompt_template),
                "prompt_template": prompt_template,
            })
        if not stages:
            return None

        description = str(payload.get("description", f"Workflow: {' -> '.join(phases)}."))
        note = str(payload.get("note", ""))
        fixed_field = _coerce_generated_memory_field(
            payload.get("fixed_skill_memory_field", ""),
            skill_name=skill_name,
            skill_library=skill_library,
        )
        field_description = str(payload.get("dedicated_field_description", "")).strip()

        # Golden example selection
        golden_idx = int(payload.get("golden_example_index", 1)) - 1
        snippets = candidate.get("snippets", [])
        golden_id = ""
        if 0 <= golden_idx < len(snippets):
            golden_id = f"{skill_name}_golden_{golden_idx}"

        return SkillSpec(
            skill_name=skill_name,
            base_skill=phases[-1],
            source="pattern_reuse",
            selection_metadata={
                "description": description,
                "note": note or f"Use when the task benefits from a {' then '.join(phases)} workflow.",
                "golden_example_id": golden_id,
                "selection_tags": list(phases),
                "agent_count_hint": str(len(stages)),
            },
            detail={
                "template_type": template_type,
                "memory_top_k": 5,
                "stage_count": len(stages),
                "stages": stages,
            },
            phases=phases,
            metadata={
                "generation_mode": "pattern_reuse",
                "llm_generated": True,
                "llm_suggested_name": suggested_name,
                "fixed_skill_memory_field": fixed_field,
                "fixed_skill_memory_field_role": "dedicated_skill_decision_context",
                "dedicated_field_description": field_description,
                "lifecycle": _new_skill_lifecycle(),
            },
        )

    def _coerce_failure_novel_spec(
        self,
        payload: Dict[str, Any],
        candidate: Dict[str, Any],
        skill_library: Dict[str, SkillSpec],
    ) -> SkillSpec | None:
        """For failure_novel: LLM designs the full spec (solo or parallel)."""
        suggested_name = str(payload.get("skill_name", "")).strip()
        if not suggested_name:
            return None
        skill_name = _ensure_unique_generated_name(suggested_name, skill_library)

        template_type = str(payload.get("template_type", "solo")).strip()
        if not _is_supported_template(template_type):
            template_type = "solo"

        stages_raw = list(payload.get("stages", []))
        stages: List[Dict[str, Any]] = []
        for index, stage in enumerate(stages_raw[:3]):
            if not isinstance(stage, dict):
                continue
            stage_name = str(stage.get("stage_name", f"stage_{index + 1}")).strip() or f"stage_{index + 1}"
            stage_role = str(stage.get("stage_role", "")).strip() or f"stage_{index + 1}"
            prompt_template = str(stage.get("prompt_template", "")).strip()
            if not prompt_template:
                prompt_template = f"Run stage {index + 1} as {stage_role} for skill {skill_name}."
            stages.append({
                "stage_name": stage_name,
                "stage_role": _coerce_stage_role(stage_role, prompt_template),
                "requires_action": _coerce_requires_action(stage.get("requires_action"), stage_role, prompt_template),
                "prompt_template": prompt_template,
            })
        if not stages:
            stages = [{"stage_name": "stage_1", "stage_role": "execute", "requires_action": True,
                        "prompt_template": f"Execute the core action for skill {skill_name}."}]

        # Derive phases from stage roles
        phases = _dedupe([s["stage_role"] for s in stages])
        description = str(payload.get("description", f"Novel skill: {skill_name}."))
        note = str(payload.get("note", ""))
        fixed_field = _coerce_generated_memory_field(
            payload.get("fixed_skill_memory_field", ""),
            skill_name=skill_name,
            skill_library=skill_library,
        )
        field_description = str(payload.get("dedicated_field_description", "")).strip()

        return SkillSpec(
            skill_name=skill_name,
            base_skill=phases[-1] if phases else "execute",
            source="failure_novel",
            selection_metadata={
                "description": description,
                "note": note or "Use when existing workflows repeatedly fail.",
                "golden_example_id": "",
                "selection_tags": list(phases),
                "agent_count_hint": str(len(stages)),
            },
            detail={
                "template_type": template_type,
                "memory_top_k": 5,
                "stage_count": len(stages),
                "stages": stages,
            },
            phases=phases,
            metadata={
                "generation_mode": "failure_novel",
                "llm_generated": True,
                "llm_suggested_name": suggested_name,
                "fixed_skill_memory_field": fixed_field,
                "fixed_skill_memory_field_role": "dedicated_skill_decision_context",
                "dedicated_field_description": field_description,
                "lifecycle": _new_skill_lifecycle(),
            },
        )

    @staticmethod
    def _default_stages_from_phases(phases: List[str], skill_name: str, generation_mode: str) -> List[Dict[str, Any]]:
        stages: List[Dict[str, Any]] = []
        for index, phase in enumerate(phases[:5]):
            stage_role = phase.strip() or f"stage_role_{index + 1}"
            requires_action = _coerce_requires_action(None, stage_role, "")
            stages.append(
                {
                    "stage_name": f"stage_{index + 1}_{phase}",
                    "stage_role": stage_role,
                    "requires_action": requires_action,
                    "prompt_template": (
                        f"Run stage {index + 1} as {stage_role} for evolved skill {skill_name}. "
                        "Use special_notes for critical constraints and candidate-related caveats."
                    ),
                }
            )
        if generation_mode == "failure_novel":
            for stage in stages:
                stage["prompt_template"] += " Prioritize novel recovery strategy under persistent failure."
        return stages

    def _collect_failed_counterfactual_cases(
        self,
        failed_trajectories: List[TrajectoryRecord],
        counterfactual_records: List[CounterfactualRecord],
        skill_library: Dict[str, SkillSpec],
        max_trajectories: int = 2,
    ) -> List[Dict[str, Any]]:
        """Select the worst counterfactual trajectories (still failed), return with full serialized context."""
        failed_index = {trajectory.trajectory_id: trajectory for trajectory in failed_trajectories}
        failed_cf: List[Dict[str, Any]] = []
        for record in counterfactual_records:
            source = failed_index.get(record.source_trajectory_id)
            if not source:
                continue
            if record.success or _counterfactual_improved(source, record):
                continue
            cf_payload = _counterfactual_payload(record)
            cf_steps = cf_payload.get("steps", [])
            total_reward = sum(float(s.get("reward", 0.0)) for s in cf_steps) if cf_steps else record.average_reward
            failed_cf.append({
                "total_reward": total_reward,
                "record": record,
                "source_trajectory": source,
                "cf_steps": cf_steps,
            })
        failed_cf.sort(key=lambda x: x["total_reward"])

        trajectories_data: List[Dict[str, Any]] = []
        for item in failed_cf[:max_trajectories]:
            record = item["record"]
            source = item["source_trajectory"]
            cf_steps = item["cf_steps"]
            trajectories_data.append({
                "task_description": source.task.description,
                "task_goal": source.task.task_goal,
                "steps": [_serialize_step_brief(s, skill_library) for s in cf_steps],
            })

        if not trajectories_data:
            return []

        # Derive phases from the worst trajectory for skill generation
        worst = failed_cf[0] if failed_cf else None
        if worst:
            seed_phases, _ = _pivot_window_from_counterfactual(
                worst["source_trajectory"], worst["record"], worst["cf_steps"], skill_library
            )
        else:
            seed_phases = []
        if not seed_phases:
            seed_phases = ["reflect", "execute"]

        # Build existing skill details for prompt.
        existing_skill_details = _skill_metadata_for_prompt(skill_library)

        return [{
            "phases": seed_phases[:3],
            "source": "counterfactual_failed",
            "trajectories": trajectories_data,
            "existing_skill_details": existing_skill_details,
        }]

    def _collect_success_patterns(
        self,
        trajectories: Iterable[TrajectoryRecord],
        skill_library: Dict[str, SkillSpec],
        top_k: int = 2,
        snippets_per_pattern: int = 3,
    ) -> List[Dict[str, Any]]:
        """Stage-aware sliding window over success trajectories.

        Window slides over steps but total stage count must be 2-5.
        Window reward = sum(step rewards) / total_stages_in_window.
        Snippet selection favors earlier positions: occ_score = 2*reward + 1*early_ratio.
        """
        traj_list = list(trajectories)
        traj_index = {t.trajectory_id: t for t in traj_list}

        pattern_data: Dict[Tuple[str, ...], Dict[str, Any]] = defaultdict(
            lambda: {"score": 0.0, "occurrences": []}
        )
        for trajectory in traj_list:
            steps = list(trajectory.steps)
            total_steps = len(steps)
            stage_counts = [_step_stage_count(step, skill_library) for step in steps]
            step_scores = compute_step_scores(steps)

            for start in range(total_steps):
                total_stages = 0
                for end in range(start, total_steps):
                    total_stages += stage_counts[end]
                    if total_stages < 2:
                        continue
                    if total_stages > 5:
                        break
                    window_steps = steps[start: end + 1]
                    skill_seq = _collapsed_skill_sequence(window_steps, skill_library)
                    if len(skill_seq) < 2:
                        continue
                    key = tuple(skill_seq)
                    window_score = sum(step_scores[start: end + 1]) / total_stages
                    early_ratio = 1.0 - (start / max(1, total_steps))
                    occ_score = 2.0 * window_score + 1.0 * early_ratio
                    pattern_data[key]["score"] += window_score
                    pattern_data[key]["occurrences"].append({
                        "trajectory_id": trajectory.trajectory_id,
                        "start": start,
                        "end": end,
                        "total_stages": total_stages,
                        "early_ratio": early_ratio,
                        "occ_score": occ_score,
                    })

        results: List[Dict[str, Any]] = []
        for window, data in pattern_data.items():
            occurrences = data["occurrences"]
            raw_score = data["score"]
            avg_early = sum(o["early_ratio"] for o in occurrences) / max(1, len(occurrences))
            weighted = 2.0 * raw_score + 1.0 * avg_early
            results.append({
                "phases": list(window),
                "weighted_score": weighted,
                "source": "success_pattern",
                "raw_occurrences": occurrences,
            })
        results.sort(key=lambda x: x["weighted_score"], reverse=True)

        final: List[Dict[str, Any]] = []
        for pattern in results[:top_k]:
            occs = sorted(pattern["raw_occurrences"], key=lambda o: o["occ_score"], reverse=True)[:snippets_per_pattern]
            snippets: List[Dict[str, Any]] = []
            for occ in occs:
                traj = traj_index.get(occ["trajectory_id"])
                if not traj:
                    continue
                pre_window = [_serialize_step_brief(traj.steps[i], skill_library) for i in range(occ["start"])]
                window_detail = [
                    _serialize_step_detail(traj.steps[i], skill_library)
                    for i in range(occ["start"], occ["end"] + 1)
                    if i < len(traj.steps)
                ]
                snippets.append({
                    "task_description": traj.task.description,
                    "task_goal": traj.task.task_goal,
                    "pre_window_steps": pre_window,
                    "window_steps": window_detail,
                })
            final.append({
                "phases": pattern["phases"],
                "source": pattern["source"],
                "snippets": snippets,
            })
        return final


class SkillMemoryEvolutionEngine:
    def __init__(
        self,
        max_contrast_cases: int = 3,
        llm_client: Any | None = None,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.2,
    ) -> None:
        self.max_contrast_cases = max(1, int(max_contrast_cases))
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature

    def evolve(
        self,
        success_trajectories: List[TrajectoryRecord],
        failed_trajectories: List[TrajectoryRecord],
        counterfactual_records: List[CounterfactualRecord],
        skill_memory_store: SkillMemoryStore,
        skill_library: Dict[str, SkillSpec] | None = None,
    ) -> EvolutionArtifact:
        del success_trajectories
        lib = skill_library or {}
        if self.llm_client is None:
            return EvolutionArtifact(
                name="skill_memory_evolution",
                applied=False,
                summary="Skill memory evolution skipped because no LLM client is available.",
                payload={"entry_count": len(skill_memory_store.entries)},
            )

        contrast_cases = self._collect_pivot_contrast_cases(
            failed_trajectories=failed_trajectories,
            counterfactual_records=counterfactual_records,
            skill_library=lib,
        )
        if not contrast_cases:
            return EvolutionArtifact(
                name="skill_memory_evolution",
                applied=False,
                summary="Skill memory evolution found no successful counterfactual contrast pair.",
                payload={"entry_count": len(skill_memory_store.entries)},
            )

        current_metadata_fields = list(skill_memory_store.metadata_fields)
        existing_skill_metadata = _skill_metadata_for_prompt(lib)
        proposal = self._propose_metadata_fields(
            current_metadata_fields=current_metadata_fields,
            contrast_cases=contrast_cases,
            existing_skill_metadata=existing_skill_metadata,
        )
        if not proposal:
            return EvolutionArtifact(
                name="skill_memory_evolution",
                applied=False,
                summary="Skill memory evolution failed to produce a valid proposal.",
                payload={"entry_count": len(skill_memory_store.entries)},
            )

        new_fields = [str(f) for f in proposal.get("fields", []) if str(f).strip()]
        field_descriptions = dict(proposal.get("field_descriptions", {}))
        reason = str(proposal.get("rationale", "")).strip() or "pivot_contrastive_evolution"

        structure_changed = skill_memory_store.apply_structure_evolution(
            new_fields=new_fields,
            reason=reason,
        )

        return EvolutionArtifact(
            name="skill_memory_evolution",
            applied=bool(structure_changed),
            summary=(
                "Skill memory evolution updated the future memory_metadata field structure."
                if structure_changed
                else "Skill memory evolution ran but fields remained unchanged."
            ),
            payload={
                "entry_count": len(skill_memory_store.entries),
                "structure_changed": structure_changed,
                "new_fields": new_fields,
                "field_descriptions": field_descriptions,
                "rationale": reason,
            },
        )

    def _collect_pivot_contrast_cases(
        self,
        failed_trajectories: List[TrajectoryRecord],
        counterfactual_records: List[CounterfactualRecord],
        skill_library: Dict[str, SkillSpec],
    ) -> List[Dict[str, Any]]:
        """Collect up to max_contrast_cases pivot-based contrast pairs."""
        failed_index = {t.trajectory_id: t for t in failed_trajectories}
        cases: List[Dict[str, Any]] = []

        for record in counterfactual_records:
            if len(cases) >= self.max_contrast_cases:
                break
            source = failed_index.get(record.source_trajectory_id)
            if not source or not _counterfactual_improved(source, record):
                continue
            cf_payload = _counterfactual_payload(record)
            cf_steps = cf_payload.get("steps", [])
            if not cf_steps:
                continue

            _, pivot_meta = _pivot_window_from_counterfactual(source, record, cf_steps, skill_library)
            pivot_index = pivot_meta.get("pivot_index")
            if pivot_index is None:
                pivot_index = record.removal_start_step

            fork_step = max(1, record.removal_start_step)

            # Shared prefix: state_trace_summary up to fork + brief steps before fork
            shared_steps = [
                _serialize_step_brief(step, skill_library)
                for step in source.steps
                if step.step_index < fork_step
            ]
            shared_trace = str(source.steps[fork_step - 2].metadata.get("state_trace_summary", "")) if fork_step >= 2 and fork_step - 2 < len(source.steps) else ""

            # Failed branch: fork to pivot (brief)
            failed_to_pivot = [
                _serialize_step_brief(step, skill_library)
                for step in source.steps
                if fork_step <= step.step_index <= pivot_index
            ]
            # Success branch: fork to pivot (brief)
            cf_step_index = {int(s.get("step_index", -1)): s for s in cf_steps}
            success_to_pivot = [
                _serialize_step_brief(s, skill_library)
                for idx, s in sorted(cf_step_index.items())
                if fork_step <= idx <= pivot_index
            ]

            # Pivot step detailed (both branches)
            failed_pivot_step = [
                _serialize_step_detail(step, skill_library)
                for step in source.steps
                if step.step_index == pivot_index
            ]
            success_pivot_step = [
                _serialize_step_detail(s, skill_library)
                for s in cf_steps
                if int(s.get("step_index", -1)) == pivot_index
            ]

            cases.append({
                "task_goal": source.task.task_goal,
                "task_description": source.task.description,
                "shared_prefix_trace": shared_trace if shared_trace else "(no shared prefix)",
                "failed_steps_to_pivot": failed_to_pivot,
                "success_steps_to_pivot": success_to_pivot,
                "failed_pivot_step": failed_pivot_step,
                "success_pivot_step": success_pivot_step,
            })

        return cases

    def _propose_metadata_fields(
        self,
        current_metadata_fields: List[str],
        contrast_cases: List[Dict[str, Any]],
        existing_skill_metadata: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        system_prompt, user_prompt = build_skill_memory_structure_evolution_prompts(
            current_memory_metadata_fields=current_metadata_fields,
            contrast_cases=contrast_cases,
            existing_skill_metadata=existing_skill_metadata,
        )
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.llm_temperature,
                max_tokens=1200,
                top_p=1.0,
            )
            content = response.choices[0].message.content or "{}"
            payload = parse_json_response(content)
            if not isinstance(payload, dict):
                return None
            if not isinstance(payload.get("fields", []), list):
                return None
            return payload
        except Exception:
            return None


class ExecutorMemoryEvolutionEngine:
    def __init__(
        self,
        max_contrast_cases: int = 3,
        llm_client: Any | None = None,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.2,
    ) -> None:
        self.max_contrast_cases = max(1, int(max_contrast_cases))
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature

    def evolve(
        self,
        success_trajectories: List[TrajectoryRecord],
        failed_trajectories: List[TrajectoryRecord],
        counterfactual_records: List[CounterfactualRecord],
        executor_memories: Dict[str, ExecutorMemoryStore],
        skill_library: Dict[str, SkillSpec] | None = None,
    ) -> EvolutionArtifact:
        del success_trajectories
        lib = skill_library or {}
        if self.llm_client is None:
            return EvolutionArtifact(
                name="executor_memory_evolution",
                applied=False,
                summary="Executor memory evolution skipped because no LLM client is available.",
                payload={},
            )

        total_changed = 0
        changed_by_agent: Dict[str, int] = {}
        success_fields_by_agent: Dict[str, List[str]] = {}
        failure_fields_by_agent: Dict[str, List[str]] = {}
        existing_skill_metadata = _skill_metadata_for_prompt(lib)

        for agent_name, store in executor_memories.items():
            # Find counterfactuals where THIS agent was removed and task succeeded
            agent_cases = self._collect_pivot_cases_for_agent(
                agent_name=agent_name,
                failed_trajectories=failed_trajectories,
                counterfactual_records=counterfactual_records,
                skill_library=lib,
            )
            if not agent_cases:
                changed_by_agent[agent_name] = 0
                continue

            current_fields = list(store.metadata_fields)
            current_failure_fields = list(store.failure_metadata_fields)
            proposal = self._propose_metadata_fields(
                agent_name=agent_name,
                current_metadata_fields=current_fields,
                current_failure_metadata_fields=current_failure_fields,
                contrast_cases=agent_cases,
                existing_skill_metadata=existing_skill_metadata,
            )
            if not proposal:
                changed_by_agent[agent_name] = 0
                continue

            new_success_fields = [str(f) for f in proposal.get("success_fields", proposal.get("fields", [])) if str(f).strip()]
            new_failure_fields = [
                str(f)
                for f in proposal.get("failure_fields", current_failure_fields)
                if str(f).strip()
            ]
            if not new_success_fields and not new_failure_fields:
                changed_by_agent[agent_name] = 0
                continue

            changed = 0
            if new_success_fields:
                changed += store.apply_structure_evolution(new_fields=new_success_fields)
                success_fields_by_agent[agent_name] = new_success_fields
            if new_failure_fields:
                changed += store.set_failure_structure_fields(new_fields=new_failure_fields)
                failure_fields_by_agent[agent_name] = new_failure_fields
            changed_by_agent[agent_name] = changed
            total_changed += changed

        return EvolutionArtifact(
            name="executor_memory_evolution",
            applied=bool(total_changed),
            summary=(
                f"Executor memory evolution updated success/failure metadata templates for {sum(1 for v in changed_by_agent.values() if v > 0)} agents."
                if total_changed
                else "Executor memory evolution ran but no metadata template changed."
            ),
            payload={
                "changed_entries": total_changed,
                "total_changed": total_changed,
                "changed_by_agent": changed_by_agent,
                "success_fields_by_agent": success_fields_by_agent,
                "failure_fields_by_agent": failure_fields_by_agent,
            },
        )

    def _collect_pivot_cases_for_agent(
        self,
        agent_name: str,
        failed_trajectories: List[TrajectoryRecord],
        counterfactual_records: List[CounterfactualRecord],
        skill_library: Dict[str, SkillSpec],
    ) -> List[Dict[str, Any]]:
        """Collect contrast pairs where this specific agent was removed and the task improved."""
        failed_index = {t.trajectory_id: t for t in failed_trajectories}
        cases: List[Dict[str, Any]] = []

        for record in counterfactual_records:
            if len(cases) >= self.max_contrast_cases:
                break
            if record.removed_agent != agent_name:
                continue
            source = failed_index.get(record.source_trajectory_id)
            if not source or not _counterfactual_improved(source, record):
                continue
            cf_payload = _counterfactual_payload(record)
            cf_steps = cf_payload.get("steps", [])
            if not cf_steps:
                continue

            _, pivot_meta = _pivot_window_from_counterfactual(source, record, cf_steps, skill_library)
            pivot_index = pivot_meta.get("pivot_index")
            if pivot_index is None:
                pivot_index = record.removal_start_step

            fork_step = max(1, record.removal_start_step)

            shared_trace = ""
            if fork_step >= 2 and fork_step - 2 < len(source.steps):
                shared_trace = str(source.steps[fork_step - 2].metadata.get("state_trace_summary", ""))

            failed_to_pivot = [
                _serialize_step_brief(step, skill_library)
                for step in source.steps
                if fork_step <= step.step_index <= pivot_index
            ]
            cf_step_index = {int(s.get("step_index", -1)): s for s in cf_steps}
            success_to_pivot = [
                _serialize_step_brief(s, skill_library)
                for idx, s in sorted(cf_step_index.items())
                if fork_step <= idx <= pivot_index
            ]
            failed_pivot_step = [
                _serialize_step_detail(step, skill_library)
                for step in source.steps
                if step.step_index == pivot_index
            ]
            success_pivot_step = [
                _serialize_step_detail(s, skill_library)
                for s in cf_steps
                if int(s.get("step_index", -1)) == pivot_index
            ]

            cases.append({
                "task_goal": source.task.task_goal,
                "task_description": source.task.description,
                "shared_prefix_trace": shared_trace if shared_trace else "(no shared prefix)",
                "failed_steps_to_pivot": failed_to_pivot,
                "success_steps_to_pivot": success_to_pivot,
                "failed_pivot_step": failed_pivot_step,
                "success_pivot_step": success_pivot_step,
            })

        return cases

    def _propose_metadata_fields(
        self,
        agent_name: str,
        current_metadata_fields: List[str],
        current_failure_metadata_fields: List[str],
        contrast_cases: List[Dict[str, Any]],
        existing_skill_metadata: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        system_prompt, user_prompt = build_executor_memory_structure_evolution_prompts(
            agent_name=agent_name,
            current_memory_metadata_fields=current_metadata_fields,
            current_failure_metadata_fields=current_failure_metadata_fields,
            contrast_cases=contrast_cases,
            existing_skill_metadata=existing_skill_metadata,
        )
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.llm_temperature,
                max_tokens=1200,
                top_p=1.0,
            )
            content = response.choices[0].message.content or "{}"
            payload = parse_json_response(content)
            if not isinstance(payload, dict):
                return None
            success_fields = payload.get("success_fields", payload.get("fields", []))
            failure_fields = payload.get("failure_fields", current_failure_metadata_fields)
            if not isinstance(success_fields, list):
                return None
            if not isinstance(failure_fields, list):
                return None
            payload["success_fields"] = success_fields
            payload["failure_fields"] = failure_fields
            return payload
        except Exception:
            return None


def _base_skill_name(skill_name: str, skill_library: Dict[str, SkillSpec]) -> str:
    spec = skill_library.get(skill_name)
    return spec.base_skill if spec else skill_name


def _counterfactual_payload(record: CounterfactualRecord) -> Dict[str, Any]:
    payload = record.metadata.get("counterfactual_trajectory", {})
    return payload if isinstance(payload, dict) else {}


def _counterfactual_step(record: CounterfactualRecord, step_index: int) -> Dict[str, Any]:
    payload = _counterfactual_payload(record)
    steps = payload.get("steps", [])
    for step in steps:
        if int(step.get("step_index", -1)) == step_index:
            return step
    if 0 < step_index <= len(steps):
        return steps[step_index - 1]
    return {}


def _counterfactual_improved(source: TrajectoryRecord, record: CounterfactualRecord) -> bool:
    return bool(record.success and not source.success) or (record.average_reward > source.average_reward + 0.05)


def _pivot_window_from_counterfactual(
    source: TrajectoryRecord,
    record: CounterfactualRecord,
    steps: List[Dict[str, Any]],
    skill_library: Dict[str, SkillSpec],
) -> Tuple[List[str], Dict[str, Any]]:
    """Find pivot where counterfactual exploration score surpasses original.

    Uses cumulative exploration score diff instead of reward diff.
    """
    start = max(1, int(record.removal_start_step or 1))
    cf_index = {int(step.get("step_index", -1)): step for step in steps if isinstance(step, dict)}
    source_index = {step.step_index: step for step in source.steps}

    source_scores = compute_step_scores(source.steps)
    source_score_map = {step.step_index: sc for step, sc in zip(source.steps, source_scores)}
    cf_scores = compute_step_scores(steps)
    cf_score_map = {int(s.get("step_index", -1)): sc for s, sc in zip(steps, cf_scores)}

    cumulative_diff = 0.0
    pivot_step = None
    max_idx = max(len(source.steps), max(cf_index.keys(), default=0))
    for step_idx in range(start, max_idx + 1):
        cf_sc = cf_score_map.get(step_idx, 0.0)
        src_sc = source_score_map.get(step_idx, 0.0)
        cumulative_diff += cf_sc - src_sc
        if cumulative_diff > 0 and pivot_step is None:
            pivot_step = step_idx

    if pivot_step is None:
        return [], {}

    best_window: List[str] = []
    sorted_indices = sorted(idx for idx in cf_index if idx >= pivot_step)
    total_stages = 0
    window_steps_indices: List[int] = []
    for idx in sorted_indices:
        s = cf_index[idx]
        sc = _step_stage_count(s, skill_library)
        if total_stages + sc > 5:
            break
        total_stages += sc
        window_steps_indices.append(idx)
        if total_stages >= 2:
            seq = _collapsed_skill_sequence([cf_index[i] for i in window_steps_indices], skill_library)
            if len(seq) >= 2:
                best_window = seq

    return (
        best_window[:5],
        {
            "pivot_index": pivot_step,
            "cumulative_diff_at_pivot": cumulative_diff,
            "removal_start_step": start,
        },
    )


def _nested_get(payload: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _ensure_unique_generated_name(suggested_name: str, skill_library: Dict[str, SkillSpec]) -> str:
    base_name = re.sub(r"[^a-zA-Z0-9_]+", "_", suggested_name).strip("_") or "evolved_skill"
    if base_name not in skill_library:
        return base_name
    suffix = 2
    while f"{base_name}_{suffix}" in skill_library:
        suffix += 1
    return f"{base_name}_{suffix}"


def _coerce_generated_memory_field(
    field_name: Any,
    skill_name: str,
    skill_library: Dict[str, SkillSpec],
) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(field_name or "").strip().lower()).strip("_")
    if not normalized or normalized.startswith("skill_focus_"):
        normalized = re.sub(r"[^a-z0-9]+", "_", str(skill_name).strip().lower()).strip("_")
        normalized = f"{normalized}_decision_context" if normalized else "skill_decision_context"
    used = {
        str(dict(spec.metadata or {}).get("fixed_skill_memory_field", "")).strip()
        for spec in skill_library.values()
    }
    if normalized not in used:
        return normalized
    suffix = re.sub(r"[^a-z0-9]+", "_", str(skill_name).strip().lower()).strip("_") or "skill"
    candidate = f"{normalized}_{suffix}"
    if candidate not in used:
        return candidate
    index = 2
    while f"{candidate}_{index}" in used:
        index += 1
    return f"{candidate}_{index}"


def _new_skill_lifecycle() -> Dict[str, Any]:
    return {
        "status": "probation",
        "trial_rounds": 0,
        "failed_promotion_rounds": 0,
        "last_task_count": 0,
        "last_success_count": 0,
        "last_success_rate": "N/A",
        "last_average_reward": "N/A",
    }


def _template_type_from_phases(phases: List[str], prefer_parallel: bool = False) -> str:
    length = max(1, len(phases))
    if length == 1:
        return "solo"
    if length == 2:
        return "parallel_2" if prefer_parallel else "sequential_2"
    if length == 3:
        return "parallel_3" if prefer_parallel else "sequential_3"
    return "parallel_chain" if prefer_parallel else "sequential_chain"


def _is_supported_template(template_type: str) -> bool:
    if template_type == "solo":
        return True
    if template_type.startswith("sequential_"):
        return True
    if template_type.startswith("parallel_"):
        return True
    return False


def _coerce_stage_role(stage_role: str, prompt_template: str) -> str:
    role = stage_role.strip()
    if role:
        return role
    prompt = prompt_template.lower()
    if any(token in prompt for token in ["take an action", "proposed_action", "environment action", "execute"]):
        return "action_stage"
    if any(token in prompt for token in ["plan", "subgoal", "decompose"]):
        return "planning_stage"
    if any(token in prompt for token in ["review", "risk", "repair", "reflect", "critique"]):
        return "review_stage"
    return "custom_stage"


def _coerce_requires_action(value: Any, stage_role: str, prompt_template: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    role_lower = stage_role.strip().lower()
    if role_lower in {"execute", "action", "action_stage"}:
        return True
    prompt = prompt_template.lower()
    if any(token in prompt for token in ["requires taking an environment action", "proposed_action must", "must take an action now"]):
        return True
    return False


def _step_stage_count(step: Any, skill_library: Dict[str, SkillSpec]) -> int:
    """Get the number of stages in a step. Uses executor_outputs length or skill spec stage_count."""
    if isinstance(step, dict):
        outputs = step.get("executor_outputs", {})
        skill_name = str(_nested_get(step, ["decision", "skill_name"], default=""))
    else:
        outputs = step.executor_outputs
        skill_name = step.decision.skill_name
    count = len(outputs) if isinstance(outputs, dict) else 0
    if count > 0:
        return count
    spec = skill_library.get(skill_name)
    if spec:
        return max(1, int(spec.detail.get("stage_count", len(spec.detail.get("stages", [1])))))
    return 1


def _collapsed_skill_sequence(steps: list, skill_library: Dict[str, SkillSpec]) -> List[str]:
    """Get the skill sequence from steps while preserving repeated skills."""
    seq: List[str] = []
    for step in steps:
        if isinstance(step, dict):
            name = str(_nested_get(step, ["decision", "skill_name"], default=""))
        else:
            name = step.decision.skill_name
        base = _base_skill_name(name, skill_library)
        if base:
            seq.append(base)
    return seq


def _serialize_step_brief(step: Any, skill_library: Dict[str, SkillSpec]) -> Dict[str, Any]:
    """Brief step: skill_name, assignment, stage outputs/actions, env_feedback only."""
    if isinstance(step, dict):
        skill_name = str(_nested_get(step, ["decision", "skill_name"], default=""))
        assignment = _nested_get(step, ["decision", "assignment"], default={})
        outputs = step.get("executor_outputs", {})
        metadata = step.get("metadata", {})
    else:
        skill_name = step.decision.skill_name
        assignment = step.decision.assignment
        outputs = step.executor_outputs
        metadata = step.metadata

    base_skill = _base_skill_name(skill_name, skill_library)
    env_feedback = str(metadata.get("environment_feedback", "")).strip()

    stages: List[Dict[str, Any]] = []
    if isinstance(outputs, dict):
        for agent_name, payload in outputs.items():
            if not isinstance(payload, dict):
                continue
            stages.append({
                "stage_name": str(payload.get("stage_name", "")),
                "agent": agent_name,
                "output": str(payload.get("output", "")).strip()[:200],
                "proposed_action": str(payload.get("proposed_action", "")).strip(),
            })

    return {
        "skill_name": base_skill,
        "assignment": dict(assignment) if isinstance(assignment, dict) else {},
        "stages": stages,
        "environment_feedback": env_feedback or "No environment action in this step.",
    }


def _serialize_step_detail(step: Any, skill_library: Dict[str, SkillSpec]) -> Dict[str, Any]:
    """Detailed step: adds subgoal, stage_subgoals, special_notes on top of brief."""
    if isinstance(step, dict):
        skill_name = str(_nested_get(step, ["decision", "skill_name"], default=""))
        subgoal = str(_nested_get(step, ["decision", "subgoal"], default=""))
        assignment = _nested_get(step, ["decision", "assignment"], default={})
        stage_subgoals = _nested_get(step, ["decision", "stage_subgoals"], default={})
        outputs = step.get("executor_outputs", {})
        metadata = step.get("metadata", {})
    else:
        skill_name = step.decision.skill_name
        subgoal = step.decision.subgoal
        assignment = step.decision.assignment
        stage_subgoals = step.decision.stage_subgoals
        outputs = step.executor_outputs
        metadata = step.metadata

    base_skill = _base_skill_name(skill_name, skill_library)
    env_feedback = str(metadata.get("environment_feedback", "")).strip()

    stages: List[Dict[str, Any]] = []
    if isinstance(outputs, dict):
        for agent_name, payload in outputs.items():
            if not isinstance(payload, dict):
                continue
            stage_name = str(payload.get("stage_name", ""))
            stages.append({
                "stage_name": stage_name,
                "agent": agent_name,
                "output": str(payload.get("output", "")).strip()[:200],
                "proposed_action": str(payload.get("proposed_action", "")).strip(),
                "special_notes": [str(n).strip() for n in (payload.get("special_notes", []) or []) if str(n).strip()],
                "stage_subgoal": str(stage_subgoals.get(stage_name, "") if isinstance(stage_subgoals, dict) else ""),
            })

    return {
        "skill_name": base_skill,
        "subgoal": subgoal,
        "assignment": dict(assignment) if isinstance(assignment, dict) else {},
        "stages": stages,
        "environment_feedback": env_feedback or "No environment action in this step.",
    }
