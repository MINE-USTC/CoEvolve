from __future__ import annotations

from dataclasses import asdict
import json
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..schema import SkillMemoryEntry, TrajectoryRecord
from ..storage.io import load_json
from .base import BaseMemoryStore
from .embedding import get_embedder


_LEGACY_SKILL_STRUCTURE_FIELDS = [
    "task_description",
    "task_goal",
    "state_trace_summary",
    "memory_metadata",
    "recent_environment_feedback",
    "subgoal",
    "stage_assignments",
    "stage_subgoals",
    "skill_name",
    "success",
]

_DEFAULT_SKILL_METADATA_FIELDS = ["lesson"]


class SkillMemoryStore(BaseMemoryStore):
    def __init__(
        self,
        storage_path: str,
        initial_score: float = 0.5,
        score_lambda: float = 0.2,
        score_omega: float = 0.05,
        min_score_to_keep: float = 0.05,
        structure_fields: List[str] | None = None,
    ) -> None:
        super().__init__(storage_path)
        self.entries: List[SkillMemoryEntry] = []
        self.initial_score = max(0.0, float(initial_score))
        self.score_lambda = max(0.0, min(1.0, float(score_lambda)))
        self.score_omega = max(0.0, float(score_omega))
        self.min_score_to_keep = max(0.0, float(min_score_to_keep))
        self.structure_fields = self._normalize_fields(
            structure_fields or list(_DEFAULT_SKILL_METADATA_FIELDS)
        )
        self.structure_field_mapping: Dict[str, str] = {}
        self.total_retrieval_count = 0
        self._lock = threading.Lock()
        self._snapshot_filename = "skill_memory.json"
        self._updates_filename = "skill_memory_updates.jsonl"
        self._hybrid_weights: Dict[str, float] = {"text": 0.35, "semantic": 0.65}
        self.fixed_field_names: set[str] = set()
        self._index_dirty = True
        self._entry_text_cache: Dict[str, str] = {}
        self._entry_embedding_cache: Dict[str, np.ndarray] = {}
        self._query_embedding_cache: Dict[str, np.ndarray] = {}
        self._index_entry_ids: List[str] = []
        self._entry_position: Dict[str, int] = {}
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._embedding_matrix: np.ndarray | None = None

    def retrieve(self, query: str, state: str, top_k: int) -> List[Dict[str, object]]:
        ranked = self._rank_entries(
            task_description=query,
            task_goal="",
            state_trace_summary="",
            memory_metadata=self._build_query_metadata_template(self.metadata_fields),
            recent_environment_feedback=state,
            subgoal="",
            entries=self.entries,
            top_k=top_k,
        )
        return [payload for payload, _ in ranked[:top_k]]

    @property
    def metadata_fields(self) -> List[str]:
        inferred = self._infer_metadata_fields_from_entries()
        if self._is_default_metadata_template(self.structure_fields) and inferred:
            return inferred
        if self.structure_fields and not self._looks_like_legacy_structure_fields(self.structure_fields):
            return list(self.structure_fields)
        if inferred:
            return inferred
        return list(_DEFAULT_SKILL_METADATA_FIELDS)

    def retrieve_for_skill(
        self,
        skill_name: str,
        task_description: str,
        task_goal: str,
        state_trace_summary: str,
        memory_metadata: Dict[str, Any] | None,
        recent_environment_feedback: str,
        subgoal: str,
        top_k: int,
        golden_example_id: str = "",
    ) -> List[Dict[str, object]]:
        with self._lock:
            entries = [item for item in self.entries if item.skill_name == skill_name]
        ranked = self._rank_entries(
            task_description=task_description,
            task_goal=task_goal,
            state_trace_summary=state_trace_summary,
            memory_metadata=memory_metadata or {},
            recent_environment_feedback=recent_environment_feedback,
            subgoal=subgoal,
            entries=entries,
            top_k=top_k,
        )
        results = [payload for payload, _ in ranked[:top_k]]
        if golden_example_id:
            golden = next((item for item in entries if item.entry_id == golden_example_id), None)
            if golden is not None:
                payload = asdict(golden)
                payload["_retrieval_meta"] = {
                    "source": "golden_example",
                    "similarity": 1.0,
                    "memory_score": float(golden.weight or self.initial_score),
                }
                deduped = [item for item in results if str(item.get("entry_id", "")) != golden_example_id]
                results = [payload] + deduped
                results = results[:top_k]
        return results

    def update_from_trajectory(
        self,
        trajectory: TrajectoryRecord,
        selected_steps: List[int] | None = None,
        llm_metadata: Dict[int, Dict[str, Any]] | None = None,
        fixed_fields_by_step: Dict[int, str] | None = None,
    ) -> Dict[str, List[str]]:
        """Create entries for selected steps with LLM-filled metadata.

        Args:
            trajectory: completed trajectory
            selected_steps: step indices to create entries for (if None, use all)
            llm_metadata: {step_index: {field: value}} from LLM, merged into memory_metadata
        """
        added_entry_ids: List[str] = []
        metadata_map = llm_metadata or {}
        fixed_field_map = fixed_fields_by_step or {}
        steps_to_add = selected_steps if selected_steps is not None else [s.step_index for s in trajectory.steps]
        step_index_map = {s.step_index: s for s in trajectory.steps}
        expected_fields = [
            field for field in self.metadata_fields
            if field not in self.fixed_field_names
        ]

        with self._lock:
            for step_idx in steps_to_add:
                step = step_index_map.get(step_idx)
                if step is None:
                    continue
                raw_meta = dict(metadata_map.get(step_idx, {}) or {})
                allowed_fixed_field = str(fixed_field_map.get(step_idx, "")).strip()
                meta = self._normalize_memory_metadata(
                    metadata=raw_meta,
                    expected_fields=expected_fields,
                )
                if allowed_fixed_field:
                    for fixed_field in self.fixed_field_names:
                        if fixed_field != allowed_fixed_field:
                            meta.pop(fixed_field, None)
                meta.update(self._extract_fixed_memory_metadata(raw_meta, allowed_field=allowed_fixed_field))
                self.entries.append(
                    SkillMemoryEntry(
                        entry_id=str(uuid4()),
                        skill_name=step.decision.skill_name,
                        task_description=trajectory.task.description,
                        task_goal=trajectory.task.task_goal,
                        subgoal=step.decision.subgoal,
                        state_trace_summary=str(step.metadata.get("state_trace_summary", "")),
                        recent_environment_feedback=str(step.metadata.get("environment_feedback", "")),
                        stage_assignments=dict(step.decision.assignment),
                        stage_subgoals=dict(step.decision.stage_subgoals),
                        success=trajectory.success,
                        weight=self.initial_score,
                        memory_metadata=meta,
                    )
                )
                added_entry_ids.append(self.entries[-1].entry_id)
            if added_entry_ids:
                self._mark_index_dirty()
            updated_entry_ids = self._update_retrieved_scores_from_trajectory(trajectory)
            removed_entry_ids = self.prune_low_score(self.min_score_to_keep)
        return {
            "added_entry_ids": added_entry_ids,
            "updated_entry_ids": updated_entry_ids,
            "removed_entry_ids": removed_entry_ids,
            "touched_entry_ids": self._dedupe_ids(added_entry_ids + updated_entry_ids),
        }

    def snapshot(self) -> Dict[str, object]:
        return {
            "entry_count": len(self.entries),
            "initial_score": self.initial_score,
            "score_lambda": self.score_lambda,
            "score_omega": self.score_omega,
            "min_score_to_keep": self.min_score_to_keep,
            "structure_fields": list(self.structure_fields),
            "structure_field_mapping": dict(self.structure_field_mapping),
            "fixed_field_names": sorted(self.fixed_field_names),
            "total_retrieval_count": self.total_retrieval_count,
            "entries": [asdict(item) for item in self.entries],
        }

    def content_snapshot(self) -> Dict[str, object]:
        return {
            "entry_count": len(self.entries),
            "initial_score": self.initial_score,
            "score_lambda": self.score_lambda,
            "score_omega": self.score_omega,
            "min_score_to_keep": self.min_score_to_keep,
            "total_retrieval_count": self.total_retrieval_count,
            "entries": [asdict(item) for item in self.entries],
        }

    def structure_snapshot(self) -> Dict[str, object]:
        return {
            "structure_fields": list(self.structure_fields),
            "structure_field_mapping": dict(self.structure_field_mapping),
            "fixed_field_names": sorted(self.fixed_field_names),
        }

    def load_snapshot(self, payload: Dict[str, object] | None) -> None:
        self.entries = []
        if not payload:
            self._mark_index_dirty()
            return
        self.initial_score = max(0.0, float(payload.get("initial_score", self.initial_score)))
        self.score_lambda = max(0.0, min(1.0, float(payload.get("score_lambda", self.score_lambda))))
        self.score_omega = max(0.0, float(payload.get("score_omega", self.score_omega)))
        self.min_score_to_keep = max(0.0, float(payload.get("min_score_to_keep", self.min_score_to_keep)))
        self.structure_fields = self._normalize_fields(
            list(payload.get("structure_fields", self.structure_fields))
        )
        self.structure_field_mapping = {
            str(k): str(v)
            for k, v in dict(payload.get("structure_field_mapping", self.structure_field_mapping)).items()
            if str(k).strip() and str(v).strip()
        }
        self.fixed_field_names = {
            str(item).strip()
            for item in list(payload.get("fixed_field_names", []) or [])
            if str(item).strip()
        }
        self.total_retrieval_count = int(payload.get("total_retrieval_count", self.total_retrieval_count))
        for item in payload.get("entries", []):
            entry = self._coerce_entry(item)
            self.entries.append(entry)
        if self._looks_like_legacy_structure_fields(self.structure_fields):
            inferred = self._infer_metadata_fields_from_entries()
            if inferred:
                self.structure_fields = inferred
        self._mark_index_dirty()

    def get_entry_by_id(self, entry_id: str) -> Dict[str, object] | None:
        if not entry_id:
            return None
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return asdict(entry)
        return None

    def clear_entries_keep_structure(self) -> None:
        self.entries = []
        self.total_retrieval_count = 0
        self._mark_index_dirty()

    def load_from_storage(self) -> bool:
        snapshot_path = self._snapshot_path()
        updates_path = self._updates_path()
        if snapshot_path is None:
            return False
        payload = load_json(snapshot_path, default=None)
        if payload:
            self.load_snapshot(payload)
        else:
            self.entries = []
        if updates_path is not None and updates_path.exists():
            self._apply_incremental_updates()
        self._mark_index_dirty()
        return bool(payload) or (updates_path is not None and updates_path.exists())

    def persist_incremental(
        self,
        touched_entry_ids: List[str],
        removed_entry_ids: List[str] | None = None,
        reason: str = "",
    ) -> None:
        updates_path = self._updates_path()
        if updates_path is None:
            return
        touched = self._dedupe_ids(touched_entry_ids)
        removed = self._dedupe_ids(removed_entry_ids or [])
        if not touched and not removed:
            return
        index = {entry.entry_id: entry for entry in self.entries}
        entries_payload = [asdict(index[entry_id]) for entry_id in touched if entry_id in index]
        operation = {
            "op": "upsert",
            "entries": entries_payload,
            "deleted_entry_ids": removed,
            "reason": reason,
        }
        updates_path.parent.mkdir(parents=True, exist_ok=True)
        with open(updates_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(operation, ensure_ascii=False) + "\n")

    def persist_full_rewrite(self, reason: str = "") -> None:
        snapshot_path = self._snapshot_path()
        updates_path = self._updates_path()
        if snapshot_path is None or updates_path is None:
            return
        payload = self.snapshot()
        payload["rewrite_reason"] = reason
        self.persist_snapshot(self._snapshot_filename, payload=payload)
        updates_path.parent.mkdir(parents=True, exist_ok=True)
        with open(updates_path, "w", encoding="utf-8") as handle:
            handle.write("")

    def best_entry_ids_by_skill(self) -> Dict[str, str]:
        best: Dict[str, SkillMemoryEntry] = {}
        for entry in self.entries:
            current = best.get(entry.skill_name)
            if current is None or entry.weight > current.weight:
                best[entry.skill_name] = entry
        return {skill: entry.entry_id for skill, entry in best.items()}

    def apply_structure_evolution(
        self,
        new_fields: List[str],
        field_mapping: Dict[str, str] | None = None,
        reason: str = "",
        source_case_ids: List[str] | None = None,
    ) -> int:
        normalized_fields = self._normalize_fields(new_fields)
        if not normalized_fields:
            return 0
        mapping = {str(k): str(v) for k, v in (field_mapping or {}).items() if str(k).strip() and str(v).strip()}
        changed = int(
            normalized_fields != self.metadata_fields
            or mapping != self.structure_field_mapping
        )
        self.structure_fields = normalized_fields
        self.structure_field_mapping = dict(mapping)
        return changed

    def register_fixed_field(self, field_name: str) -> None:
        key = str(field_name).strip()
        if not key:
            return
        self.fixed_field_names.add(key)

    def prune_low_score(self, threshold: float | None = None) -> List[str]:
        floor = self.min_score_to_keep if threshold is None else max(0.0, float(threshold))
        removed = [entry.entry_id for entry in self.entries if float(entry.weight) < floor]
        if not removed:
            return []
        removed_set = set(removed)
        self.entries = [entry for entry in self.entries if entry.entry_id not in removed_set]
        self._mark_index_dirty()
        return removed

    def _rank_entries(
        self,
        task_description: str,
        task_goal: str,
        state_trace_summary: str,
        memory_metadata: Dict[str, Any],
        recent_environment_feedback: str,
        subgoal: str,
        entries: List[SkillMemoryEntry],
        top_k: int,
    ) -> List[Tuple[Dict[str, object], float]]:
        if not entries:
            return []
        entries = list(entries)  # snapshot to avoid concurrent modification
        self._ensure_index()
        embedder = get_embedder()
        query_text = self._retrieval_text_from_fields(
            task_description=task_description,
            task_goal=task_goal,
            state_trace_summary=state_trace_summary,
            memory_metadata=memory_metadata,
            recent_environment_feedback=recent_environment_feedback,
            subgoal=subgoal,
        )
        candidate_pairs = [
            (entry, self._entry_position.get(entry.entry_id))
            for entry in entries
            if entry.entry_id in self._entry_position
        ]
        if not candidate_pairs:
            return []
        candidate_entries = [entry for entry, _ in candidate_pairs]
        row_indexes = [int(pos) for _, pos in candidate_pairs if pos is not None]
        query_emb = self._encode_query_embedding(query_text, embedder)
        semantic_scores = cosine_similarity([query_emb], self._embedding_matrix[row_indexes])[0]
        tfidf_scores = np.zeros(len(candidate_entries), dtype=float)
        if self._tfidf_vectorizer is not None and self._tfidf_matrix is not None:
            query_vec = self._tfidf_vectorizer.transform([query_text])
            tfidf_scores = cosine_similarity(query_vec, self._tfidf_matrix[row_indexes]).flatten()
        combined_scores = (
            self._hybrid_weights["text"] * tfidf_scores
            + self._hybrid_weights["semantic"] * semantic_scores
        )
        candidate_count = max(1, min(len(candidate_entries), top_k * 4))
        candidate_indexes = sorted(
            range(len(candidate_entries)),
            key=lambda index: combined_scores[index],
            reverse=True,
        )[:candidate_count]
        top_slice = [combined_scores[index] for index in candidate_indexes]
        sim_min, sim_max = min(top_slice), max(top_slice)
        sim_span = max(0.01, sim_max - sim_min)
        top_weights = [float(candidate_entries[index].weight or self.initial_score) for index in candidate_indexes]
        min_w, max_w = min(top_weights), max(top_weights)
        ranked: List[Tuple[Dict[str, object], float]] = []
        for index in candidate_indexes:
            entry = candidate_entries[index]
            sim = float(combined_scores[index])
            bias = float(entry.weight or self.initial_score)
            if max_w > min_w:
                bias_norm = (bias - min_w) / (max_w - min_w)
            else:
                bias_norm = 0.5
            bias_scaled = (bias_norm * 2.0 - 1.0) * (sim_span / 2.0)
            bonus = self.score_omega * math.sqrt(
                max(0.0, math.log(max(1, self.total_retrieval_count)) / max(1, entry.retrieval_count))
            )
            score = float(sim) + bias_scaled + float(bonus)
            payload = asdict(entry)
            payload["_retrieval_meta"] = {
                "similarity": float(sim),
                "text_score": float(tfidf_scores[index]),
                "semantic_score": float(semantic_scores[index]),
                "combined_score": float(sim),
                "memory_score": bias,
                "memory_score_scaled": float(bias_scaled),
                "exploration_bonus": float(bonus),
                "score": score,
            }
            ranked.append((payload, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _retrieval_text_from_fields(
        self,
        task_description: str,
        task_goal: str,
        state_trace_summary: str,
        memory_metadata: Dict[str, Any],
        recent_environment_feedback: str,
        subgoal: str,
    ) -> str:
        normalized_metadata = {
            str(key): value
            for key, value in dict(memory_metadata or {}).items()
            if str(key).strip()
            and not str(key).startswith("_")
            and str(key).strip() in self.fixed_field_names
        }
        parts = [
            f"task_description: {str(task_description).strip()}",
            f"task_goal: {str(task_goal).strip()}",
            f"state_trace_summary: {str(state_trace_summary).strip()}",
            f"memory_metadata: {json.dumps(normalized_metadata, ensure_ascii=False, sort_keys=True)}",
            f"recent_environment_feedback: {str(recent_environment_feedback).strip()}",
            f"subgoal: {str(subgoal).strip()}",
        ]
        return "\n".join([item for item in parts if item.split(":", 1)[1].strip()])

    def _retrieval_text_from_entry(self, entry: SkillMemoryEntry) -> str:
        return self._retrieval_text_from_fields(
            task_description=entry.task_description,
            task_goal=entry.task_goal,
            state_trace_summary=entry.state_trace_summary,
            memory_metadata=entry.memory_metadata,
            recent_environment_feedback=entry.recent_environment_feedback,
            subgoal=entry.subgoal,
        )

    def _update_retrieved_scores_from_trajectory(self, trajectory: TrajectoryRecord) -> List[str]:
        retrieval_counts: Dict[str, int] = {}
        for step in trajectory.steps:
            for hit in step.metadata.get("skill_detail_memory_hits", []) or []:
                entry_id = str(hit.get("entry_id", "")).strip()
                if not entry_id:
                    continue
                retrieval_counts[entry_id] = retrieval_counts.get(entry_id, 0) + 1
        if not retrieval_counts:
            return []

        task_score = self._task_score_from_trajectory(trajectory)
        entry_index = {entry.entry_id: entry for entry in self.entries}
        updated_ids: List[str] = []
        for entry_id, count in retrieval_counts.items():
            entry = entry_index.get(entry_id)
            if entry is None:
                continue
            updated_ids.append(entry_id)
            for _ in range(count):
                self.total_retrieval_count += 1
                entry.retrieval_count += 1
                ema = (1.0 - self.score_lambda) * float(entry.weight) + self.score_lambda * task_score
                entry.weight = max(0.0, ema)
        self._mark_index_dirty()
        return self._dedupe_ids(updated_ids)

    def _mark_index_dirty(self) -> None:
        self._index_dirty = True

    def _ensure_index(self) -> None:
        if self._index_dirty:
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        previous_text_cache = dict(self._entry_text_cache)
        entries = list(self.entries)
        self._entry_text_cache = {}
        self._index_entry_ids = []
        self._entry_position = {}
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._embedding_matrix = None
        if not entries:
            self._index_dirty = False
            return

        corpus: List[str] = []
        for entry in entries:
            text = self._retrieval_text_from_entry(entry)
            self._entry_text_cache[entry.entry_id] = text
            self._index_entry_ids.append(entry.entry_id)
            corpus.append(text)
        self._entry_position = {entry_id: index for index, entry_id in enumerate(self._index_entry_ids)}
        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)
        missing_ids: List[str] = []
        missing_texts: List[str] = []
        for entry_id, text in zip(self._index_entry_ids, corpus):
            if entry_id not in self._entry_embedding_cache or previous_text_cache.get(entry_id) != text:
                missing_ids.append(entry_id)
                missing_texts.append(text)
        if missing_texts:
            embeddings = get_embedder().encode(
                missing_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for entry_id, embedding in zip(missing_ids, embeddings):
                self._entry_embedding_cache[entry_id] = np.asarray(embedding, dtype=float)
        active_ids = set(self._index_entry_ids)
        self._entry_embedding_cache = {
            entry_id: embedding
            for entry_id, embedding in self._entry_embedding_cache.items()
            if entry_id in active_ids
        }
        self._embedding_matrix = np.vstack([self._entry_embedding_cache[entry_id] for entry_id in self._index_entry_ids])
        self._index_dirty = False

    def _encode_query_embedding(self, query_text: str, embedder) -> np.ndarray:
        cached = self._query_embedding_cache.get(query_text)
        if cached is not None:
            return cached
        embedding = np.asarray(
            embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0],
            dtype=float,
        )
        if len(self._query_embedding_cache) > 512:
            self._query_embedding_cache.clear()
        self._query_embedding_cache[query_text] = embedding
        return embedding

    @staticmethod
    def _task_score_from_trajectory(trajectory: TrajectoryRecord) -> float:
        if trajectory.success:
            return 1.0
        return max(0.0, min(1.0, float(trajectory.average_reward)))

    @staticmethod
    def _normalize_fields(fields: List[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for raw in fields:
            key = str(raw).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            normalized.append(key)
        return normalized

    @staticmethod
    def _looks_like_legacy_structure_fields(fields: List[str]) -> bool:
        return any(field in _LEGACY_SKILL_STRUCTURE_FIELDS for field in fields)

    @staticmethod
    def _is_default_metadata_template(fields: List[str]) -> bool:
        return list(fields or []) == list(_DEFAULT_SKILL_METADATA_FIELDS)

    def _infer_metadata_fields_from_entries(self) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for entry in self.entries:
            for raw_key in dict(entry.memory_metadata or {}).keys():
                key = str(raw_key).strip()
                if not key or key.startswith("_") or key in seen:
                    continue
                if key in self.fixed_field_names:
                    continue
                seen.add(key)
                ordered.append(key)
        return ordered

    @staticmethod
    def _build_query_metadata_template(fields: List[str]) -> Dict[str, str]:
        return {str(field): "" for field in fields if str(field).strip()}

    @staticmethod
    def _normalize_memory_metadata(
        metadata: Dict[str, Any] | None,
        expected_fields: List[str],
    ) -> Dict[str, str]:
        raw = {
            str(key): str(value).strip()
            for key, value in dict(metadata or {}).items()
            if str(key).strip() and not str(key).startswith("_")
        }
        normalized: Dict[str, str] = {}
        for field in expected_fields or list(_DEFAULT_SKILL_METADATA_FIELDS):
            key = str(field).strip()
            if not key:
                continue
            normalized[key] = raw.get(key, "") or "N/A"
        for key, value in raw.items():
            if key not in normalized and value:
                normalized[key] = value
        return normalized

    def _extract_fixed_memory_metadata(
        self,
        metadata: Dict[str, Any] | None,
        allowed_field: str = "",
    ) -> Dict[str, str]:
        extracted: Dict[str, str] = {}
        allowed = str(allowed_field or "").strip()
        for raw_key, raw_value in dict(metadata or {}).items():
            key = str(raw_key).strip()
            value = str(raw_value).strip()
            if not key or key.startswith("_") or key not in self.fixed_field_names:
                continue
            if allowed and key != allowed:
                continue
            extracted[key] = value or "N/A"
        return extracted

    def _snapshot_path(self) -> Path | None:
        if self.storage_dir is None:
            return None
        return Path(self.storage_dir) / self._snapshot_filename

    def _updates_path(self) -> Path | None:
        if self.storage_dir is None:
            return None
        return Path(self.storage_dir) / self._updates_filename

    @staticmethod
    def _dedupe_ids(ids: List[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for raw in ids:
            item = str(raw).strip()
            if not item or item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    def _apply_incremental_updates(self) -> None:
        updates_path = self._updates_path()
        if updates_path is None or not updates_path.exists():
            return
        index = {entry.entry_id: entry for entry in self.entries}
        with open(updates_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for item in payload.get("entries", []):
                    try:
                        entry = self._coerce_entry(item)
                    except Exception:
                        continue
                    index[entry.entry_id] = entry
                for entry_id in payload.get("deleted_entry_ids", []):
                    key = str(entry_id).strip()
                    if key and key in index:
                        del index[key]
        self.entries = list(index.values())

    @staticmethod
    def _coerce_entry(payload: Dict[str, Any]) -> SkillMemoryEntry:
        item = dict(payload or {})
        # Extract from legacy metadata before stripping
        legacy_meta = dict(item.get("metadata", {}) or {})
        item.setdefault("task_description", str(legacy_meta.get("task_description", "")))
        item.setdefault("task_goal", str(legacy_meta.get("task_goal", "")))
        item.setdefault("subgoal", str(legacy_meta.get("subgoal", "")))
        item.setdefault("state_trace_summary", str(legacy_meta.get("state_trace_summary", "")))
        item.setdefault("recent_environment_feedback", str(legacy_meta.get("recent_environment_feedback", legacy_meta.get("environment_feedback", ""))))
        # Strip legacy fields
        for legacy_key in ("state_signature", "task_id", "participants", "step_index",
                           "workflow_phase", "participant_signature", "failure_mode",
                           "structure_version", "metadata", "expected_output"):
            item.pop(legacy_key, None)
        if not dict(item.get("memory_metadata", {}) or {}):
            item["memory_metadata"] = {"lesson": "N/A"}
        item.setdefault("stage_assignments", {})
        item.setdefault("stage_subgoals", {})
        item.setdefault("retrieval_count", 0)
        return SkillMemoryEntry(**item)
