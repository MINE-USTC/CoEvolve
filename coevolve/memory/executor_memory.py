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

from ..schema import ExecutorMemoryEntry, TrajectoryRecord
from ..storage.io import load_json
from .base import BaseMemoryStore
from .embedding import get_embedder


class ExecutorMemoryStore(BaseMemoryStore):
    def __init__(
        self,
        storage_path: str,
        agent_name: str,
        initial_score: float = 0.5,
        score_lambda: float = 0.2,
        score_omega: float = 0.05,
        min_score_to_keep: float = 0.05,
    ) -> None:
        super().__init__(storage_path)
        self.agent_name = agent_name
        self.entries: List[ExecutorMemoryEntry] = []
        self.failure_entries: List[ExecutorMemoryEntry] = []
        self.initial_score = max(0.0, float(initial_score))
        self.score_lambda = max(0.0, min(1.0, float(score_lambda)))
        self.score_omega = max(0.0, float(score_omega))
        self.min_score_to_keep = max(0.0, float(min_score_to_keep))
        self.total_retrieval_count = 0
        self.failure_retrieval_count = 0
        self._lock = threading.Lock()
        self._metadata_field_template: List[str] = ["lesson"]
        self._failure_metadata_field_template: List[str] = ["lesson", "error_analysis"]
        self._hybrid_weights: Dict[str, float] = {"text": 0.35, "semantic": 0.65}
        self._index_dirty = True
        self._failure_index_dirty = True
        self._entry_text_cache: Dict[str, str] = {}
        self._entry_embedding_cache: Dict[str, np.ndarray] = {}
        self._query_embedding_cache: Dict[str, np.ndarray] = {}
        self._index_entry_ids: List[str] = []
        self._entry_position: Dict[str, int] = {}
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._embedding_matrix: np.ndarray | None = None
        self._failure_text_cache: Dict[str, str] = {}
        self._failure_embedding_cache: Dict[str, np.ndarray] = {}
        self._failure_query_embedding_cache: Dict[str, np.ndarray] = {}
        self._failure_index_entry_ids: List[str] = []
        self._failure_entry_position: Dict[str, int] = {}
        self._failure_tfidf_vectorizer: TfidfVectorizer | None = None
        self._failure_tfidf_matrix = None
        self._failure_embedding_matrix: np.ndarray | None = None

    def retrieve(
        self,
        task_description: str,
        stage_subgoal: str,
        subgoal: str,
        state_trace_summary: str,
        recent_environment_feedback: str,
        memory_metadata: Dict[str, Any] | None,
        skill_name: str,
        assigned_parts: List[str] | None,
        top_k: int,
    ) -> List[Dict[str, object]]:
        with self._lock:
            snapshot = list(self.entries)
        normalized_parts = [
            str(item).strip()
            for item in (assigned_parts or [])
            if str(item).strip()
        ]
        filtered_snapshot = self._prefilter_entries(
            entries=snapshot,
            skill_name=skill_name,
            assigned_parts=normalized_parts,
        )
        ranked = self._rank_entries(
            task_description=task_description,
            stage_subgoal=stage_subgoal,
            subgoal=subgoal,
            state_trace_summary=state_trace_summary,
            recent_environment_feedback=recent_environment_feedback,
            memory_metadata=memory_metadata or self._build_query_metadata_template(self.metadata_fields),
            entries=filtered_snapshot,
            top_k=top_k,
        )
        return [payload for payload, _ in ranked[:top_k]]

    def retrieve_failure(
        self,
        task_description: str,
        stage_subgoal: str,
        subgoal: str,
        state_trace_summary: str,
        recent_environment_feedback: str,
        memory_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, object] | None:
        """Retrieve top-1 failure memory by similarity + exploration only (no weight bias)."""
        with self._lock:
            snapshot = list(self.failure_entries)
        if not snapshot:
            return None
        self._ensure_failure_index()
        embedder = get_embedder()
        query_text = self._retrieval_text_from_fields(
            task_description=task_description,
            stage_subgoal=stage_subgoal,
            subgoal=subgoal,
            state_trace_summary=state_trace_summary,
            recent_environment_feedback=recent_environment_feedback,
            memory_metadata=memory_metadata or self._build_query_metadata_template(self.failure_metadata_fields),
        )
        row_indexes = [
            self._failure_entry_position[entry.entry_id]
            for entry in snapshot
            if entry.entry_id in self._failure_entry_position
        ]
        if not row_indexes:
            return None
        query_emb = self._encode_query_embedding(query_text, embedder)
        semantic_scores = cosine_similarity([query_emb], self._failure_embedding_matrix[row_indexes])[0]
        tfidf_scores = np.zeros(len(row_indexes), dtype=float)
        if self._failure_tfidf_vectorizer is not None and self._failure_tfidf_matrix is not None:
            query_vec = self._failure_tfidf_vectorizer.transform([query_text])
            tfidf_scores = cosine_similarity(query_vec, self._failure_tfidf_matrix[row_indexes]).flatten()
        similarities = (
            self._hybrid_weights["text"] * tfidf_scores
            + self._hybrid_weights["semantic"] * semantic_scores
        ).tolist()
        best_idx = 0
        best_score = -1.0
        for i, entry in enumerate(snapshot):
            sim = float(similarities[i])
            bonus = self.score_omega * math.sqrt(
                max(0.0, math.log(max(1, self.failure_retrieval_count + 1)) / max(1, entry.retrieval_count + 1))
            )
            score = sim + bonus
            if score > best_score:
                best_score = score
                best_idx = i
        self.failure_retrieval_count += 1
        snapshot[best_idx].retrieval_count += 1
        return asdict(snapshot[best_idx])

    def add_failure_entry(self, entry: ExecutorMemoryEntry) -> None:
        """Add a failure memory entry (no weight update, no pruning)."""
        with self._lock:
            self.failure_entries.append(entry)
        self._mark_failure_index_dirty()

    def update_from_trajectory(
        self,
        trajectory: TrajectoryRecord,
        selected_steps: List[int] | None = None,
        llm_metadata: Dict[int, Dict[str, Any]] | None = None,
    ) -> None:
        """Create entries for selected steps where this agent participated.

        Args:
            trajectory: completed trajectory
            selected_steps: step indices to create entries for (if None, use all where agent participated)
            llm_metadata: {step_index: {field: value}} from LLM
        """
        # Update scores for previously retrieved entries
        touched_entry_ids: Dict[str, int] = {}
        for step in trajectory.steps:
            for hit in step.executor_memory_hits.get(self.agent_name, []):
                if hit.get("entry_id"):
                    entry_id = str(hit["entry_id"])
                    touched_entry_ids[entry_id] = touched_entry_ids.get(entry_id, 0) + 1

        metadata_map = llm_metadata or {}
        if selected_steps is not None:
            steps_to_add = selected_steps
        else:
            steps_to_add = [s.step_index for s in trajectory.steps if self.agent_name in s.executor_outputs]

        step_index_map = {s.step_index: s for s in trajectory.steps}
        expected_fields = self.metadata_fields
        with self._lock:
            for step_idx in steps_to_add:
                step = step_index_map.get(step_idx)
                if step is None:
                    continue
                payload = dict(step.executor_outputs.get(self.agent_name, {}))
                if not payload:
                    continue
                meta = self._normalize_memory_metadata(
                    metadata=metadata_map.get(step_idx, {}),
                    expected_fields=expected_fields,
                )
                self.entries.append(
                    ExecutorMemoryEntry(
                        entry_id=str(uuid4()),
                        agent_name=self.agent_name,
                        task_description=trajectory.task.description,
                        subgoal=step.decision.subgoal,
                        stage_subgoal=str(step.decision.stage_subgoals.get(str(payload.get("stage_name", "")), "")).strip(),
                        state_trace_summary=str(step.metadata.get("state_trace_summary", "")),
                        recent_environment_feedback=str(step.metadata.get("environment_feedback", "")),
                        memory_metadata=meta,
                        action_summary=str(payload.get("proposed_action") or payload.get("output") or ""),
                        assigned_part=self._compose_assigned_part(
                            stage_name=str(payload.get("stage_name", "")),
                            stage_role=str(payload.get("stage_role", "")),
                        ),
                        workflow_name=step.decision.skill_name,
                        success=trajectory.success,
                        reward=float(step.reward),
                        weight=self.initial_score,
                    )
                )

            self._update_retrieved_scores(trajectory, touched_entry_ids)
            self.prune_low_score(self.min_score_to_keep)
            self._mark_index_dirty()

    def snapshot(self) -> Dict[str, object]:
        return {
            "agent_name": self.agent_name,
            "entry_count": len(self.entries),
            "initial_score": self.initial_score,
            "score_lambda": self.score_lambda,
            "score_omega": self.score_omega,
            "min_score_to_keep": self.min_score_to_keep,
            "total_retrieval_count": self.total_retrieval_count,
            "metadata_fields": self.metadata_fields,
            "failure_metadata_fields": self.failure_metadata_fields,
            "entries": [asdict(item) for item in self.entries],
            "failure_entries": [asdict(item) for item in self.failure_entries],
            "failure_retrieval_count": self.failure_retrieval_count,
        }

    def content_snapshot(self) -> Dict[str, object]:
        return {
            "agent_name": self.agent_name,
            "entry_count": len(self.entries),
            "initial_score": self.initial_score,
            "score_lambda": self.score_lambda,
            "score_omega": self.score_omega,
            "min_score_to_keep": self.min_score_to_keep,
            "total_retrieval_count": self.total_retrieval_count,
            "entries": [asdict(item) for item in self.entries],
            "failure_entries": [asdict(item) for item in self.failure_entries],
            "failure_retrieval_count": self.failure_retrieval_count,
        }

    def structure_snapshot(self) -> Dict[str, object]:
        return {
            "agent_name": self.agent_name,
            "metadata_fields": self.metadata_fields,
            "failure_metadata_fields": self.failure_metadata_fields,
        }

    def load_snapshot(self, payload: Dict[str, object] | None) -> None:
        self.entries = []
        self.failure_entries = []
        if not payload:
            self._mark_index_dirty()
            self._mark_failure_index_dirty()
            return
        self.initial_score = max(0.0, float(payload.get("initial_score", self.initial_score)))
        self.score_lambda = max(0.0, min(1.0, float(payload.get("score_lambda", self.score_lambda))))
        self.score_omega = max(0.0, float(payload.get("score_omega", self.score_omega)))
        self.min_score_to_keep = max(0.0, float(payload.get("min_score_to_keep", self.min_score_to_keep)))
        self.total_retrieval_count = int(payload.get("total_retrieval_count", self.total_retrieval_count))
        for item in payload.get("entries", []):
            entry = self._coerce_entry(item)
            self.entries.append(entry)
        for item in payload.get("failure_entries", []):
            entry = self._coerce_entry(item)
            self.failure_entries.append(entry)
        self.failure_retrieval_count = int(payload.get("failure_retrieval_count", 0))
        configured_fields = [
            str(field).strip()
            for field in list(payload.get("metadata_fields", []) or [])
            if str(field).strip()
        ]
        configured_failure_fields = [
            str(field).strip()
            for field in list(payload.get("failure_metadata_fields", []) or [])
            if str(field).strip()
        ]
        if configured_fields:
            self._metadata_field_template = configured_fields
        else:
            inferred = self._infer_metadata_fields(self.entries)
            if inferred:
                self._metadata_field_template = inferred
        if configured_failure_fields:
            self._failure_metadata_field_template = configured_failure_fields
        else:
            inferred_failure = self._infer_metadata_fields(self.failure_entries)
            if inferred_failure:
                self._failure_metadata_field_template = inferred_failure
        self._mark_index_dirty()
        self._mark_failure_index_dirty()

    def clear(self) -> None:
        self.entries = []
        self.failure_entries = []
        self._mark_index_dirty()
        self._mark_failure_index_dirty()

    def clear_entries_keep_structure(self) -> None:
        """Clear all entries but preserve the evolved metadata field template."""
        self.entries = []
        self.failure_entries = []
        self.total_retrieval_count = 0
        self.failure_retrieval_count = 0
        self._mark_index_dirty()
        self._mark_failure_index_dirty()

    def load_from_storage(self) -> bool:
        snapshot_path = self._snapshot_path()
        if snapshot_path is None:
            return False
        payload = load_json(snapshot_path, default=None)
        if not payload:
            self.load_snapshot(None)
            return False
        self.load_snapshot(payload)
        return True

    @property
    def metadata_fields(self) -> List[str]:
        """Return the current metadata field structure."""
        inferred = self._infer_metadata_fields(self.entries)
        if self._is_default_metadata_template(self._metadata_field_template) and inferred:
            return inferred
        if self._metadata_field_template:
            return list(self._metadata_field_template)
        if inferred:
            return inferred
        return list(self._metadata_field_template)

    @property
    def failure_metadata_fields(self) -> List[str]:
        inferred = self._infer_metadata_fields(self.failure_entries)
        if self._is_default_failure_metadata_template(self._failure_metadata_field_template) and inferred:
            return inferred
        if self._failure_metadata_field_template:
            return list(self._failure_metadata_field_template)
        if inferred:
            return inferred
        return ["lesson", "error_analysis"]

    def persist_full_rewrite(self, reason: str = "") -> None:
        snapshot_path = self._snapshot_path()
        if snapshot_path is None:
            return
        payload = self.snapshot()
        payload["rewrite_reason"] = reason
        self.persist_snapshot(f"executor_memory_{self.agent_name}.json", payload=payload)

    def apply_structure_evolution(
        self,
        new_fields: List[str],
        field_mapping: Dict[str, str] | None = None,
        reason: str = "",
        source_case_ids: List[str] | None = None,
    ) -> int:
        """Update the metadata-field template used for future executor memories."""
        normalized_fields = [
            str(field).strip()
            for field in list(new_fields or [])
            if str(field).strip()
        ]
        if not normalized_fields:
            return 0
        changed = int(normalized_fields != self.metadata_fields)
        self._metadata_field_template = normalized_fields
        return changed

    def set_failure_structure_fields(self, new_fields: List[str]) -> int:
        normalized_fields = [
            str(field).strip()
            for field in list(new_fields or [])
            if str(field).strip()
        ]
        if not normalized_fields:
            return 0
        changed = int(normalized_fields != self.failure_metadata_fields)
        self._failure_metadata_field_template = normalized_fields
        return changed

    def prune_low_score(self, threshold: float | None = None) -> int:
        floor = self.min_score_to_keep if threshold is None else max(0.0, float(threshold))
        before = len(self.entries)
        self.entries = [entry for entry in self.entries if float(entry.weight) >= floor]
        if len(self.entries) != before:
            self._mark_index_dirty()
        return max(0, before - len(self.entries))

    def _rank_entries(
        self,
        task_description: str,
        stage_subgoal: str,
        subgoal: str,
        state_trace_summary: str,
        recent_environment_feedback: str,
        memory_metadata: Dict[str, Any],
        entries: List[ExecutorMemoryEntry],
        top_k: int,
    ) -> List[Tuple[Dict[str, object], float]]:
        if not entries:
            return []
        entries = list(entries)  # snapshot to avoid concurrent modification
        self._ensure_index()
        embedder = get_embedder()
        query_text = self._retrieval_text_from_fields(
            task_description=task_description,
            stage_subgoal=stage_subgoal,
            subgoal=subgoal,
            state_trace_summary=state_trace_summary,
            recent_environment_feedback=recent_environment_feedback,
            memory_metadata=memory_metadata,
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

    @staticmethod
    def _retrieval_text_from_fields(
        task_description: str,
        stage_subgoal: str,
        subgoal: str,
        state_trace_summary: str,
        recent_environment_feedback: str,
        memory_metadata: Dict[str, Any],
    ) -> str:
        parts = [
            f"task_description: {str(task_description).strip()}",
            f"stage_subgoal: {str(stage_subgoal).strip()}",
            f"subgoal: {str(subgoal).strip()}",
            f"state_trace_summary: {str(state_trace_summary).strip()}",
            f"recent_environment_feedback: {str(recent_environment_feedback).strip()}",
        ]
        return "\n".join([item for item in parts if item.split(':', 1)[1].strip()])

    def _retrieval_text_from_entry(self, entry: ExecutorMemoryEntry) -> str:
        return self._retrieval_text_from_fields(
            task_description=entry.task_description,
            stage_subgoal=entry.stage_subgoal,
            subgoal=entry.subgoal,
            state_trace_summary=entry.state_trace_summary,
            recent_environment_feedback=entry.recent_environment_feedback,
            memory_metadata=entry.memory_metadata,
        )

    @staticmethod
    def _compose_assigned_part(stage_name: str, stage_role: str) -> str:
        name = str(stage_name).strip()
        role = str(stage_role).strip()
        if name and role:
            return f"{name}:{role}"
        return name or role

    @staticmethod
    def _select_ranked_hits(
        candidate_pool: List[Dict[str, object]],
        skill_name: str,
        assigned_parts: List[str],
        top_k: int,
    ) -> List[Dict[str, object]]:
        if not candidate_pool:
            return []
        selected: List[Dict[str, object]] = []
        selected_ids: set[str] = set()

        def add_if_match(predicate) -> None:
            for payload in candidate_pool:
                entry_id = str(payload.get("entry_id", "")).strip()
                if not entry_id or entry_id in selected_ids or not predicate(payload):
                    continue
                selected.append(payload)
                selected_ids.add(entry_id)
                if len(selected) >= top_k:
                    return

        add_if_match(
            lambda payload: str(payload.get("workflow_name", "")).strip() == str(skill_name).strip()
            and (
                not assigned_parts
                or str(payload.get("assigned_part", "")).strip() in assigned_parts
            )
        )
        if len(selected) < top_k:
            add_if_match(lambda payload: str(payload.get("workflow_name", "")).strip() == str(skill_name).strip())
        if len(selected) < top_k:
            add_if_match(lambda payload: True)
        return selected[:top_k]

    @staticmethod
    def _prefilter_entries(
        entries: List[ExecutorMemoryEntry],
        skill_name: str,
        assigned_parts: List[str],
    ) -> List[ExecutorMemoryEntry]:
        normalized_skill = str(skill_name).strip()
        exact = [
            entry for entry in entries
            if str(entry.workflow_name).strip() == normalized_skill
            and (not assigned_parts or str(entry.assigned_part).strip() in assigned_parts)
        ]
        if exact:
            return exact
        same_skill = [entry for entry in entries if str(entry.workflow_name).strip() == normalized_skill]
        if same_skill:
            return same_skill
        return entries

    def _update_retrieved_scores(self, trajectory: TrajectoryRecord, retrieval_counts: Dict[str, int]) -> None:
        if not retrieval_counts:
            return
        task_score = self._task_score_from_trajectory(trajectory)
        entry_index = {entry.entry_id: entry for entry in self.entries}
        for entry_id, count in retrieval_counts.items():
            entry = entry_index.get(entry_id)
            if entry is None:
                continue
            for _ in range(count):
                self.total_retrieval_count += 1
                entry.retrieval_count += 1
                ema = (1.0 - self.score_lambda) * float(entry.weight) + self.score_lambda * task_score
                entry.weight = max(0.0, ema)
        self._mark_index_dirty()

    @staticmethod
    def _task_score_from_trajectory(trajectory: TrajectoryRecord) -> float:
        if trajectory.success:
            return 1.0
        return max(0.0, min(1.0, float(trajectory.average_reward)))

    def _mark_index_dirty(self) -> None:
        self._index_dirty = True

    def _mark_failure_index_dirty(self) -> None:
        self._failure_index_dirty = True

    def _ensure_index(self) -> None:
        if self._index_dirty:
            self._rebuild_index()

    def _ensure_failure_index(self) -> None:
        if self._failure_index_dirty:
            self._rebuild_failure_index()

    def _rebuild_index(self) -> None:
        previous_text_cache = dict(self._entry_text_cache)
        self._entry_text_cache = {}
        self._index_entry_ids = []
        self._entry_position = {}
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._embedding_matrix = None
        entries = list(self.entries)
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
            embeddings = get_embedder().encode(missing_texts, convert_to_numpy=True, normalize_embeddings=True)
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

    def _rebuild_failure_index(self) -> None:
        previous_text_cache = dict(self._failure_text_cache)
        self._failure_text_cache = {}
        self._failure_index_entry_ids = []
        self._failure_entry_position = {}
        self._failure_tfidf_vectorizer = None
        self._failure_tfidf_matrix = None
        self._failure_embedding_matrix = None
        entries = list(self.failure_entries)
        if not entries:
            self._failure_index_dirty = False
            return
        corpus: List[str] = []
        for entry in entries:
            text = self._retrieval_text_from_entry(entry)
            self._failure_text_cache[entry.entry_id] = text
            self._failure_index_entry_ids.append(entry.entry_id)
            corpus.append(text)
        self._failure_entry_position = {
            entry_id: index for index, entry_id in enumerate(self._failure_index_entry_ids)
        }
        self._failure_tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self._failure_tfidf_matrix = self._failure_tfidf_vectorizer.fit_transform(corpus)
        missing_ids: List[str] = []
        missing_texts: List[str] = []
        for entry_id, text in zip(self._failure_index_entry_ids, corpus):
            if entry_id not in self._failure_embedding_cache or previous_text_cache.get(entry_id) != text:
                missing_ids.append(entry_id)
                missing_texts.append(text)
        if missing_texts:
            embeddings = get_embedder().encode(missing_texts, convert_to_numpy=True, normalize_embeddings=True)
            for entry_id, embedding in zip(missing_ids, embeddings):
                self._failure_embedding_cache[entry_id] = np.asarray(embedding, dtype=float)
        active_ids = set(self._failure_index_entry_ids)
        self._failure_embedding_cache = {
            entry_id: embedding
            for entry_id, embedding in self._failure_embedding_cache.items()
            if entry_id in active_ids
        }
        self._failure_embedding_matrix = np.vstack([
            self._failure_embedding_cache[entry_id]
            for entry_id in self._failure_index_entry_ids
        ])
        self._failure_index_dirty = False

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

    def _find_anchor_step(self, trajectory: TrajectoryRecord, related_steps: List[int]):
        by_index = {step.step_index: step for step in trajectory.steps}
        for idx in related_steps:
            step = by_index.get(idx)
            if step is not None and self.agent_name in step.executor_outputs:
                return step
        for step in reversed(trajectory.steps):
            if self.agent_name in step.executor_outputs:
                return step
        if trajectory.steps:
            return trajectory.steps[-1]
        return None

    def _snapshot_path(self) -> Path | None:
        if self.storage_dir is None:
            return None
        return Path(self.storage_dir) / f"executor_memory_{self.agent_name}.json"

    @staticmethod
    def _build_query_metadata_template(fields: List[str]) -> Dict[str, str]:
        return {str(field): "" for field in fields if str(field).strip()}

    @staticmethod
    def _is_default_metadata_template(fields: List[str]) -> bool:
        return list(fields or []) == ["lesson"]

    @staticmethod
    def _is_default_failure_metadata_template(fields: List[str]) -> bool:
        return list(fields or []) == ["lesson", "error_analysis"]

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
        for field in expected_fields or ["lesson"]:
            key = str(field).strip()
            if not key:
                continue
            normalized[key] = raw.get(key, "") or "N/A"
        for key, value in raw.items():
            if key not in normalized and value:
                normalized[key] = value
        return normalized

    @staticmethod
    def _infer_metadata_fields(entries: List[ExecutorMemoryEntry]) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for entry in entries:
            for raw_key in dict(entry.memory_metadata or {}).keys():
                key = str(raw_key).strip()
                if not key or key.startswith("_") or key in seen:
                    continue
                seen.add(key)
                ordered.append(key)
        return ordered

    @staticmethod
    def _coerce_entry(payload: Dict[str, Any]) -> ExecutorMemoryEntry:
        item = dict(payload or {})
        # Strip legacy fields
        for legacy_key in ("state_signature", "task_id", "workflow_phase", "task_participants",
                           "failure_mode", "counterfactual_action", "counterfactual_owner",
                           "applicability_scope", "structure_version", "metadata",
                           "participant_signature", "expected_output"):
            item.pop(legacy_key, None)
        if not dict(item.get("memory_metadata", {}) or {}):
            item["memory_metadata"] = {"lesson": "N/A"}
        item.setdefault("retrieval_count", 0)
        return ExecutorMemoryEntry(**item)
