"""WebShop task manager — communicates with the WebShop HTTP server."""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from random import Random
from typing import Any, Dict, List
from uuid import uuid4

import requests

from ..schema import TaskRecord, TrajectoryRecord
from .base import BaseTaskManager


class WebShopTaskManager(BaseTaskManager):
    """Task manager for WebShop e-commerce benchmark via HTTP API."""

    def __init__(
        self,
        tasks_path: str,
        server_url: str = "http://localhost:5000",
        few_shots_path: str = "",
        max_trials: int = 50,
        mode: str = "train",
    ):
        super().__init__(tasks_path=tasks_path)
        self.server_url = server_url.rstrip("/")
        self.max_trials = max_trials
        self.mode = mode
        self.few_shots = self._load_few_shots(few_shots_path)
        self._session_counter = 0
        self._lock = threading.Lock()
        # Ensure localhost bypasses proxy (don't affect LLM API calls to external servers)
        import os
        no_proxy = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
        if "localhost" not in no_proxy:
            os.environ["NO_PROXY"] = f"{no_proxy},localhost,127.0.0.1" if no_proxy else "localhost,127.0.0.1"
        self._http = requests.Session()
        # Health check
        try:
            resp = self._http.get(f"{self.server_url}/health", timeout=5)
            data = resp.json()
            print(f"[WebShop] Server OK — {data.get('num_goals', '?')} goals loaded")
        except Exception as e:
            print(f"[WebShop] WARNING: server at {self.server_url} unreachable: {e}")

    # ── Core TaskManager methods ──

    def load_tasks(self) -> List[TaskRecord]:
        path = Path(self.tasks_path)
        rows = json.loads(path.read_text(encoding="utf-8"))
        tasks: List[TaskRecord] = []
        for idx, row in enumerate(rows, start=1):
            category = row.get("category", "")
            instruction = row.get("instruction_text", "")
            selected_shots, shot_key = self._select_few_shots(category, instruction)
            # Build the goal dict needed by /reset
            goal = dict(row)
            goal.setdefault("price_upper", 1000000)
            goal.setdefault("weight", 1)
            goal.setdefault("query", "")
            tasks.append(
                TaskRecord(
                    task_id=str(idx),
                    description=instruction,
                    task_goal=instruction,
                    metadata={
                        "goal": goal,
                        "category": category,
                        "asin": row.get("asin", ""),
                        "attributes": row.get("attributes", []),
                        "goal_options": row.get("goal_options", {}),
                        "few_shots": selected_shots,
                        "few_shot_key": shot_key,
                    },
                )
            )
        return tasks

    def sample_tasks(self, tasks: List[TaskRecord], sample_size: int, seed: int) -> List[TaskRecord]:
        if not tasks:
            return []
        size = min(max(1, sample_size), len(tasks))
        if size >= len(tasks):
            return list(tasks)
        rng = Random(seed)
        return rng.sample(tasks, size)

    def build_initial_state(self, task: TaskRecord) -> Dict[str, Any]:
        return {
            "task_id": task.task_id,
            "summary": task.full_text(),
        }

    def start_task(self, task: TaskRecord) -> Dict[str, Any]:
        session_id = self._make_session_id(task.task_id)
        goal = task.metadata.get("goal", {})
        try:
            resp = self._http.post(
                f"{self.server_url}/reset",
                json={"session_id": session_id, "goal": goal},
                timeout=30,
            )
            data = resp.json()
        except Exception as e:
            import sys
            print(f"[WebShop] Failed to start task {task.task_id}: {e}", file=sys.stderr)
            return {
                "env": None,
                "task": task,
                "session_id": session_id,
                "state_summary": f"Task failed to load: {e}",
                "last_observation": f"Task failed to load: {e}",
                "done": True,
                "step_count": 0,
            }
        observation = data.get("observation", "")
        available_actions = data.get("available_actions", {})
        prepared_observation = self.prepare_observation(
            observation=observation,
            available_actions=available_actions,
            task=task,
        )
        return {
            "env": "webshop",
            "task": task,
            "session_id": session_id,
            "state_summary": prepared_observation,
            "last_observation": prepared_observation,
            "raw_last_observation": observation,
            "available_actions": available_actions,
            "done": False,
            "step_count": 0,
        }

    def step_task(self, episode: Dict[str, Any], action: str) -> Dict[str, Any]:
        session_id = episode.get("session_id", "")
        if not session_id or episode.get("env") is None:
            return {
                "state_summary": "Task not loaded",
                "last_observation": "Task not loaded",
                "reward": 0.0,
                "done": True,
                "success": False,
                "info": {"step_count": 0, "executed_action": ""},
            }
        clean_action = self._process_action(action)
        if not clean_action:
            next_step = episode.get("step_count", 0) + 1
            invalid_observation = self._format_invalid_action_feedback(
                episode=episode,
                message=f"Invalid action format from model output: {action}. Use exactly search[keywords] or click[value].",
            )
            episode["step_count"] = next_step
            episode["last_observation"] = invalid_observation
            return {
                "state_summary": invalid_observation,
                "last_observation": invalid_observation,
                "reward": -1.0,
                "done": False,
                "success": False,
                "info": {
                    "step_count": next_step,
                    "executed_action": action,
                    "invalid_reason": "invalid_action_format",
                    "available_actions": episode.get("available_actions", {}),
                },
            }
        previous_available_actions = episode.get("available_actions", {})
        if not self._is_action_available(clean_action, previous_available_actions):
            next_step = episode.get("step_count", 0) + 1
            invalid_observation = self._format_invalid_action_feedback(
                episode=episode,
                message=f"Invalid WebShop action: {clean_action} is not available on the current page.",
            )
            episode["step_count"] = next_step
            episode["last_observation"] = invalid_observation
            return {
                "state_summary": invalid_observation,
                "last_observation": invalid_observation,
                "reward": -1.0,
                "done": False,
                "success": False,
                "info": {
                    "step_count": next_step,
                    "executed_action": clean_action,
                    "invalid_reason": "action_not_available",
                    "available_actions": previous_available_actions,
                },
            }
        try:
            resp = self._http.post(
                f"{self.server_url}/step",
                json={"session_id": session_id, "action": clean_action},
                timeout=30,
            )
            data = resp.json()
        except Exception as e:
            return {
                "state_summary": f"Server error: {e}",
                "last_observation": f"Server error: {e}",
                "reward": -1.0,
                "done": True,
                "success": False,
                "info": {"step_count": episode.get("step_count", 0) + 1, "executed_action": clean_action},
            }
        observation = data.get("observation", "")
        available_actions = data.get("available_actions", {})
        prepared_observation = self.prepare_observation(
            observation=observation,
            available_actions=available_actions,
            task=episode.get("task"),
        )
        env_reward = float(data.get("reward", 0.0))
        done = bool(data.get("done", False))
        next_step = episode.get("step_count", 0) + 1
        episode["step_count"] = next_step
        episode["last_observation"] = prepared_observation
        episode["raw_last_observation"] = observation
        episode["available_actions"] = available_actions

        # Reward shaping: align with ALFWorld's -1/0/1 scheme
        # -1: action was rejected by WebShop
        #  0: normal intermediate step
        #  env_reward: final step (done=True), continuous 0-1 from WebShop
        if done:
            processed_reward = env_reward
        elif self._is_invalid_action_observation(observation):
            processed_reward = -1.0
        else:
            processed_reward = 0.0

        if done:
            # Close session on the server
            try:
                self._http.post(
                    f"{self.server_url}/close",
                    json={"session_id": session_id},
                    timeout=10,
                )
            except Exception:
                pass
        return {
            "state_summary": prepared_observation,
            "last_observation": prepared_observation,
            "reward": processed_reward,
            "done": done,
            "success": done and env_reward >= self.get_success_reward_threshold(),
            "info": {
                "step_count": next_step,
                "executed_action": clean_action,
                "env_reward": env_reward,
                "raw_observation": observation,
                "available_actions": available_actions,
            },
        }

    def evaluate_trajectory(self, trajectory: TrajectoryRecord) -> Dict[str, Any]:
        return {
            "success": trajectory.success,
            "average_reward": trajectory.average_reward,
            "step_count": len(trajectory.steps),
        }

    def prepare_observation(
        self,
        observation: str,
        available_actions: Dict[str, Any] | None = None,
        task: TaskRecord | None = None,
    ) -> str:
        del task
        text = str(observation or "").strip()
        if not text:
            return ""
        if "[SEP]" not in text:
            return text

        parts = [part.strip() for part in text.split("[SEP]") if part.strip()]
        parts = self._strip_instruction_prefix(parts)
        if not parts:
            return ""

        page_type = self._infer_page_type(parts)
        page_info = next((part for part in parts if self._looks_like_page_marker(part)), "")
        result_candidates = self._extract_result_candidates(parts)
        option_groups = self._extract_option_groups(parts)
        if result_candidates:
            product_title = ""
            detail_snippets: List[str] = []
        else:
            product_title = self._extract_product_title(parts, option_groups)
            detail_snippets = self._extract_detail_snippets(parts, product_title)
        click_values = self._merge_click_values(
            self._extract_click_values_from_observation(parts, option_groups, result_candidates),
            self._extract_click_values_from_available_actions(available_actions),
        )

        lines = [
            "WebShop page summary:",
            f"- Page type: {page_type}",
        ]
        if page_info:
            lines.append(f"- Page info: {page_info}")
        if self._search_available(parts, available_actions):
            lines.append("- Search action available: search[short keywords]")
        if click_values:
            preview = ", ".join(f"click[{value}]" for value in click_values[:10])
            if len(click_values) > 10:
                preview += ", ..."
            lines.append(f"- Available clicks now: {preview}")
        if result_candidates:
            lines.append("- Top visible results:")
            for asin, title, price in result_candidates[:4]:
                suffix = f" ({price})" if price else ""
                lines.append(f"  - {asin}: {title}{suffix}")
        if product_title:
            lines.append(f"- Product in view: {product_title}")
        if option_groups:
            lines.append("- Selectable options:")
            for key, values in option_groups[:4]:
                preview = ", ".join(values[:8])
                if len(values) > 8:
                    preview += ", ..."
                lines.append(f"  - {key}: {preview}")
        if detail_snippets:
            lines.append("- Key page content:")
            for snippet in detail_snippets[:2]:
                lines.append(f"  - {snippet}")
        return "\n".join(lines)

    # ── Task-specific environment metadata ──

    def get_action_format_prompt(self) -> str:
        return (
            "WebShop environment action format:\n"
            "There are two action types:\n"
            "\n"
            "1. search[keywords]  - Search for products. Put your search query inside the brackets.\n"
            "   Example: search[wireless headphones]\n"
            "\n"
            "2. click[value]      - Click on a button or link. The value must EXACTLY match one of "
            "the available clickable items shown in the observation.\n"
            "   Examples: click[b07abc1234]  click[buy now]  click[black]  click[< prev]\n"
            "\n"
            "Use only search[...] or click[...]. Return exactly one action string."
        )

    def get_trivial_feedbacks(self) -> set:
        return {"no environment action in this step.", "no environment action in this stage.", ""}

    def get_failure_reward_threshold(self) -> float:
        # -1.0 means invalid/rejected WebShop action.
        # reward < 0 triggers failure memory, same as ALFWorld.
        return 0.0

    def get_success_reward_threshold(self) -> float:
        return 0.5

    def compute_average_reward(self, steps: list) -> float:
        """WebShop reward is meaningful at purchase time; do not dilute it by step count."""
        if not steps:
            return 0.0
        for step in reversed(steps):
            metadata = getattr(step, "metadata", {}) if not isinstance(step, dict) else step.get("metadata", {})
            workflow_trace = metadata.get("workflow_trace", {}) if isinstance(metadata, dict) else {}
            interactions = workflow_trace.get("env_interactions", []) if isinstance(workflow_trace, dict) else []
            for interaction in reversed(interactions):
                if isinstance(interaction, dict) and bool(interaction.get("done", False)):
                    return max(0.0, float(interaction.get("reward", 0.0)))
        return 0.0

    def use_reward_for_best_round(self) -> bool:
        return True

    def get_metric_label(self) -> str:
        return "final_score"

    # ── Internal helpers ──

    def _make_session_id(self, task_id: str) -> str:
        with self._lock:
            self._session_counter += 1
            return f"ws_{task_id}_{self._session_counter}_{threading.get_ident()}"

    @staticmethod
    def _process_action(raw: str) -> str:
        """Extract a valid WebShop action (search[...] or click[...]) from raw LLM output."""
        text = str(raw or "").strip()
        # Try to find search[...] pattern
        match = re.search(r'search\[([^\]]+)\]', text, re.IGNORECASE)
        if match:
            return f"search[{match.group(1).strip()}]"
        # Try to find click[...] pattern
        match = re.search(r'click\[([^\]]+)\]', text, re.IGNORECASE)
        if match:
            return f"click[{match.group(1).strip()}]"
        # Fallback: if the text itself looks like a bare action
        lowered = text.lower().strip()
        if lowered.startswith("search[") or lowered.startswith("click["):
            return text
        return ""

    @staticmethod
    def _format_invalid_action_feedback(episode: Dict[str, Any], message: str) -> str:
        current_state = str(
            episode.get("last_observation")
            or episode.get("state_summary")
            or episode.get("raw_last_observation")
            or ""
        ).strip()
        if not current_state:
            return message
        return (
            f"{current_state}\n\n"
            f"{message}\n"
            "Stay on the same page. Choose the next action only from the currently available actions above."
        )

    @staticmethod
    def _split_action(action: str) -> tuple[str, str]:
        match = re.match(r"^\s*(search|click)\[([^\]]+)\]\s*$", str(action or ""), flags=re.IGNORECASE)
        if not match:
            return "", ""
        return match.group(1).lower(), match.group(2).strip()

    def _is_action_available(self, action: str, available_actions: Any) -> bool:
        """Match the proposed action against WebShop's current action list.

        Option clicks can update server-side selected options without changing the
        rendered page text, so unchanged observations are not an invalid-action signal.
        """
        action_type, value = self._split_action(action)
        if not action_type:
            return False
        if not available_actions:
            return True
        if action_type == "search":
            return self._contains_true_flag(available_actions, "has_search_bar")
        if action_type == "click":
            if value.strip().lower() == "search":
                return False
            target = self._canonical_click_value(value).lower()
            click_values = self._extract_click_values_from_available_actions(available_actions)
            return any(self._canonical_click_value(candidate).lower() == target for candidate in click_values)
        return False

    @staticmethod
    def _is_invalid_action_observation(observation: str) -> bool:
        text = str(observation or "").strip().lower()
        invalid_markers = (
            "invalid action",
            "invalid click",
            "invalid search",
            "action is not available",
            "not available on the current page",
            "not a valid action",
            "unknown action",
            "use search[keywords] or click[value]",
        )
        return any(marker in text for marker in invalid_markers)

    @staticmethod
    def _clip_text(text: str, limit: int) -> str:
        clean = str(text or "").strip()
        if len(clean) <= limit:
            return clean
        return clean[: max(0, limit - 3)] + "..."

    @staticmethod
    def _strip_instruction_prefix(parts: List[str]) -> List[str]:
        cleaned = list(parts)
        if cleaned and cleaned[0].lower() == "webshop":
            cleaned = cleaned[1:]
        result: List[str] = []
        skip_next = False
        for part in cleaned:
            if skip_next:
                skip_next = False
                continue
            lowered = part.lower()
            if lowered in {"instruction", "instruction:"}:
                skip_next = True
                continue
            result.append(part)
        return result

    @staticmethod
    def _looks_like_page_marker(text: str) -> bool:
        return bool(re.match(r"^Page\s+\d+\s+\(Total results:\s*\d+\)$", str(text or "").strip(), flags=re.IGNORECASE))

    @staticmethod
    def _looks_like_price(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        return lowered.startswith("price:") or "$" in lowered

    @staticmethod
    def _looks_like_rating(text: str) -> bool:
        return str(text or "").strip().lower().startswith("rating:")

    @staticmethod
    def _looks_like_asin(text: str) -> bool:
        return bool(re.match(r"^[A-Z0-9]{10}$", str(text or "").strip().upper()))

    @staticmethod
    def _canonical_click_value(text: str) -> str:
        clean = str(text or "").strip()
        mapping = {
            "back to search": "back to search",
            "< prev": "< prev",
            "next >": "Next >",
            "buy now": "buy now",
            "features": "features",
            "description": "description",
            "reviews": "reviews",
        }
        lowered = clean.lower()
        if lowered in mapping:
            return mapping[lowered]
        if WebShopTaskManager._looks_like_asin(clean):
            return clean.upper()
        return clean

    @staticmethod
    def _looks_like_option_name(text: str) -> bool:
        clean = str(text or "").strip()
        lowered = clean.lower()
        known = {
            "size",
            "color",
            "colour",
            "style",
            "flavor",
            "flavour",
            "scent",
            "material",
            "pack size",
            "count",
            "quantity",
            "width",
            "length",
        }
        return lowered in known

    @classmethod
    def _is_structural_token(cls, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        return (
            lowered in {"search", "back to search", "< prev", "next >", "buy now", "features", "description", "reviews"}
            or cls._looks_like_page_marker(text)
            or cls._looks_like_price(text)
            or cls._looks_like_rating(text)
            or cls._looks_like_asin(text)
        )

    def _infer_page_type(self, parts: List[str]) -> str:
        lowered = [part.lower() for part in parts]
        if lowered == ["search"] or (len(lowered) == 1 and lowered[0] == "search"):
            return "search page"
        if any(self._looks_like_page_marker(part) for part in parts):
            return "search results page"
        if any(part.lower() == "buy now" for part in parts):
            return "product page"
        if any(part.lower() in {"features", "description", "reviews"} for part in parts):
            return "product detail page"
        if any(len(part) > 80 for part in parts):
            return "detail content page"
        return "page view"

    def _search_available(self, parts: List[str], available_actions: Dict[str, Any] | None) -> bool:
        if any(part.lower() == "search" for part in parts):
            return True
        return self._contains_true_flag(available_actions, "has_search_bar")

    def _contains_true_flag(self, payload: Any, target_key: str) -> bool:
        if isinstance(payload, dict):
            for key, value in payload.items():
                if str(key).strip().lower() == target_key.lower() and bool(value):
                    return True
                if self._contains_true_flag(value, target_key):
                    return True
        elif isinstance(payload, (list, tuple, set)):
            return any(self._contains_true_flag(item, target_key) for item in payload)
        return False

    def _extract_result_candidates(self, parts: List[str]) -> List[tuple[str, str, str]]:
        results: List[tuple[str, str, str]] = []
        i = 0
        while i < len(parts):
            token = parts[i]
            if not self._looks_like_asin(token):
                i += 1
                continue
            asin = token.upper()
            title = ""
            price = ""
            if i + 1 < len(parts):
                title = self._clip_text(parts[i + 1], 120)
            if i + 2 < len(parts) and self._looks_like_price(parts[i + 2]):
                price = self._clip_text(parts[i + 2], 40)
            results.append((asin, title or "Unknown product", price))
            i += 3
        return results

    def _extract_option_groups(self, parts: List[str]) -> List[tuple[str, List[str]]]:
        groups: List[tuple[str, List[str]]] = []
        i = 0
        while i < len(parts):
            key = parts[i]
            if not self._looks_like_option_name(key) or self._is_structural_token(key):
                i += 1
                continue
            values: List[str] = []
            j = i + 1
            while j < len(parts):
                candidate = parts[j]
                if self._is_structural_token(candidate):
                    break
                if len(candidate) > 80:
                    break
                if self._looks_like_option_name(candidate) and values:
                    break
                values.append(self._clip_text(candidate, 40))
                j += 1
            if values:
                groups.append((self._clip_text(key, 24), values))
                i = j
                continue
            i += 1
        return groups

    def _extract_product_title(self, parts: List[str], option_groups: List[tuple[str, List[str]]]) -> str:
        option_keys = {key for key, _ in option_groups}
        option_values = {value for _, values in option_groups for value in values}
        for part in parts:
            if self._is_structural_token(part):
                continue
            if part in option_keys or part in option_values:
                continue
            if ":" in part and len(part.split(":", 1)[0].strip()) <= 24:
                continue
            if len(part) >= 28:
                return self._clip_text(part, 140)
        return ""

    def _extract_detail_snippets(self, parts: List[str], product_title: str) -> List[str]:
        snippets: List[str] = []
        for part in parts:
            if self._is_structural_token(part):
                continue
            if part == product_title:
                continue
            if len(part) < 40:
                continue
            snippets.append(self._clip_text(part, 160))
        return snippets

    def _extract_click_values_from_observation(
        self,
        parts: List[str],
        option_groups: List[tuple[str, List[str]]],
        result_candidates: List[tuple[str, str, str]],
    ) -> List[str]:
        values: List[str] = []
        for part in parts:
            lowered = part.lower()
            if lowered in {"back to search", "< prev", "next >", "buy now", "features", "description", "reviews"}:
                values.append(self._canonical_click_value(part))
        for asin, _, _ in result_candidates[:6]:
            values.append(self._canonical_click_value(asin))
        for _, option_values in option_groups[:4]:
            for value in option_values[:6]:
                values.append(self._canonical_click_value(value))
        return values

    def _extract_click_values_from_available_actions(self, available_actions: Any) -> List[str]:
        values: List[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if str(key).strip().lower() == "has_search_bar":
                        continue
                    walk(value)
                return
            if isinstance(node, (list, tuple, set)):
                for item in node:
                    walk(item)
                return
            if not isinstance(node, str):
                return
            clean = str(node).strip()
            if not clean or len(clean) > 80:
                return
            if self._looks_like_price(clean) or self._looks_like_rating(clean) or self._looks_like_page_marker(clean):
                return
            if clean.lower() in {"instruction", "instruction:", "webshop"}:
                return
            values.append(self._canonical_click_value(clean))

        walk(available_actions)
        return values

    @staticmethod
    def _merge_click_values(*groups: List[str]) -> List[str]:
        merged: List[str] = []
        seen: set[str] = set()
        for group in groups:
            for value in group:
                clean = str(value or "").strip()
                if not clean:
                    continue
                lowered = clean.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                merged.append(clean)
        return merged

    @staticmethod
    def _load_few_shots(few_shots_path: str) -> Dict[str, str]:
        if not few_shots_path:
            return {}
        path = Path(few_shots_path)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if isinstance(payload, dict):
            return {str(k): str(v) for k, v in payload.items() if str(k).strip() and str(v).strip()}
        return {}

    def _select_few_shots(self, category: str, goal: str) -> tuple[List[str], str]:
        if not self.few_shots:
            return [], ""
        cat = category.lower().strip()
        candidate_keys = [f"react_{cat}_0", f"act_{cat}_0"]
        for key in candidate_keys:
            if key in self.few_shots:
                return [self.few_shots[key]], key
        # Fallback: first available
        if self.few_shots:
            first_key = next(iter(self.few_shots))
            return [self.few_shots[first_key]], first_key
        return [], ""
