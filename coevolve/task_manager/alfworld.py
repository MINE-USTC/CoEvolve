from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Monkey-patch: force cleanup of TemporaryDirectory (parallel thread race condition)
import shutil as _shutil
_orig_td_cleanup = tempfile.TemporaryDirectory.cleanup
def _safe_cleanup(self):
    try:
        _shutil.rmtree(self.name, ignore_errors=True)
    except Exception:
        pass
tempfile.TemporaryDirectory.cleanup = _safe_cleanup

# Global lock: alfworld env operations (init, start_task, step_task) are NOT thread-safe.
# Each AlfworldTaskManager instance has its own env, but the underlying TextWorld/PDDL
# solver uses shared temp directories and file I/O that conflict across threads.
_ALFWORLD_GLOBAL_LOCK = threading.Lock()

from ..schema import TaskRecord, TrajectoryRecord
from .base import BaseTaskManager

try:
    import alfworld
    import alfworld.agents.environment
except ImportError:
    alfworld = None


PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


_ACTION_PATTERNS = [
    re.compile(r"^(take\s+.+?\s+from\s+.+?)$", re.IGNORECASE),
    re.compile(r"^(go to\s+.+?)$", re.IGNORECASE),
    re.compile(r"^(open\s+.+?)$", re.IGNORECASE),
    re.compile(r"^(move\s+.+?\s+to\s+.+?)$", re.IGNORECASE),
    re.compile(r"^(clean\s+.+?\s+with\s+.+?)$", re.IGNORECASE),
    re.compile(r"^(heat\s+.+?\s+with\s+.+?)$", re.IGNORECASE),
    re.compile(r"^(cool\s+.+?\s+with\s+.+?)$", re.IGNORECASE),
    re.compile(r"^(use\s+.+?)$", re.IGNORECASE),
    re.compile(r"^(close\s+.+?)$", re.IGNORECASE),
    re.compile(r"^(examine\s+.+?)$", re.IGNORECASE),
    re.compile(r"^(slice\s+.+?\s+with\s+.+?)$", re.IGNORECASE),
]


def _get_env_name_from_gamefile(gamefile: str) -> str | None:
    for key in PREFIXES:
        if key in gamefile:
            return key
    return None


def _split_task_text(goal_text: str) -> tuple[str, str]:
    marker = "Your task is to:"
    text = str(goal_text or "").strip()
    if marker not in text:
        return text, ""
    prefix, suffix = text.split(marker, 1)
    goal = suffix.strip()
    return prefix.strip(), f"{marker} {goal}" if goal else marker


class AlfworldTaskManager(BaseTaskManager):
    def __init__(
        self,
        tasks_path: str,
        env_config_path: str,
        max_trials: int = 50,
        few_shots_path: str = "",
        split_override: str = "",
    ):
        super().__init__(tasks_path=tasks_path)
        self.env_config_path = env_config_path
        self.max_trials = max_trials
        self.project_root = Path(env_config_path).resolve().parent.parent
        self.env_config = self._load_env_config(env_config_path)
        if split_override:
            self.env_config["split"] = split_override
        self.few_shots_path = few_shots_path
        self.few_shots = self._load_few_shots(few_shots_path)
        self.default_few_shot_key = self._resolve_default_few_shot_key(self.few_shots)
        self.main_env = None
        self._env_lock = threading.Lock()
        if alfworld is not None:
            env_type = self.env_config["env"]["type"]
            with _ALFWORLD_GLOBAL_LOCK:
                with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                    self.main_env = alfworld.agents.environment.get_environment(env_type)(
                        self.env_config,
                        train_eval=self.env_config["split"],
                    )

    def load_tasks(self) -> List[TaskRecord]:
        path = Path(self.tasks_path)
        rows = json.loads(path.read_text(encoding="utf-8"))
        tasks: List[TaskRecord] = []
        for idx, row in enumerate(rows, start=1):
            env_name = _get_env_name_from_gamefile(row["gamefile"])
            gamefile = row["gamefile"]
            if not Path(gamefile).is_absolute():
                gamefile = str((self.project_root / gamefile).resolve())
            task_type = PREFIXES.get(env_name, "")
            task_description, task_goal = _split_task_text(str(row.get("goal", "")))
            selected_shots, shot_key = self._select_few_shots(
                task_type=task_type,
                goal=task_goal,
            )
            tasks.append(
                TaskRecord(
                    task_id=str(idx),
                    description=task_description,
                    task_goal=task_goal,
                    metadata={
                        "env_kwargs": {"gamefile": gamefile, "config": "alfworld"},
                        "env_name": env_name,
                        "task_type": task_type,
                        "few_shots": selected_shots,
                        "few_shot_key": shot_key,
                        "raw_goal": str(row.get("goal", "")),
                    },
                )
            )
        return tasks

    def sample_tasks(self, tasks: List[TaskRecord], sample_size: int, seed: int) -> List[TaskRecord]:
        from random import Random

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
        if self.main_env is None:
            raise RuntimeError("alfworld is not installed, so AlfworldTaskManager cannot start tasks.")

        gamefile = task.metadata["env_kwargs"]["gamefile"]
        env_name = task.metadata["env_name"]
        try:
            with _ALFWORLD_GLOBAL_LOCK:
                self.main_env.game_files = [gamefile]
                with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                    env = self.main_env.init_env(batch_size=1)
                    observation, info = env.reset()
        except Exception as e:
            print(f"[WARN] Failed to load task {task.task_id} ({gamefile}): {type(e).__name__}: {e}", file=sys.stderr)
            # Return a dummy episode that immediately finishes as failed
            return {
                "env": None,
                "task": task,
                "env_name": env_name,
                "state_summary": f"Task failed to load: {e}",
                "last_observation": f"Task failed to load: {e}",
                "done": True,
                "step_count": 0,
            }
        initial_observation = self._process_observation(observation[0])
        return {
            "env": env,
            "task": task,
            "env_name": env_name,
            "state_summary": initial_observation,
            "last_observation": initial_observation,
            "done": False,
            "step_count": 0,
        }

    def step_task(self, episode: Dict[str, Any], action: str) -> Dict[str, Any]:
        env = episode["env"]
        if env is None:
            return {
                "state_summary": "Task not loaded",
                "last_observation": "Task not loaded",
                "reward": 0.0,
                "done": True,
                "success": False,
                "info": {"step_count": 0, "executed_action": ""},
            }
        clean_action = self._process_action(action)
        with _ALFWORLD_GLOBAL_LOCK:
            observation, reward, done, info = env.step([clean_action])
        processed_observation = self._process_observation(observation[0])
        won = info.get("won", [False])[0]
        processed_reward = 0.0 if not won else 1.0
        if "think:" in clean_action or processed_observation == "Nothing happens.":
            processed_reward = -1.0
        next_step = int(episode.get("step_count", 0)) + 1
        done_flag = bool(done[0])
        if done_flag and processed_reward != 1.0:
            done_flag = False
        return {
            "state_summary": processed_observation,
            "last_observation": processed_observation,
            "reward": processed_reward,
            "done": done_flag,
            "success": bool(won),
            "info": {"step_count": next_step, "executed_action": clean_action},
        }

    def evaluate_trajectory(self, trajectory: TrajectoryRecord) -> Dict[str, Any]:
        return {
            "success": trajectory.success,
            "average_reward": trajectory.average_reward,
            "step_count": len(trajectory.steps),
        }

    def compute_average_reward(self, steps: list) -> float:
        if not steps:
            return 0.0
        for step in reversed(steps):
            success = bool(getattr(step, "success", False) if not isinstance(step, dict) else step.get("success", False))
            if success:
                return 1.0
        return 0.0

    def get_metric_label(self) -> str:
        return "final_result"

    @staticmethod
    def _load_env_config(env_config_path: str) -> Dict[str, Any]:
        config_path = Path(env_config_path).resolve()
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        base_dir = config_path.parent
        alt_base_dir = config_path.parent.parent

        def resolve_path(path_value: str) -> str:
            path = Path(path_value)
            if path.is_absolute():
                return str(path)
            primary = (base_dir / path).resolve()
            if primary.exists():
                return str(primary)
            secondary = (alt_base_dir / path).resolve()
            if secondary.exists():
                return str(secondary)
            if len(path.parts) > 1 and path.parts[0] == "data" and path.parts[1] == "alfworld":
                trimmed = Path(*([path.parts[0]] + list(path.parts[2:])))
                tertiary = (alt_base_dir / trimmed).resolve()
                if tertiary.exists():
                    return str(tertiary)
            return str(primary)

        if "task_file" in config:
            config["task_file"] = resolve_path(config["task_file"])
        if "dataset" in config:
            for key in ["data_path", "eval_id_data_path", "eval_ood_data_path"]:
                value = config["dataset"].get(key)
                if value:
                    config["dataset"][key] = resolve_path(value)
        if "logic" in config:
            for key in ["domain", "grammar"]:
                value = config["logic"].get(key)
                if value:
                    config["logic"][key] = resolve_path(value)
        if "mask_rcnn" in config and config["mask_rcnn"].get("pretrained_model_path"):
            config["mask_rcnn"]["pretrained_model_path"] = resolve_path(config["mask_rcnn"]["pretrained_model_path"])
        if "general" in config and config["general"].get("save_path"):
            config["general"]["save_path"] = resolve_path(config["general"]["save_path"])
        return config

    @staticmethod
    def _process_action(action: str) -> str:
        clean = str(action or "").strip().replace("<", "").split("\n")[0]
        clean = clean.replace(">", "").replace("OK.", "").replace("OK", "").strip()
        clean = clean.strip('"').strip("'")
        clean = re.sub(r"\s+", " ", clean).strip()
        clean = re.sub(r"[。.!;,]+$", "", clean).strip()
        clean = re.sub(r"\s*\([^)]*\)\s*$", "", clean).strip()
        lowered = clean.lower()
        for pattern in _ACTION_PATTERNS:
            match = pattern.match(lowered)
            if match:
                return match.group(1).strip()
        return lowered

    @staticmethod
    def _process_observation(observation: str) -> str:
        if observation.startswith("You arrive at loc "):
            observation = observation[observation.find(". ") + 2 :]
        return observation

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
        few_shots: Dict[str, str]
        if isinstance(payload, dict):
            few_shots = {str(key): str(value) for key, value in payload.items() if str(key).strip() and str(value).strip()}
        elif isinstance(payload, list):
            few_shots = {f"few_shot_{idx}": str(value) for idx, value in enumerate(payload, start=1) if str(value).strip()}
        else:
            return {}
        return AlfworldTaskManager._ensure_react_few_shots(few_shots)

    @staticmethod
    def _resolve_default_few_shot_key(few_shots: Dict[str, str]) -> str:
        if "react_puttwo_0" in few_shots:
            return "react_puttwo_0"
        if "act_puttwo_0" in few_shots:
            return "act_puttwo_0"
        for key in few_shots:
            if key.startswith("react_"):
                return key
        if not few_shots:
            return ""
        return next(iter(few_shots.keys()))

    def _select_few_shots(self, task_type: str, goal: str) -> tuple[List[str], str]:
        if not self.few_shots:
            return [], ""
        inferred_type = self._infer_task_type_from_goal(goal)
        resolved_type = task_type or inferred_type
        candidate_keys: List[str] = []
        if resolved_type:
            candidate_keys.extend([f"react_{resolved_type}_0", f"act_{resolved_type}_0"])
        if inferred_type and inferred_type != resolved_type:
            candidate_keys.extend([f"react_{inferred_type}_0", f"act_{inferred_type}_0"])
        candidate_keys.append(self.default_few_shot_key)
        for key in candidate_keys:
            token = str(key or "").strip()
            if token and token in self.few_shots:
                return [self.few_shots[token]], token
        return [next(iter(self.few_shots.values()))], next(iter(self.few_shots.keys()))

    @staticmethod
    def _infer_task_type_from_goal(goal: str) -> str:
        lowered = goal.lower()
        if "two" in lowered and "put" in lowered:
            return "puttwo"
        if "clean" in lowered:
            return "clean"
        if "heat" in lowered:
            return "heat"
        if "cool" in lowered:
            return "cool"
        if any(token in lowered for token in ["look at", "examine"]):
            return "examine"
        if "put" in lowered:
            return "put"
        return ""

    @staticmethod
    def _ensure_react_few_shots(few_shots: Dict[str, str]) -> Dict[str, str]:
        enriched = dict(few_shots)
        for key, value in list(few_shots.items()):
            if not key.startswith("act_"):
                continue
            suffix = key[len("act_") :]
            react_key = f"react_{suffix}"
            if react_key in enriched:
                continue
            enriched[react_key] = AlfworldTaskManager._act_to_react_trajectory(value)
        return enriched

    @staticmethod
    def _act_to_react_trajectory(text: str) -> str:
        lines = str(text or "").split("\n")
        output: List[str] = []
        for raw in lines:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped.startswith(">"):
                output.append(line)
                continue
            action = stripped[1:].strip()
            if not action:
                output.append(line)
                continue
            if action.lower().startswith("think:"):
                output.append(f"> {action}")
                continue
            thought = AlfworldTaskManager._thought_for_action(action)
            output.append(f"> think: {thought}")
            output.append("OK.")
            output.append(f"> {action}")
        return "\n".join(output)

    @staticmethod
    def _thought_for_action(action: str) -> str:
        lowered = action.lower()
        if lowered.startswith("go to "):
            target = action[6:].strip()
            return f"I should move to {target} to inspect it and unlock the next interaction."
        if lowered.startswith("open "):
            target = action[5:].strip()
            return f"{target} may be closed and hide useful objects, so I should open it."
        if lowered.startswith("take "):
            return "I need to pick up the required item so I can complete the placement or processing goal."
        if lowered.startswith("move "):
            return "Now I should place the carried object at the target receptacle to satisfy the goal."
        if lowered.startswith("clean "):
            return "The goal requires a clean object, so I should clean it first."
        if lowered.startswith("heat "):
            return "The goal requires a heated object, so I should heat it now."
        if lowered.startswith("cool "):
            return "The goal requires a cooled object, so I should cool it now."
        if lowered.startswith("use "):
            return "Using this object should trigger the required environment state change."
        if lowered.startswith("examine "):
            return "I should examine this target to gather decisive information."
        if lowered.startswith("close "):
            return "Closing this receptacle can reduce clutter and keep state consistent."
        if lowered.startswith("slice "):
            return "I should slice the object to match the required processed state."
        return "I should take the next concrete action that best advances the current subgoal."

    # ── Task-specific environment metadata ──

    def get_action_format_prompt(self) -> str:
        return (
            "ALFWorld environment action format:\n"
            "Actions must follow one of these forms (a and b are single object/location identifiers):\n"
            "1. take a from b    - Pick up object a from location/container b.\n"
            "2. go to a          - Navigate to location a. This also reveals what is at/in a.\n"
            "3. open a           - Open a closed container (cabinet, fridge, drawer, microwave, safe).\n"
            "4. move a to b      - Place object a onto/into location b.\n"
            "5. clean a with b   - Clean object a using b (e.g. sinkbasin).\n"
            "6. heat a with b    - Heat object a using b (e.g. microwave).\n"
            "7. cool a with b    - Cool object a using b (e.g. fridge).\n"
            "8. use a            - Activate/toggle a device (e.g. desklamp). You must be AT the device and HOLDING the target object.\n"
            "9. close a          - Close an open container.\n"
            "10. examine a       - Examine a small held object.\n"
            "11. slice a with b  - Slice object a with knife b.\n"
            "Return exactly one action string with no trailing period."
        )

    def get_trivial_feedbacks(self) -> set:
        return {"nothing happens.", "no environment action in this step.",
                "no environment action in this stage.", ""}

    def get_failure_reward_threshold(self) -> float:
        return 0.0  # reward=-1 means failure in ALFWorld

    def get_success_reward_threshold(self) -> float:
        return 0.5
