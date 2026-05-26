from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def default_memory_metadata() -> Dict[str, Any]:
    return {"lesson": ""}


@dataclass
class ModelSettings:
    name: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 1024
    top_p: float = 1.0


@dataclass
class AgentProfile:
    role: str
    persona: str
    collaboration_style: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMemoryConfig:
    storage_path: str = ""
    structure_name: str = "base"
    retrieval_top_k: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    name: str
    role: str
    profile: AgentProfile
    model: ModelSettings
    memory: AgentMemoryConfig


@dataclass
class TaskRecord:
    task_id: str
    description: str
    task_goal: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def full_text(self) -> str:
        description = str(self.description or "").strip()
        task_goal = str(self.task_goal or "").strip()
        if description and task_goal:
            return f"{description}\n\n{task_goal}"
        return description or task_goal


@dataclass
class SkillDecision:
    skill_name: str
    participants: List[str]
    subgoal: str
    assignment: Dict[str, List[str]] = field(default_factory=dict)
    stage_subgoals: Dict[str, str] = field(default_factory=dict)


@dataclass
class PromptStepContext:
    current_state_summary: str = ""
    recent_environment_feedback: str = ""
    state_trace_summary: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)
    special_notes: List[str] = field(default_factory=list)
    current_step_index: int = 0
    max_steps: int = 0
    remaining_steps: int = 0


@dataclass
class TrajectoryStep:
    step_index: int
    decision: SkillDecision
    executor_outputs: Dict[str, Any]
    reward: float
    success: bool
    executor_memory_hits: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryRecord:
    trajectory_id: str
    task: TaskRecord
    steps: List[TrajectoryStep] = field(default_factory=list)
    success: bool = False
    average_reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundMetrics:
    round_id: int
    task_count: int
    success_count: int
    average_success_rate: float
    average_reward: float
    best_so_far: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BestRoundState:
    round_id: Optional[int] = None
    average_success_rate: float = -1.0
    round_dir: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterfactualRecord:
    source_trajectory_id: str
    task_id: str
    removed_agent: str
    removal_start_step: int
    counterfactual_trajectory_id: str
    success: bool
    average_reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillSpec:
    skill_name: str
    base_skill: str
    source: str = "primitive"
    selection_metadata: Dict[str, Any] = field(default_factory=dict)
    detail: Dict[str, Any] = field(default_factory=dict)
    phases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionArtifact:
    name: str
    applied: bool = False
    summary: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundStateSnapshot:
    round_id: int
    round_dir: str
    average_success_rate: float
    is_best: bool
    failed_trajectory_ids: List[str] = field(default_factory=list)
    counterfactual_source: str = "current_round"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillMemoryEntry:
    entry_id: str
    skill_name: str
    task_description: str = ""
    task_goal: str = ""
    subgoal: str = ""
    state_trace_summary: str = ""
    recent_environment_feedback: str = ""
    stage_assignments: Dict[str, List[str]] = field(default_factory=dict)
    stage_subgoals: Dict[str, str] = field(default_factory=dict)
    success: bool = False
    weight: float = 1.0
    memory_metadata: Dict[str, Any] = field(default_factory=default_memory_metadata)
    retrieval_count: int = 0


@dataclass
class ExecutorMemoryEntry:
    entry_id: str
    agent_name: str
    task_description: str = ""
    subgoal: str = ""
    stage_subgoal: str = ""
    state_trace_summary: str = ""
    recent_environment_feedback: str = ""
    memory_metadata: Dict[str, Any] = field(default_factory=default_memory_metadata)
    action_summary: str = ""
    assigned_part: str = ""
    workflow_name: str = ""
    success: bool = False
    reward: float = 0.0
    weight: float = 1.0
    retrieval_count: int = 0


@dataclass
class ProfileSnapshot:
    agent_name: str
    collaboration_style: str = ""
    completed_tasks: int = 0
    successful_tasks: int = 0
    average_reward: float = 0.0
    preferred_skills: Dict[str, float] = field(default_factory=dict)
    preferred_stages: Dict[str, float] = field(default_factory=dict)
    preferred_skill_stages: Dict[str, float] = field(default_factory=dict)
    teammate_affinity: Dict[str, float] = field(default_factory=dict)
    collaboration_tags: List[str] = field(default_factory=list)
    recent_strengths: List[str] = field(default_factory=list)
    recent_failures: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
