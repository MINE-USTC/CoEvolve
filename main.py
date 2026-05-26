from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from coevolve.agents import build_agents
from coevolve.config import load_agent_configs
from coevolve.llm.openai_client import build_client_pool, ThreadAwareClient
from coevolve.pipeline import RoundRunner
from coevolve.task_manager import build_task_manager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Co-Evolve training and evaluation entrypoint")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--dotenv-path", type=str, default=".env")
    parser.add_argument("--orchestrator-config", type=str, default="configs/agents/orchestrator.yaml")
    parser.add_argument("--executor-config", type=str, default="configs/agents/executors.yaml")
    parser.add_argument("--task-manager", type=str, choices=["alfworld", "webshop"], default="alfworld")
    parser.add_argument("--tasks-path", type=str, default="")
    parser.add_argument("--env-config-path", type=str, default="benchmarks/alfworld/environment/alfworld_config.yaml")
    parser.add_argument("--few-shots-path", type=str, default="")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:5000",
        help="WebShop server URL (only used with --task-manager webshop)",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/train")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=2)
    parser.add_argument("--max-steps-per-task", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--default-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--skills-path", type=str, default="configs/skills/default_skills.json")
    parser.add_argument("--counterfactual-samples-per-round", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-best", action="store_true")
    parser.add_argument("--disable-skill-memory", action="store_true")
    parser.add_argument("--disable-executor-memory", action="store_true")
    parser.add_argument("--disable-profiles", action="store_true")
    parser.add_argument("--disable-counterfactual", action="store_true")
    parser.add_argument("--disable-skill-evolution", action="store_true")
    parser.add_argument("--disable-skill-memory-evolution", action="store_true")
    parser.add_argument("--disable-executor-memory-evolution", action="store_true")
    parser.add_argument("--max-reuse-skills-per-round", type=int, default=2)
    parser.add_argument("--max-failure-summary-skills-per-round", type=int, default=1)
    parser.add_argument("--max-evidence-per-candidate", type=int, default=5)
    parser.add_argument("--memory-structure-max-cases", type=int, default=3)
    parser.add_argument("--memory-score-initial", type=float, default=0.5)
    parser.add_argument("--memory-score-lambda", type=float, default=0.2)
    parser.add_argument("--memory-score-omega", type=float, default=0.05)
    parser.add_argument("--memory-score-min", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for task execution and counterfactual sampling")
    parser.add_argument("--clean", action="store_true", help="Clean start: ignore any pre-existing memory/state files")
    parser.add_argument("--api-keys", type=str, default="", help="Comma-separated API keys for round-robin across threads")
    return parser


def _resolve_tasks_path(repo_root: Path, mode: str, task_manager: str, tasks_path_arg: str) -> str:
    if tasks_path_arg:
        return str((repo_root / tasks_path_arg).resolve())
    if task_manager == "alfworld":
        if mode == "train":
            preferred = repo_root / "benchmarks/alfworld/tasks/alfworld_tasks_train.json"
            fallback = repo_root / "benchmarks/alfworld/tasks/alfworld_tasks_valid_train.json"
            return str(preferred.resolve()) if preferred.exists() else str(fallback.resolve())
        preferred = repo_root / "benchmarks/alfworld/tasks/alfworld_tasks_valid_unseen.json"
        fallback = repo_root / "benchmarks/alfworld/tasks/alfworld_tasks_valid_seen.json"
        return str(preferred.resolve()) if preferred.exists() else str(fallback.resolve())
    if task_manager == "webshop":
        if mode == "train":
            return str((repo_root / "benchmarks/webshop/tasks/webshop_tasks_train.json").resolve())
        return str((repo_root / "benchmarks/webshop/tasks/webshop_tasks_test.json").resolve())
    return ""


def _resolve_few_shots_path(repo_root: Path, task_manager: str, few_shots_arg: str) -> str:
    if few_shots_arg:
        return str((repo_root / few_shots_arg).resolve())
    if task_manager == "alfworld":
        return str((repo_root / "benchmarks/alfworld/data/alfworld_few_shots.json").resolve())
    if task_manager == "webshop":
        return str((repo_root / "benchmarks/webshop/data/webshop_few_shots.json").resolve())
    return ""


def _resolve_output_dir(
    repo_root: Path,
    output_dir_arg: str,
    task_manager: str,
    now: datetime | None = None,
) -> Path:
    raw_path = Path(output_dir_arg)
    if output_dir_arg.strip() == "outputs/train":
        timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
        benchmark_name = str(task_manager or "run").strip() or "run"
        return repo_root / "outputs" / f"{benchmark_name}_{timestamp}"
    return raw_path if raw_path.is_absolute() else repo_root / raw_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    orchestrator_config_path = repo_root / args.orchestrator_config
    executor_config_path = repo_root / args.executor_config
    output_dir = _resolve_output_dir(
        repo_root=repo_root,
        output_dir_arg=args.output_dir,
        task_manager=args.task_manager,
    )
    tasks_path = _resolve_tasks_path(
        repo_root=repo_root,
        mode=args.mode,
        task_manager=args.task_manager,
        tasks_path_arg=args.tasks_path,
    )
    env_config_path = str((repo_root / args.env_config_path).resolve()) if args.env_config_path else ""
    few_shots_path = _resolve_few_shots_path(repo_root, args.task_manager, args.few_shots_path)
    skills_path = str((repo_root / args.skills_path).resolve()) if args.skills_path else ""

    orchestrator_config, executor_configs = load_agent_configs(
        orchestrator_config_path=orchestrator_config_path,
        executor_config_path=executor_config_path,
        default_model=args.default_model,
    )
    pool = build_client_pool(api_keys_csv=args.api_keys, dotenv_path=repo_root / args.dotenv_path)
    client = ThreadAwareClient(pool)
    orchestrator, executors = build_agents(orchestrator_config, executor_configs, client)
    task_manager = build_task_manager(
        args.task_manager,
        tasks_path=tasks_path,
        env_config_path=env_config_path,
        max_trials=args.max_trials,
        few_shots_path=few_shots_path,
        mode=args.mode,
        server_url=args.server_url,
    )
    task_manager_kwargs = {
        "tasks_path": tasks_path,
        "env_config_path": env_config_path,
        "max_trials": args.max_trials,
        "few_shots_path": few_shots_path,
        "mode": args.mode,
        "server_url": args.server_url,
    }

    runner = RoundRunner(
        orchestrator=orchestrator,
        executors=executors,
        task_manager=task_manager,
        output_dir=str(output_dir),
        skills_path=skills_path,
        enable_skill_memory=not args.disable_skill_memory,
        enable_executor_memory=not args.disable_executor_memory,
        enable_profiles=not args.disable_profiles,
        enable_counterfactual=not args.disable_counterfactual,
        enable_skill_evolution=not args.disable_skill_evolution,
        enable_skill_memory_evolution=not args.disable_skill_memory_evolution,
        enable_executor_memory_evolution=not args.disable_executor_memory_evolution,
        max_reuse_skills_per_round=args.max_reuse_skills_per_round,
        max_failure_summary_skills_per_round=args.max_failure_summary_skills_per_round,
        max_evidence_per_candidate=args.max_evidence_per_candidate,
        memory_structure_max_cases=args.memory_structure_max_cases,
        memory_score_initial=args.memory_score_initial,
        memory_score_lambda=args.memory_score_lambda,
        memory_score_omega=args.memory_score_omega,
        memory_score_min=args.memory_score_min,
        max_workers=args.workers,
        clean_start=args.clean,
        task_manager_name=args.task_manager,
        task_manager_kwargs=task_manager_kwargs,
    )
    runner.run(
        mode=args.mode,
        rounds=args.rounds,
        sample_size=args.sample_size,
        max_steps_per_task=args.max_steps_per_task,
        base_seed=args.seed,
        counterfactual_samples_per_round=args.counterfactual_samples_per_round,
        resume=args.resume,
        evaluate_best=args.evaluate_best,
    )


if __name__ == "__main__":
    main()
