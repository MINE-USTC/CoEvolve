# Co-Evolve

Co-Evolve is a multi-agent skill and memory co-evolution framework for interactive decision-making tasks. The system coordinates multiple executor agents, selects and assigns reusable workflow skills, collects trajectory-level evidence, samples counterfactual executions, and evolves both the skill library and memory schemas from successful and failed experience.

The current anonymous release keeps the core training and evaluation pipeline for ALFWorld and WebShop. Task data, external simulators, and benchmark-specific environment assets should be prepared separately.

## Overview

At a high level, each task is solved through the following loop:

1. The orchestrator selects one skill from the current skill library.
2. The orchestrator assigns skill stages to executor agents.
3. Executors run the assigned stages and optionally interact with the environment.
4. The trajectory, memories, and profiles are updated from task experience.
5. Counterfactual sampling compares alternative executor participation or workflow choices.
6. Evolution modules propose new skills and update memory metadata structures.

The framework supports:

- Skill-level evolution from successful workflow reuse and failed trajectory gaps.
- Skill memory evolution for better future skill selection.
- Executor memory evolution for better stage-level actions and failure avoidance.
- Counterfactual trajectory sampling for contrastive evidence.
- Separate task managers for ALFWorld and WebShop.

## Repository Structure

```text
coevolve/
  agents/           Orchestrator and executor agent wrappers.
  counterfactual/   Counterfactual sampling logic.
  evolution/        Skill and memory evolution engines.
  memory/           Skill memory, executor memory, and profile stores.
  pipeline/         Round-level train/test runner and state management.
  skills/           Skill library loading and runtime execution.
  task_manager/     ALFWorld and WebShop task interfaces.
  trajectory/       Trajectory recording utilities.
configs/
  agents/           Orchestrator and executor agent configs.
  skills/           Initial task-specific skill libraries.
main.py             Training and evaluation entrypoint.
```

## Setup

Install the Python dependencies required by your environment, then configure an OpenAI-compatible API key in `.env`:

```bash
OPENAI_API_KEY=your_api_key_here
```

The repository includes a blank `.env` placeholder for convenience. You can also pass multiple keys at runtime:

```bash
python main.py --api-keys key1,key2,key3
```

## Running

Run ALFWorld training:

```bash
python main.py \
  --mode train \
  --task-manager alfworld \
  --tasks-path benchmarks/alfworld/tasks/alfworld_tasks_train.json \
  --env-config-path benchmarks/alfworld/environment/alfworld_config.yaml \
  --few-shots-path benchmarks/alfworld/data/alfworld_few_shots.json \
  --output-dir outputs/alfworld_train
```

Run WebShop training:

```bash
python main.py \
  --mode train \
  --task-manager webshop \
  --tasks-path benchmarks/webshop/tasks/webshop_tasks_train.json \
  --few-shots-path benchmarks/webshop/data/webshop_few_shots.json \
  --server-url http://localhost:5000 \
  --output-dir outputs/webshop_train
```

For evaluation, use `--mode test` and provide the corresponding test task file. If `--tasks-path` or `--few-shots-path` is omitted, `main.py` falls back to the default benchmark paths under `benchmarks/`.

## Initial Skills

Initial skills are defined in `configs/skills/default_skills.json`. The file contains separate `plan`, `execute`, and `reflect` primitive skills for each supported task manager:

```text
task_skills.alfworld.skills
task_skills.webshop.skills
```

At runtime, the pipeline automatically loads the task-specific initial skills according to `--task-manager`. You can override this behavior by passing a custom `--skills-path`.

## Outputs

Each run writes structured artifacts under the selected output directory, including:

- Per-round trajectory records.
- Skill library snapshots.
- Skill and executor memory snapshots.
- Counterfactual sampling records.
- Round metrics and logs.

Use `--resume` to continue from an existing output directory and `--clean` to ignore prior run state.

## Notes

- This release focuses on the core Co-Evolve pipeline and does not bundle benchmark datasets.
- ALFWorld and WebShop environments must be installed and launched according to their own instructions.
- The default code supports only `alfworld` and `webshop` task managers in this anonymous version.
- Generated files such as `outputs/`, memory stores, caches, and bytecode are ignored by `.gitignore`.
