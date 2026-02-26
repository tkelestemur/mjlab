# Franka Research 3 (FR3)

7-DOF robot arm with Franka Hand (parallel-jaw gripper), sourced from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie).

## Registered Tasks

| Task ID | Description |
|---------|-------------|
| `Mjlab-Lift-Cube-Franka-FR3` | Lift a cube to a target height |

## Training

```bash
uv run train Mjlab-Lift-Cube-Franka-FR3
```

Override defaults with CLI flags:

```bash
uv run train Mjlab-Lift-Cube-Franka-FR3 --env.scene.num-envs 4096 --agent.max-iterations 10000
```

## Visualization

View a trained policy:

```bash
uv run play Mjlab-Lift-Cube-Franka-FR3 --wandb-run-path <user/project/run_id>
```

Or test with a zero/random agent (no checkpoint needed):

```bash
uv run play Mjlab-Lift-Cube-Franka-FR3 --agent zero
uv run play Mjlab-Lift-Cube-Franka-FR3 --agent random
```

## Standalone Viewer

Preview the robot model without an environment:

```bash
uv run python -m mjlab.asset_zoo.robots.franka_fr3.fr3_constants
```
