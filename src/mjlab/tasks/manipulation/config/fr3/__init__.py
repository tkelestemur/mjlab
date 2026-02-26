from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import fr3_lift_cube_env_cfg
from .rl_cfg import fr3_lift_cube_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Franka-FR3",
  env_cfg=fr3_lift_cube_env_cfg(),
  play_env_cfg=fr3_lift_cube_env_cfg(play=True),
  rl_cfg=fr3_lift_cube_ppo_runner_cfg(),
)
