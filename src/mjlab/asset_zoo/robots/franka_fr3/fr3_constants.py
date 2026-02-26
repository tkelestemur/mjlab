"""Franka Research 3 (FR3) constants.

FR3 7-DOF arm with Franka Hand (parallel-jaw gripper from Panda).
Actuator gains and joint properties from mujoco_menagerie.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

FR3_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "franka_fr3" / "xmls" / "fr3.xml"
)
assert FR3_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, FR3_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(FR3_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config (values from mujoco_menagerie).
##

# Joints 1-2: large shoulder motors.
ACTUATOR_SHOULDER = BuiltinPositionActuatorCfg(
  target_names_expr=("joint1", "joint2"),
  stiffness=4500.0,
  damping=450.0,
  effort_limit=87.0,
  armature=0.195,
)

# Joints 3-4: elbow motors.
ACTUATOR_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=("joint3", "joint4"),
  stiffness=3500.0,
  damping=350.0,
  effort_limit=87.0,
  armature=0.195,
)

# Joints 5-7: wrist motors.
ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=("joint5", "joint6", "joint7"),
  stiffness=2000.0,
  damping=200.0,
  effort_limit=12.0,
  armature=0.074,
)

# Gripper: only finger_joint1 is actuated; finger_joint2 coupled via equality.
ACTUATOR_GRIPPER = BuiltinPositionActuatorCfg(
  target_names_expr=("finger_joint1",),
  stiffness=100.0,
  damping=10.0,
  effort_limit=20.0,
  armature=0.01,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "joint1": 0.0,
    "joint2": -0.785,
    "joint3": 0.0,
    "joint4": -2.356,
    "joint5": 0.0,
    "joint6": 1.571,
    "joint7": 0.785,
    "finger_joint1": 0.04,
    "finger_joint2": 0.04,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={
    "(left|right)_pad[0-9]+_collision": 6,
    ".*_collision": 3,
  },
  friction={
    "(left|right)_pad[0-9]+_collision": (1, 5e-3, 5e-4),
    ".*_collision": (0.6,),
  },
  solref={
    "(left|right)_pad[0-9]+_collision": (0.01, 1),
  },
  priority={
    "(left|right)_pad[0-9]+_collision": 1,
  },
)

GRIPPER_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype={
    "(hand|left_finger|right_finger|(left|right)_pad[0-9]+)_collision": 1,
    ".*_collision": 0,
  },
  conaffinity={
    "(hand|left_finger|right_finger|(left|right)_pad[0-9]+)_collision": 1,
    ".*_collision": 0,
  },
  condim={
    "(left|right)_pad[0-9]+_collision": 6,
    ".*_collision": 3,
  },
  friction={
    "(left|right)_pad[0-9]+_collision": (1, 5e-3, 5e-4),
    ".*_collision": (0.6,),
  },
  solref={
    "(left|right)_pad[0-9]+_collision": (0.01, 1),
  },
  priority={
    "(left|right)_pad[0-9]+_collision": 1,
  },
)

##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ACTUATOR_SHOULDER, ACTUATOR_ELBOW, ACTUATOR_WRIST, ACTUATOR_GRIPPER),
  soft_joint_pos_limit_factor=0.9,
)


def get_fr3_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(GRIPPER_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


FR3_ACTION_SCALE: dict[str, float] = {}
for a in ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    FR3_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_fr3_robot_cfg())

  viewer.launch(robot.spec.compile())
