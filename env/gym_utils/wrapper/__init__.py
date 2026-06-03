from .multi_step import MultiStep
from .robomimic_lowdim import RobomimicLowdimWrapper
from .robomimic_image import RobomimicImageWrapper
from .d3il_lowdim import D3ilLowdimWrapper
from .mujoco_locomotion_lowdim import MujocoLocomotionLowdimWrapper
from .g1_kinematic_lowdim import G1KinematicEnv


wrapper_dict = {
    "multi_step": MultiStep,
    "robomimic_lowdim": RobomimicLowdimWrapper,
    "robomimic_image": RobomimicImageWrapper,
    "d3il_lowdim": D3ilLowdimWrapper,
    "mujoco_locomotion_lowdim": MujocoLocomotionLowdimWrapper,
    "g1_kinematic": G1KinematicEnv,
}
