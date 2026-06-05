"""Differentiable forward kinematics for the G1 humanoid (PyTorch).

Thin wrapper over `pytorch_kinematics`: the kinematic chain is parsed once from
the G1 MJCF and FK is evaluated in batched, autograd-friendly PyTorch.

Frame convention: positions are returned in the pelvis-local frame — the pelvis
is treated as identity at the origin, so global root pose is ignored. Validated
to ~1e-8 against mujoco mj_forward.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn
import pytorch_kinematics as pk

FOOT_LINKS   = ["left_ankle_roll_link", "right_ankle_roll_link"]
FOOT_OFFSETS = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
HAND_LINKS   = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
HAND_OFFSETS = [[0.08, 0.0, 0.0], [0.08, 0.0, 0.0]]
PELVIS_LINK  = "pelvis"


def _sanitised_mjcf(xml_path: Path) -> bytes:
    """Robot MJCF with free joint, geoms, meshes, and contacts removed."""
    robot_xml = xml_path.parent / "g1_mjx.xml"
    tree = ET.parse(str(robot_xml))
    root = tree.getroot()
    for parent in root.iter():
        to_remove = [
            c for c in parent
            if c.tag in ("geom", "mesh", "pair", "contact")
            or c.tag == "freejoint"
            or (c.tag == "joint" and c.get("type") == "free")
        ]
        for elem in to_remove:
            parent.remove(elem)
    return ET.tostring(root, encoding="unicode").encode()


def _default_qpos(xml_path: Path) -> np.ndarray:
    m   = mujoco.MjModel.from_xml_path(str(xml_path))
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid < 0:
        raise RuntimeError("No 'home' keyframe in G1 XML")
    return m.key_qpos[kid, 7:].astype(np.float32).copy()


def _hinge_order(xml_path: Path) -> list[str]:
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    return [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]


def _joint_limits(xml_path: Path) -> np.ndarray:
    """(29, 2) absolute [lo, hi] limits for hinge joints in qpos[7:] order.

    Unlimited joints get [-inf, +inf], matching the joint_pos feature layout.
    """
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    lims = [
        m.jnt_range[j].copy() if m.jnt_limited[j] else np.array([-np.inf, np.inf])
        for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    return np.asarray(lims, dtype=np.float32)


class G1Kinematics(nn.Module):
    """Differentiable pelvis-local FK for the G1, evaluated from joint offsets.

    forward() takes joint offsets from the default pose — the ``joint_pos`` block
    from the canonical features (angle minus default_qpos) — with shape (..., 29)
    in MuJoCo qpos[7:] order, and returns pelvis-local site positions:
        {"feet": (..., 2, 3), "hands": (..., 2, 3)}
    Index 0 is the left site, index 1 the right, for both groups.
    """

    def __init__(self, xml_path: str) -> None:
        super().__init__()
        xml_path = Path(xml_path)
        self.chain = pk.build_chain_from_mjcf(_sanitised_mjcf(xml_path))

        chain_joints = self.chain.get_joint_parameter_names()
        mj_order     = _hinge_order(xml_path)
        perm = [mj_order.index(name) for name in chain_joints]
        self.register_buffer("perm",        torch.tensor(perm, dtype=torch.long))
        self.register_buffer("default_qpos", torch.from_numpy(_default_qpos(xml_path)))
        self.register_buffer("joint_limits", torch.from_numpy(_joint_limits(xml_path)))

        self._foot_off = torch.tensor(FOOT_OFFSETS)
        self._hand_off = torch.tensor(HAND_OFFSETS)

    def _to(self, ref: torch.Tensor) -> None:
        self.chain = self.chain.to(device=ref.device, dtype=ref.dtype)
        if self._foot_off.device != ref.device or self._foot_off.dtype != ref.dtype:
            self._foot_off = self._foot_off.to(device=ref.device, dtype=ref.dtype)
            self._hand_off = self._hand_off.to(device=ref.device, dtype=ref.dtype)

    def forward(self, joint_pos: torch.Tensor) -> dict[str, torch.Tensor]:
        """(..., 29) joint offsets → {"feet": (..., 2, 3), "hands": (..., 2, 3)}."""
        self._to(joint_pos)
        lead  = joint_pos.shape[:-1]
        q_abs = joint_pos.reshape(-1, 29) + self.default_qpos
        q_chain = q_abs.index_select(-1, self.perm)

        ret      = self.chain.forward_kinematics(q_chain)
        pelvis_p = ret[PELVIS_LINK].get_matrix()[:, :3, 3]

        def _sites(links: list[str], offsets: torch.Tensor) -> torch.Tensor:
            pts = []
            for k, name in enumerate(links):
                M     = ret[name].get_matrix()
                world = M[:, :3, 3] + (M[:, :3, :3] @ offsets[k])
                pts.append(world - pelvis_p)
            return torch.stack(pts, dim=1)

        feet  = _sites(FOOT_LINKS,  self._foot_off).reshape(*lead, 2, 3)
        hands = _sites(HAND_LINKS,  self._hand_off).reshape(*lead, 2, 3)
        return {"feet": feet, "hands": hands}
