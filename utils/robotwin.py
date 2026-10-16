"""RoboTwin joint/EE6D validation and the original X-VLA EE6D convention.

Reference: vlarp/src/vlarp/{data/xvla_contract,policies/xvla,envs/robotwin_xvla}.py.
This encoding intentionally differs from the LIBERO X-VLA rotation layout.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROBOTWIN_JOINT_CONTRACT = "robotwin-joint-14d"
ROBOTWIN_CONTRACT = "robotwin-xvla-official-v2"
ROBOTWIN_CAMERA_MAP = {
    "observation.images.cam_high": "observation.images.head_camera",
    "observation.images.cam_left_wrist": "observation.images.left_camera",
    "observation.images.cam_right_wrist": "observation.images.right_camera",
}


def robotwin_uses_ee6d(model, model_path):
    if model != "xvla":
        return False
    path = Path(model_path)
    if path.name == "X-VLA-RoboTwin2":
        return True
    config_path = path / "config.json"
    return config_path.is_file() and json.loads(config_path.read_text()).get("action_mode") == "ee6d"


def default_robotwin_dataset(model, model_path):
    name = "robotwin" if robotwin_uses_ee6d(model, model_path) else "robotwin_unified"
    return Path(__file__).resolve().parents[1] / "data" / name


def read_robotwin_info(dataset_root):
    info = json.loads((Path(dataset_root) / "meta/info.json").read_text())
    shapes = [info["features"][key]["shape"] for key in ("observation.state", "action")]
    if "xvla_ee6d" not in info and shapes == [[14], [14]]:
        return info
    contract = info.get("xvla_ee6d", {})
    if contract.get("contract_version") != ROBOTWIN_CONTRACT:
        raise ValueError(f"RoboTwin EE6D action contract must be {ROBOTWIN_CONTRACT}; native joint data must have width 14.")
    if contract.get("domain_id") != 6:
        raise ValueError("RoboTwin EE6D requires domain_id=6")
    if shapes != [[20], [20]]:
        raise ValueError("RoboTwin EE6D requires state/action shape [20]")
    return info


def validate_robotwin_training(dataset_root, model, model_path):
    """Reject incompatible action spaces before allocating the model on a GPU."""
    info = read_robotwin_info(dataset_root)
    width = info["features"]["action"]["shape"][0]
    if width == 20 and model != "xvla":
        raise ValueError(f"The {model} RoboTwin policy uses 14D joint actions; it cannot consume 20D EE6D data.")
    config = json.loads((Path(model_path) / "config.json").read_text())
    if width == 20:
        if config.get("action_mode") != "ee6d":
            raise ValueError("RoboTwin EE6D requires an X-VLA ee6d checkpoint")
        for feature, features in [("action", "output_features"), ("observation.state", "input_features")]:
            shape = config.get(features, {}).get(feature, {}).get("shape")
            if shape is not None and shape != [20]:
                raise ValueError(f"RoboTwin checkpoint {feature} must have shape [20], got {shape}")
        return ROBOTWIN_CONTRACT
    if robotwin_uses_ee6d(model, model_path):
        raise ValueError("X-VLA EE6D requires 20D data, not 14D joint data")
    shape = config.get("output_features", {}).get("action", {}).get("shape")
    if shape != [14]:
        raise ValueError(f"RoboTwin joint checkpoint action must have shape [14], got {shape}")
    return ROBOTWIN_JOINT_CONTRACT


def to_xvla_robotwin_observation(observation):
    result = dict(observation)
    for source, destination in ROBOTWIN_CAMERA_MAP.items():
        if source in result:
            result[destination] = result.pop(source)
    return result


def absolute_ee_state_to_ee6d(state):
    """[... ,16] absolute xyz/raw-wxyz/open -> [... ,20] model observation.

    Preserve the official client's interpretation of raw wxyz as SciPy components;
    do not permute the quaternion or concatenate entire matrix columns here.
    """
    state = np.asarray(state, dtype=np.float64)
    if state.ndim < 1 or state.shape[-1] != 16 or not np.isfinite(state).all():
        raise ValueError("Expected finite absolute RoboTwin EE state [...,16]")
    shape = state.shape[:-1]
    rows = state.reshape(-1, 16)
    parts = []
    for offset in (0, 8):
        pose = rows[:, offset:offset + 8]
        rotation = Rotation.from_quat(pose[:, 3:7]).as_matrix()[:, :, :2].reshape(-1, 6)
        parts.extend((pose[:, :3], rotation, 1 - 2 * pose[:, 7:8]))
    return np.concatenate(parts, axis=-1).reshape(*shape, 20).astype(np.float32)


def ee6d_actions_to_absolute_ee(action):
    """[... ,20] model actions -> [... ,16] absolute RoboTwin EE commands.

    Grippers are close probabilities, thresholded exactly as in the official client.
    These absolute commands still need the vlarp bridge when using LeRobot's
    initial-pose-relative RoboTwin environment (see docs/robotwin_ace.md).
    """
    action = np.asarray(action, dtype=np.float64)
    if action.ndim < 1 or action.shape[-1] != 20 or not np.isfinite(action).all():
        raise ValueError("Expected finite RoboTwin EE6D actions [...,20]")
    shape = action.shape[:-1]
    rows = action.reshape(-1, 20)
    parts = []
    for offset in (0, 10):
        rotation = rows[:, offset + 3:offset + 9].reshape(-1, 3, 2)
        first, second = rotation[:, :, 0], rotation[:, :, 1]
        norm = np.linalg.norm(first, axis=-1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError("Degenerate first rotation column")
        first = first / norm
        second = second - np.sum(first * second, axis=-1, keepdims=True) * first
        norm = np.linalg.norm(second, axis=-1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError("Degenerate second rotation column")
        second = second / norm
        matrix = np.stack((first, second, np.cross(first, second)), axis=-1)
        quaternion = Rotation.from_matrix(matrix).as_quat()
        gripper = 1 - 2 * (rows[:, offset + 9:offset + 10] > 0.7)
        parts.extend((rows[:, offset:offset + 3], quaternion, gripper))
    return np.concatenate(parts, axis=-1).reshape(*shape, 16).astype(np.float32)
