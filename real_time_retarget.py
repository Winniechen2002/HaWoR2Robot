import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig

import genesis as gs

def start_retargeting(pred_glob_l, pred_glob_r, robot_dir: str, config_path: str):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    config = RetargetingConfig.load_from_file(config_path)

    gs.init(backend=gs.gpu)
    
    scene = gs.Scene(
        viewer_options = gs.options.ViewerOptions(
            camera_pos    = (0, -3.5, 2.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 30,
            max_FPS       = 60,
        ),
        sim_options = gs.options.SimOptions(
            dt = 0.01,
        ),
        show_viewer = True,
    )
    
    robot_right = scene.add_entity(
        morph=gs.morphs.URDF(
            pos = (0.0, 0.0, 0.5),
            scale=1.0,
            file="/home/cf24/dex-retargeting/assets/robots/hands/shadow_hand/shadow_hand_right.urdf",
        ),
        surface=gs.surfaces.Reflective(color=(0.4, 0.4, 0.4)),
    )
    
    robot_left = scene.add_entity(
        morph=gs.morphs.URDF(
            pos = (0.0, 0.2, 0.5),
            scale=1.0,
            file="/home/cf24/dex-retargeting/assets/robots/hands/shadow_hand/shadow_hand_left.urdf",
        ),
        surface=gs.surfaces.Reflective(color=(0.4, 0.4, 0.4)),
    )
    
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    scene.build()

    # Different robot loader may have different orders for joints
    robot_joint_name = [joint.name for joint in robot.joints[1:]]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_genesis = np.array([retargeting_joint_names.index(name) for name in robot_joint_name]).astype(int)

    import pdb; pdb.set_trace()
    
    len_frame = pred_glob_l['joints'].shape[1]
    
    for _ in range(len_frame):
        
        retargeting_type = retargeting.optimizer.retargeting_type
        indices = retargeting.optimizer.target_link_human_indices
        if retargeting_type == "POSITION":
            indices = indices
            ref_value = joint_pos[indices, :]
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        qpos = retargeting.retarget(ref_value)
        robot.set_dofs_position([0.0], np.arange(0, 1))
        robot.set_dofs_position(qpos[retargeting_to_genesis], np.arange(1, len(robot.joints)))

        scene.step()