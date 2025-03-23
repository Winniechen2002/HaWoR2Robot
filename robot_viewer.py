import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import sapien
from hand_viewer import HandViewer
from pytransform3d import rotations
from tqdm import trange

from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

import torch

class RobotViewer(HandViewer):
    def __init__(self, data, robot_names: List[RobotName], headless=True, use_ray_tracing=False):
        super().__init__(data, headless=headless, use_ray_tracing=use_ray_tracing)

        self.robot_names = robot_names
        
        self.right_robots: List[sapien.Articulation] = []
        self.right_robot_file_names: List[str] = []
        self.right_retargetings: List[SeqRetargeting] = []
        self.right_retarget2sapien: List[np.ndarray] = []
        
        self.left_robots: List[sapien.Articulation] = []
        self.left_robot_file_names: List[str] = []
        self.left_retargetings: List[SeqRetargeting] = []
        self.left_retarget2sapien: List[np.ndarray] = []

        # Load optimizer and filter
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        
        for robot_name in robot_names:
            config_path = get_default_config_path(robot_name, RetargetingType.position, HandType.right)

            # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            robot_file_name = Path(config.urdf_path).stem
            self.right_robot_file_names.append(robot_file_name)
            self.right_retargetings.append(retargeting)

            # Build robot
            urdf_path = Path(config.urdf_path)
            if "glb" not in urdf_path.stem:
                urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
            robot_urdf = urdf.URDF.load(str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
            urdf_name = urdf_path.name
            temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
            temp_path = f"{temp_dir}/{urdf_name}"
            robot_urdf.write_xml_file(temp_path)

            robot = loader.load(temp_path)
            self.right_robots.append(robot)
            sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
            retarget2sapien = np.array([retargeting.joint_names.index(n) for n in sapien_joint_names]).astype(int)
            self.right_retarget2sapien.append(retarget2sapien)
        
        
        for robot_name in robot_names:
            config_path = get_default_config_path(robot_name, RetargetingType.position, HandType.left)

            # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            robot_file_name = Path(config.urdf_path).stem
            self.left_robot_file_names.append(robot_file_name)
            self.left_retargetings.append(retargeting)

            # Build robot
            urdf_path = Path(config.urdf_path)
            if "glb" not in urdf_path.stem:
                urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
            robot_urdf = urdf.URDF.load(str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
            urdf_name = urdf_path.name
            temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
            temp_path = f"{temp_dir}/{urdf_name}"
            robot_urdf.write_xml_file(temp_path)

            robot = loader.load(temp_path)
            self.left_robots.append(robot)
            sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
            retarget2sapien = np.array([retargeting.joint_names.index(n) for n in sapien_joint_names]).astype(int)
            self.left_retarget2sapien.append(retarget2sapien)

    def render_dexycb_data(self, fps=5, y_offset=0.8):
        # Set table and viewer pose for better visual effect only
        global_y_offset = -y_offset * len(self.right_robots) / 2
        if not self.headless:
            self.viewer.set_camera_xyz(1.5, global_y_offset, 1)
        else:
            local_pose = self.camera.get_local_pose()
            local_pose.set_p(np.array([1.5, global_y_offset, 1]))
            self.camera.set_local_pose(local_pose)

        left_vertices = self.left_hand_vertices
        left_joints = self.left_hand_joints
        
        right_vertices = self.right_hand_vertices
        right_joints = self.right_hand_joints
        
        num_frame = left_vertices.shape[0]
        
        pose_offsets = []

        for i in range(len(self.left_robots) + 1):
            pose = sapien.Pose([0, -y_offset * i, 0])
            pose_offsets.append(pose)
            if i >= 1:
                self.left_robots[i - 1].set_pose(pose)
                
        for i in range(len(self.right_robots) + 1):
            pose = sapien.Pose([0, -y_offset * (i + 0.2), 0])
            pose_offsets.append(pose)
            if i >= 1:
                self.right_robots[i - 1].set_pose(pose)

        if self.headless:
            robot_names = [robot.name for robot in self.robot_names]
            robot_names = "_".join(robot_names)
            video_path = Path(__file__).parent.resolve() / f"data/{robot_names}_video.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (self.camera.get_width(), self.camera.get_height()),
            )

        # Loop rendering
        step_per_frame = int(60 / fps)
        for i in range(num_frame):
            left_vertex, left_joint = left_vertices[i], left_joints[i]
            right_vertex, right_joint = right_vertices[i], right_joints[i]
            
            # Update pose for human hand
            self._update_left_hand(left_vertex)
            self._update_right_hand(right_vertex)

            # Update poses for robot hands
            for robot, retargeting, retarget2sapien in zip(self.right_robots, self.right_retargetings, self.right_retarget2sapien):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = right_joint[indices, :].cpu().numpy()
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                robot.set_qpos(qpos)
                
            for robot, retargeting, retarget2sapien in zip(self.left_robots, self.left_retargetings, self.left_retarget2sapien):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = left_joint[indices, :].cpu().numpy()
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                robot.set_qpos(qpos)

            self.scene.update_render()
            if self.headless:
                self.camera.take_picture()
                rgb = self.camera.get_picture("Color")[..., :3]
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                writer.write(rgb[..., ::-1])
            else:
                for _ in range(step_per_frame):
                    self.viewer.render()

        if not self.headless:
            self.viewer.paused = True
            self.viewer.render()
        else:
            writer.release()
