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



import argparse
import sys
import os

import torch
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import joblib
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam


import genesis as gs
import torch
import open3d as o3d
import numpy as np
import cv2

from scipy.spatial.transform import Rotation as R

def render_hand_on_image(background_img, vertices, faces, intrinsics, R, t):
    # Ensure correct shapes and types
    vertices = np.asarray(vertices).astype(np.float64)
    if vertices.shape[1] != 3:
        vertices = vertices.T
    faces = np.asarray(faces).astype(np.int32)
    if faces.shape[1] != 3:
        faces = faces.T

    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # Apply transformation (world to camera)
    # transform = np.eye(4)
    # transform[:3, :3] = R
    # transform[:3, 3] = t
    # mesh.transform(transform)

    # Set up offscreen renderer
    width, height = background_img.shape[1], background_img.shape[0]
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    # Create material
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"

    renderer.scene.add_geometry("hand", mesh, material)

    # Intrinsics
    fx, fy, cx, cy = intrinsics
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # Extrinsic: world to camera
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    # Set camera
    renderer.setup_camera(intrinsic, extrinsic)

    # Render
    o3d_image = renderer.render_to_image()
    
    # import pdb; pdb.set_trace()
    
    renderer.scene.clear_geometry()

    # Convert to NumPy and blend
    rgb = np.asarray(o3d_image)
    if rgb.shape[2] == 4:  # remove alpha if present
        rgb = rgb[:, :, :3]
    blended = cv2.addWeighted(rgb, 0.2, background_img, 1.0, 0)

    return blended

def write_video_from_images(image_list, output_path, fps=30):
    if not image_list:
        raise ValueError("image_list is empty")

    # Get height, width from first image
    height, width, _ = image_list[0].shape
    size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG'
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    for img in image_list:
        # Ensure the image matches the expected size
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, size)
        writer.write(img)

    writer.release()
    print(f"Video saved to: {output_path}")


from sapien.core import Pose

class RobotViewer(HandViewer):
    def __init__(self, data, robot_names: List[RobotName], headless=True, use_ray_tracing=False, simulator="sapien"):
        super().__init__(data, headless=headless, use_ray_tracing=use_ray_tracing, simulator=simulator)

        self.robot_names = robot_names
        
        if simulator == "sapien":
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
                
        elif simulator == "Genesis":
            self.right_robots = []
            self.right_robot_file_names: List[str] = []
            self.right_retargetings: List[SeqRetargeting] = []
            self.right_retarget2sapien: List[np.ndarray] = []
            
            self.left_robots = []
            self.left_robot_file_names: List[str] = []
            self.left_retargetings: List[SeqRetargeting] = []
            self.left_retarget2sapien: List[np.ndarray] = []
            
            for robot_name in robot_names:
                config_path = get_default_config_path(robot_name, RetargetingType.position, HandType.right)

                # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
                override = dict(add_dummy_free_joint=True)
                config = RetargetingConfig.load_from_file(config_path, override=override)
                retargeting = config.build()
                robot_file_name = Path(config.urdf_path).stem
                self.right_robot_file_names.append(robot_file_name)
                self.right_retargetings.append(retargeting)
                
                robot = self.scene.add_entity(
                    morph=gs.morphs.URDF(
                        scale=1.0,
                        file=config.urdf_path,
                    ),
                    surface=gs.surfaces.Reflective(color=(0.4, 0.4, 0.4)),
                )

                self.right_robots.append(robot)
                robot_joint_name = retargeting.joint_names[:6] + [joint.name for joint in robot.joints[1:]]
                retargeting_joint_names = retargeting.joint_names
                name_to_index = {name: idx for idx, name in enumerate(retargeting_joint_names)}
                retargeting_to_genesis = np.array([
                    name_to_index.get(name, -1)  # Returns -1 if name not found
                    for name in robot_joint_name
                ]).astype(int)
                self.right_retarget2sapien.append(retargeting_to_genesis)
            
            
            for robot_name in robot_names:
                config_path = get_default_config_path(robot_name, RetargetingType.position, HandType.left)

                # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
                override = dict(add_dummy_free_joint=True)
                config = RetargetingConfig.load_from_file(config_path, override=override)
                retargeting = config.build()
                robot_file_name = Path(config.urdf_path).stem
                self.left_robot_file_names.append(robot_file_name)
                self.left_retargetings.append(retargeting)

                robot = self.scene.add_entity(
                    morph=gs.morphs.URDF(
                        scale=1.0,
                        file=config.urdf_path,
                    ),
                    surface=gs.surfaces.Reflective(color=(0.4, 0.4, 0.4)),
                )
                
                self.left_robots.append(robot)
                robot_joint_name = retargeting.joint_names[:6] + [joint.name for joint in robot.joints[1:]]
                retargeting_joint_names = retargeting.joint_names
                name_to_index = {name: idx for idx, name in enumerate(retargeting_joint_names)}
                retargeting_to_genesis = np.array([
                    name_to_index.get(name, -1)  # Returns -1 if name not found
                    for name in robot_joint_name
                ]).astype(int)
                self.left_retarget2sapien.append(retargeting_to_genesis)
            
            self.scene.build()

    def render_dexycb_data(self, fps=5, y_offset=0.8):
        
        left_vertices = self.left_hand_vertices
        left_joints = self.left_hand_joints
        
        right_vertices = self.right_hand_vertices
        right_joints = self.right_hand_joints
        
        if self.headless:
            if self.simulator == "sapien":
                robot_names = [robot.name for robot in self.robot_names]
                robot_names = "_".join(robot_names)
                video_path = Path(__file__).parent.resolve() / f"data/{robot_names}_video.mp4"
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30.0,
                    (self.camera.get_width(), self.camera.get_height()),
                )
            else:
                self.camera.start_recording()
        
        num_frame = left_vertices.shape[0]
        
        pose_offsets = []
        
        in_paint_video = []

        # for i in range(len(self.left_robots)):
        #     pose = sapien.Pose([0, -y_offset * i, 0])
        #     pose_offsets.append(pose)
        #     if i >= 1:
        #         if self.simulator == "Genesis":
        #             self.left_robots[i - 1].set_pos([0, -y_offset * i, 0])
        #         elif self.simulator == "sapien":
        #             self.left_robots[i - 1].set_pose(pose)
                
        # for i in range(len(self.right_robots)):
        #     pose = sapien.Pose([0, -y_offset * (i + 0.2), 0])
        #     pose_offsets.append(pose)
        #     if i >= 1:
        #         if self.simulator == "Genesis":
        #             self.right_robots[i - 1].set_pos([0, -y_offset * i, 0])
        #         elif self.simulator == "sapien":
        #             self.right_robots[i - 1].set_pose(pose)
                
        # Set table and viewer pose for better visual effect only
        global_y_offset = -y_offset * len(self.right_robots) / 2
        if not self.headless:
            if self.simulator == "sapien":
                self.viewer.set_camera_xyz(1.5, global_y_offset, 2)
        # else:
            # local_pose = self.camera.get_local_pose()
            # local_pose.set_p(np.array([1.5, global_y_offset, 2]))
            # self.camera.set_local_pose(local_pose)

        # Loop rendering
        step_per_frame = int(60 / fps)
        for i in range(num_frame):
            left_vertex, left_joint = left_vertices[i], left_joints[i]
            right_vertex, right_joint = right_vertices[i], right_joints[i]
            
            # Update pose for human hand
            # self._update_left_hand(left_vertex)
            # self._update_right_hand(right_vertex)

            # Update poses for robot hands
            for robot, retargeting, retarget2sapien in zip(self.right_robots, self.right_retargetings, self.right_retarget2sapien):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = right_joint[indices, :].cpu().numpy()
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                if self.simulator == "Genesis":
                    robot.set_dofs_position(qpos)
                elif self.simulator == "sapien":
                    robot.set_qpos(qpos)
                
            for robot, retargeting, retarget2sapien in zip(self.left_robots, self.left_retargetings, self.left_retarget2sapien):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = left_joint[indices, :].cpu().numpy()
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                if self.simulator == "Genesis":
                    robot.set_dofs_position(qpos)
                elif self.simulator == "sapien":
                    robot.set_qpos(qpos)

            if self.simulator == "Genesis":
                for x in range(5):
                    self.scene.step()
                if self.headless:
                    R_mat = self.data.R_w2c_sla_all[i].numpy().astype(np.float32)  # (3,3)
                    t_vec = self.data.t_w2c_sla_all[i].numpy().astype(np.float32)  # (3,)
                    # Invert the pose to get camera-to-world
                    R_c2w = R_mat.T
                    pos = -R_c2w @ t_vec  # camera position in world
                    look_dir = R_c2w @ np.array([0, 0, 1], dtype=np.float32)  # camera's +Z
                    lookat = pos + look_dir
                    up = R_c2w @ np.array([0, -1, 0], dtype=np.float32)  # camera's -Y is "up"
                    
                    self.camera.set_pose(pos=pos, lookat=lookat, up=up)
                    # self.camera.set_pose(transform=transform)
                    
                    img = cv2.imread(self.image[i])
                    rgb = self.camera.render()[0]
                    img = cv2.addWeighted(rgb, 1.0, img, 1.0, 0)
                    cv2.imwrite(os.path.join('test/vis', f"video_{i}.png"), img)
                    in_paint_video.append(img)
                    
            elif self.simulator == "sapien":
                self.scene.update_render()
                if self.headless:
                    R_mat = self.data.R_w2c_sla_all[i].numpy().astype(np.float32)  # (3,3)
                    t_vec = self.data.t_w2c_sla_all[i].numpy().astype(np.float32)  # (3,)
                    
                    # Convert rotation matrix to quaternion (x, y, z, w)
                    quat = R.from_matrix(R_mat).as_quat().astype(np.float32)  # → (x, y, z, w) # 4x4 numpy array
                    pose = Pose(p=t_vec, q=(quat[3], quat[0], quat[1], quat[2]))  # (w, x, y, z)
                    self.camera.set_local_pose(pose)
                    self.camera.take_picture()
                    rgb = self.camera.get_picture("Color")[..., :3]
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                    writer.write(rgb[..., ::-1])
                else:
                    for _ in range(step_per_frame):
                        self.viewer.render()

        if not self.headless:
            if self.simulator == "sapien":
                self.viewer.paused = True
                self.viewer.render()
        else:
            if self.simulator == "Genesis":
                self.camera.stop_recording("video.mp4", fps = 24)
            elif self.simulator == "sapien":
                writer.release()
                
        # wirte impaint video
        if self.headless:
            if self.simulator == "Genesis":
                write_video_from_images(in_paint_video, "video_inpaint.mp4", fps=24)

    def render_viewer(self):
        left_vertices = self.left_hand_vertices
        left_joints = self.left_hand_joints
        
        right_vertices = self.right_hand_vertices
        right_joints = self.right_hand_joints
        
        num_frame = left_vertices.shape[0]
        
        left_robot_vertices = []
        right_robot_vertices = []
        
        for i in range(num_frame):
            left_vertex, left_joint = left_vertices[i], left_joints[i]
            right_vertex, right_joint = right_vertices[i], right_joints[i]
            
            # Update pose for human hand
            # self._update_left_hand(left_vertex)
            # self._update_right_hand(right_vertex)

            # Update poses for robot hands
            for robot, retargeting, retarget2sapien in zip(self.right_robots, self.right_retargetings, self.right_retarget2sapien):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = right_joint[indices, :].cpu().numpy()
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                if self.simulator == "Genesis":
                    robot.set_dofs_position(qpos)
                elif self.simulator == "sapien":
                    robot.set_qpos(qpos)
                
                right_robot_vertices.append(robot.get_verts())
                
            for robot, retargeting, retarget2sapien in zip(self.left_robots, self.left_retargetings, self.left_retarget2sapien):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = left_joint[indices, :].cpu().numpy()
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                if self.simulator == "Genesis":
                    robot.set_dofs_position(qpos)
                elif self.simulator == "sapien":
                    robot.set_qpos(qpos)
                    
                left_robot_vertices.append(robot.get_verts())
                
        left_robot_vertices = torch.stack(left_robot_vertices, dim=0)  # [num_frame, vertex, 3]
        right_robot_vertices = torch.stack(right_robot_vertices, dim=0)
                
        left_robot_faces = np.zeros((self.left_robots[0].n_faces, 3), dtype=int) 
        
        for geom in self.left_robots[0].geoms:
            left_robot_faces[geom.face_start - self.left_robots[0].face_start: geom.face_end - self.left_robots[0].face_start] = geom.init_faces
            
        right_robot_faces = np.zeros((self.right_robots[0].n_faces, 3), dtype=int)
        
        for geom in self.right_robots[0].geoms:
            right_robot_faces[geom.face_start - self.right_robots[0].face_start: geom.face_end - self.right_robots[0].face_start] = geom.init_faces
        
        
        left_dict = {
            'vertices': left_robot_vertices.cpu(),
            'faces': left_robot_faces,
        }
        right_dict = {
            'vertices': right_robot_vertices.cpu(),
            'faces': right_robot_faces,
        }
        
        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(self.slam_path)
        pred_cam = dict(np.load(self.slam_path, allow_pickle=True))

        focal = pred_cam["img_focal"].item()        # scalar, e.g. 600
        cx, cy = pred_cam["img_center"] 
        intrinsics = (focal, focal, cx, cy)

        # import pdb; pdb.set_trace()
        
        output_pth = os.path.join("test", f"vis")
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        
        for x in range(len(self.image) - 1):
            img = cv2.imread(self.image[x])

            # Render left hand
            img = render_hand_on_image(
                img,
                left_dict['vertices'][x],
                left_dict['faces'],
                intrinsics,
                R_w2c_sla_all[x].numpy(),
                t_w2c_sla_all[x].numpy()
            )

            # Optional: render right hand on the same frame
            img = render_hand_on_image(
                img,
                right_dict['vertices'][x],
                right_dict['faces'],
                intrinsics,
                R_w2c_sla_all[x].numpy(),
                t_w2c_sla_all[x].numpy()
            )

            cv2.imwrite(os.path.join(output_pth, f"video_{x}.png"), img)