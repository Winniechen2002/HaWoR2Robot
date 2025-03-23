import smplx
import torch
import numpy as np

from pathlib import Path
from typing import Dict, List, Optional

import cv2
from tqdm import trange
import sapien
from pytransform3d import transformations as pt
from sapien import internal_renderer as R
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer
from dataclasses import dataclass


def compute_smooth_shading_normal_np(vertices, indices):
    """
    Compute the vertex normal from vertices and triangles with numpy
    Args:
        vertices: (n, 3) to represent vertices position
        indices: (m, 3) to represent the triangles, should be in counter-clockwise order to compute normal outwards
    Returns:
        (n, 3) vertex normal

    References:
        https://www.iquilezles.org/www/articles/normals/normals.htm
    """
    # indices = indices.detach().cpu().numpy()
    v1 = vertices[indices[:, 0]]
    v2 = vertices[indices[:, 1]]
    v3 = vertices[indices[:, 2]]
    face_normal = np.cross(v2 - v1, v3 - v1)  # (n, 3) normal without normalization to 1

    vertex_normal = np.zeros_like(vertices)
    vertex_normal[indices[:, 0]] += face_normal
    vertex_normal[indices[:, 1]] += face_normal
    vertex_normal[indices[:, 2]] += face_normal
    vertex_normal /= np.linalg.norm(vertex_normal, axis=1, keepdims=True)
    return vertex_normal

class HandViewer:
    def __init__(self, data, headless=True, use_ray_tracing=False):
        if not use_ray_tracing:
            sapien.render.set_viewer_shader_dir("default")
            sapien.render.set_camera_shader_dir("default")
        else:
            sapien.render.set_viewer_shader_dir("rt")
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(64)
            sapien.render.set_ray_tracing_path_depth(8)
            sapien.render.set_ray_tracing_denoiser("oidn")

        # Scene
        scene = sapien.Scene()
        scene.set_timestep(1 / 240)

        # Lighting
        scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
        scene.add_directional_light(np.array([1, -1, -1]), np.array([2, 2, 2]), shadow=True)
        scene.add_directional_light([0, 0, -1], [1.8, 1.6, 1.6], shadow=False)
        scene.set_ambient_light(np.array([0.2, 0.2, 0.2]))

        # Add ground
        visual_material = sapien.render.RenderMaterial()
        visual_material.set_base_color(np.array([0.5, 0.5, 0.5, 1]))
        visual_material.set_roughness(0.7)
        visual_material.set_metallic(1)
        visual_material.set_specular(0.04)
        scene.add_ground(-1, render_material=visual_material)

        # Viewer
        if not headless:
            viewer = Viewer()
            viewer.set_scene(scene)
            viewer.set_camera_xyz(1.5, 0, 1)
            viewer.set_camera_rpy(0, -0.8, 3.14)
            viewer.control_window.toggle_origin_frame(False)
            self.viewer = viewer
        else:
            self.camera = scene.add_camera("cam", 1920, 640, 0.9, 0.01, 100)
            self.camera.set_local_pose(sapien.Pose([1.5, 0, 1], [0, 0.389418, 0, -0.921061]))

        self.headless = headless

        # Caches
        sapien.render.set_log_level("error")
        self.scene = scene
        self.internal_scene: R.Scene = scene.render_system._internal_scene
        self.context: R.Context = sapien.render.SapienRenderer()._internal_context
        self.mat_hand = self.context.create_material(np.zeros(4), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8, 0)

        self.objects: List[sapien.Entity] = []
        self.nodes: List[R.Node] = []
        
        self.data = data
        self.right_hand_vertices = data.right_hand_vertices
        self.left_hand_vertices = data.left_hand_vertices
        self.right_hand_joints = data.right_hand_joints
        self.left_hand_joints = data.left_hand_joints
        self.right_faces = data.right_faces
        self.left_faces = data.left_faces


    def _update_right_hand(self, vertex):
        try:
            vertex = vertex.detach().cpu().numpy()
        except:
            vertex = vertex
        normal = compute_smooth_shading_normal_np(vertex, self.right_faces)
        # import pdb; pdb.set_trace()
        mesh = self.context.create_mesh_from_array(vertex, self.right_faces, normal)
        model = self.context.create_model([mesh], [self.mat_hand])
        node = self.internal_scene.add_node()
        node.set_position(np.array([0, 0, 0]))
        obj = self.internal_scene.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = True
        obj.transparency = 0
        self.nodes.append(node)

    def _update_left_hand(self, vertex):
        self.clear_node()
        try:
            vertex = vertex.detach().cpu().numpy()
        except:
            vertex = vertex
        normal = compute_smooth_shading_normal_np(vertex, self.left_faces)
        mesh = self.context.create_mesh_from_array(vertex, self.left_faces, normal)
        model = self.context.create_model([mesh], [self.mat_hand])
        node = self.internal_scene.add_node()
        node.set_position(np.array([0, 0, 0]))
        obj = self.internal_scene.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = True
        obj.transparency = 0
        self.nodes.append(node)
        
        
    def clear_node(self):
        for _ in range(len(self.nodes)):
            node = self.nodes.pop()
            self.internal_scene.remove_node(node)
        
    def render_dexycb_data(self, fps=10):
        
        left_vertices = self.left_hand_vertices
        left_joints = self.left_hand_joints
        
        right_vertices = self.right_hand_vertices
        right_joints = self.right_hand_joints
        
        frame_num = left_vertices.shape[0]

        if self.headless:
            video_path = Path(__file__).parent.resolve() / "data/human_hand_video.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (self.camera.get_width(), self.camera.get_height())
            )

        step_per_frame = int(60 / fps)
        for i in trange(frame_num):
            left_vertex = left_vertices[i]
            left_joint = left_joints[i]
            right_vertex = right_vertices[i]
            right_joint = right_joints[i]
            
            self._update_left_hand(left_vertex)
            self._update_right_hand(right_vertex)
                
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