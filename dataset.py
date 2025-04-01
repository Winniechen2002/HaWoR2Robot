# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]
# Modified by Yuzhe Qin to use the sequential information inside the dataset

"""Egoallo dataset."""

from pathlib import Path

import numpy as np
import yaml

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


class HaWoRDataset:
    def __init__(self, args):
        start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

        frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

        slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        if not os.path.exists(slam_path):
            hawor_slam(args, start_idx, end_idx)
        slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(args, start_idx, end_idx, frame_chunks_all)

        # vis sequence for this video
        hand2idx = {
            "right": 1,
            "left": 0
        }
        vis_start = 0
        vis_end = pred_trans.shape[1] - 1
                
        # get faces
        faces = get_mano_faces()
        faces_new = np.array([[92, 38, 234],
                [234, 38, 239],
                [38, 122, 239],
                [239, 122, 279],
                [122, 118, 279],
                [279, 118, 215],
                [118, 117, 215],
                [215, 117, 214],
                [117, 119, 214],
                [214, 119, 121],
                [119, 120, 121],
                [121, 120, 78],
                [120, 108, 78],
                [78, 108, 79]])
        faces_right = np.concatenate([faces, faces_new], axis=0)

        # get right hand vertices
        hand = 'right'
        hand_idx = hand2idx[hand]
        pred_glob_r = run_mano(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
        right_verts = pred_glob_r['vertices'][0]
        right_dict = {
                'vertices': right_verts.unsqueeze(0),
                'joints': pred_glob_r['joints'][0].unsqueeze(0),
                'faces': faces_right,
            }

        # get left hand vertices
        faces_left = faces_right[:,[0,2,1]]
        hand = 'left'
        hand_idx = hand2idx[hand]
        pred_glob_l = run_mano_left(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
        left_verts = pred_glob_l['vertices'][0]
        left_dict = {
                'vertices': left_verts.unsqueeze(0),
                'joints': pred_glob_l['joints'][0].unsqueeze(0),
                'faces': faces_left,
            }

        # R_x = torch.tensor([[1,  0,  0],
        #                     [0, -1,  0],
        #                     [0,  0, -1]]).float()
        # R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
        # t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)
        # R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
        # t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
        # left_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, left_dict['vertices'].cpu())
        # right_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, right_dict['vertices'].cpu())
        # left_dict['joints'] = torch.einsum('ij,btnj->btni', R_x, left_dict['joints'].cpu())
        # right_dict['joints'] = torch.einsum('ij,btnj->btni', R_x, right_dict['joints'].cpu())
        
        self.slam_path = slam_path
        self.img_focal = img_focal
        self.imgfiles = imgfiles
        
        self.right_hand_vertices = right_dict['vertices'].squeeze(0)
        self.left_hand_vertices = left_dict['vertices'].squeeze(0)
        self.right_hand_joints = right_dict['joints'].squeeze(0)
        self.left_hand_joints = left_dict['joints'].squeeze(0)
        self.R_c2w_sla_all = R_c2w_sla_all
        self.t_c2w_sla_all = t_c2w_sla_all
        self.R_w2c_sla_all = R_w2c_sla_all
        self.t_w2c_sla_all = t_w2c_sla_all
        self.right_faces = faces_right
        self.left_faces = faces_left
