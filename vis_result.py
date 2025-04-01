from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import tyro

from dataset import HaWoRDataset
from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
from robot_viewer import RobotViewer
from hand_viewer import HandViewer 

import argparse

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_

import genesis as gs

def viz_hand_object(robots: Optional[Tuple[RobotName]], args, fps: int):
    # for x in range(16, 21):
    dataset = HaWoRDataset(args)
    if robots is None:
        viewer = HandViewer(dataset, headless=True, simulator=args.simulator)
    else:
        viewer = RobotViewer(dataset, list(robots), headless=True, simulator=args.simulator)

    if args.viewer == 'sapien':
        viewer.render_dexycb_data(fps)
    else:
        viewer.render_viewer()


def main(
    robots: Optional[List[RobotName]] = None,
    fps: int = 20,
    img_focal: Optional[float] = None,
    video_path: str = 'example/video_0.mp4',
    input_type: str = 'file',
    checkpoint: str = './weights/hawor/checkpoints/hawor.ckpt',
    infiller_weight: str = './weights/hawor/checkpoints/infiller.pt',
    viewer: str = 'sapien',
    simulator: str = 'sapien',
):
    """
    Render the human and robot trajectories for grasping object inside DexYCB dataset.
    """
    class Args:
        pass
    args = Args()
    args.img_focal = img_focal
    args.video_path = video_path
    args.input_type = input_type
    args.checkpoint = checkpoint
    args.infiller_weight = infiller_weight
    args.viewer = viewer
    args.simulator = simulator

    if simulator == 'Genesis':
        gs.init(backend=gs.gpu)   

    robot_dir = "/home/winnie/dex-retargeting/assets/robots/hands"
    RetargetingConfig.set_default_urdf_dir(robot_dir)

    viz_hand_object(robots, args, fps)


if __name__ == "__main__":
    tyro.cli(main)