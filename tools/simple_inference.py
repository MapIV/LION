import argparse
import glob
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d  # noqa: F401
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pypcd4 import PointCloud

from visual_utils import open3d_vis_utils as V

OPEN3D_FLAG = True


@dataclass
class NameSpace:
    input_dir: Path
    config: Path
    checkpoint: Path

class SimpleInference:
    def __init__(self, config: Path, checkpoint: Path) -> None:
        cfg_from_yaml_file(config, cfg)
        self.logger = common_utils.create_logger()
        self.dataset = DatasetTemplate(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            training=False,
            root_path=None,
            logger=self.logger,
        )
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.dataset)
        self.model.load_params_from_file(filename=checkpoint, logger=self.logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()

    def inference(self, points: np.ndarray) -> list[dict[str, torch.Tensor]]:
        """Inference single frame point cloud.

        Parameters
        ----------
        points : np.ndarray
            (N, 4) array. Each point is represented by (x, y, z, intensity).

        Returns
        -------
        pred_dicts : list[dict[str, torch.Tensor]]
            Prediction results. Each dict contains:
                - pred_boxes: (M, 9) array in (x, y, z, dx, dy, dz, yaw, pitch, roll) format.
                - pred_scores: (M,) array. Score of each box.
                - pred_labels: (M,) array. Label of each box.
        """
        data_dict = {
            "points": np.hstack((points, np.zeros((points.shape[0], 1)))),
            "frame_id": np.array([0], dtype=np.int32),
        }
        data_dict = self.dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        with torch.no_grad():
            pred_dicts, _ = self.model.forward(data_dict)

        return pred_dicts


class PointCloudLoader:
    def __init__(self, file_path: Path) -> None:
        self.AVAILABLE_EXTENSIONS = [".bin", ".npy", ".pcd"]

        self.file_path = file_path
        self.pointcloud_path_list = []
        if self.file_path.is_dir():
            for ext in self.AVAILABLE_EXTENSIONS:
                self.pointcloud_path_list.extend(glob.glob(str(self.file_path / f"*{ext}")))
            self.pointcloud_path_list = sorted(self.pointcloud_path_list)
        else:  # noqa: PLR5501
            if self.file_path.suffix.lower() in self.AVAILABLE_EXTENSIONS:
                self.pointcloud_path_list = [self.file_path]
            else:
                self.pointcloud_path_list = []

    def get_pointcloud(self) -> Generator[np.ndarray, None, None]:
        """Load point cloud from file.

        Returns
        -------
        points : np.ndarray
            (N, 4) array. Each point is represented by (x, y, z, intensity).
        """
        for pointcloud_path in self.pointcloud_path_list:
            file_extension = Path(pointcloud_path).suffix.lower()
            if file_extension == ".bin":
                point = self.load_bin(pointcloud_path)
            elif file_extension == ".npy":
                point = self.load_npy(pointcloud_path)
            elif file_extension == ".pcd":
                point = self.load_pcd(pointcloud_path)
            else:
                continue

            yield point

    @staticmethod
    def load_bin(bin_path: Path) -> np.ndarray:
        """Load point cloud from .bin file.
        
        Parameters
        ----------
        bin_path : Path
            Path to the .bin file. Each point is represented by (x, y, z, intensity, ring_index).

        Returns
        -------
        points : np.ndarray
            (N, 4) array. Each point is represented by (x, y, z, intensity).
        """
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)

        return points[:, :4]

    @staticmethod
    def load_npy(npy_path: Path) -> np.ndarray:
        """Load point cloud from .npy file.
        
        Parameters
        ----------
        npy_path : Path
            Path to the .npy file. Each point is represented by (x, y, z, intensity).

        Returns
        -------
        points : np.ndarray
            (N, 4) array. Each point is represented by (x, y, z, intensity).
        """
        points = np.load(npy_path)

        return points

    @staticmethod
    def load_pcd(pcd_path: Path) -> np.ndarray:
        """Load point cloud from .pcd file.
        
        Parameters
        ----------
        pcd_path : Path
            Path to the .pcd file. Each point includes (x, y, z, intensity).

        Returns
        -------
        points : np.ndarray
            (N, 4) array. Each point is represented by (x, y, z, intensity).
        """
        pc = PointCloud.from_path(pcd_path)
        points = pc.numpy(("x", "y", "z", "intensity"))

        return points


def parse_config() -> NameSpace:
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("input_dir", type=Path, help="specify the point cloud data file or directory")
    parser.add_argument("config", type=Path, help="specify the config for demo")
    parser.add_argument("checkpoint", type=Path, help="specify the pretrained model")

    args = parser.parse_args(namespace=NameSpace)

    return args


def main() -> None:
    args = parse_config()
    simple_inference = SimpleInference(config=args.config, checkpoint=args.checkpoint)
    pointcloud_loader = PointCloudLoader(file_path=args.input_dir)

    for points in pointcloud_loader.get_pointcloud():
        pred_dicts = simple_inference.inference(points)

        V.draw_scenes(
            points=points,
            ref_boxes=pred_dicts[0]["pred_boxes"],
            ref_scores=pred_dicts[0]["pred_scores"],
            ref_labels=pred_dicts[0]["pred_labels"],
            threshold=0.5,
        )


if __name__ == "__main__":
    main()
