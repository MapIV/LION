import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import numpy as np
import open3d  # noqa: F401
import torch
from output_manager import OutputManager
from pointcloud_loader import PointCloudLoader
from tqdm import tqdm

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

OPEN3D_FLAG = True


@dataclass
class NameSpace:
    input_dir: Path
    output_dir: Path
    config: Path
    checkpoint: Path
    mapping_json: Union[Path, None]
    score_threshold: float
    output_format: Literal["CSV", "ROS1", "ROS2", "VISUALIZE"]
    input_topic_name: Union[str, None]
    output_topic_name: Union[str, None]


class SimpleInference:
    def __init__(self, config: Path, checkpoint: Path, mapping_json: Union[Path, None]) -> None:
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

        # Load mapping from JSON file
        self.mapping = None
        if mapping_json is not None:
            mapping_to_class = json.load(open(mapping_json))
            self.mapping = {}
            for mapping, class_list in mapping_to_class.items():
                for class_name in class_list:
                    self.mapping[class_name] = int(mapping)

    def _map_labels(self, pred_dicts: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        if self.mapping is None:
            return pred_dicts

        device = pred_dicts[0]["pred_boxes"].device
        pred_boxes = torch.empty((0, 9), dtype=torch.float32, device=device)
        pred_scores = torch.empty((0,), dtype=torch.float32, device=device)
        pred_labels = torch.empty((0,), dtype=torch.int64, device=device)
        for pred_dict in pred_dicts:
            for box, score, label in zip(pred_dict["pred_boxes"], pred_dict["pred_scores"], pred_dict["pred_labels"]):
                class_name = self.dataset.class_names[label - 1]  # Assuming labels are 1-indexed
                if class_name in self.mapping:
                    pred_boxes = torch.cat((pred_boxes, box.unsqueeze(0)), dim=0)
                    pred_scores = torch.cat((pred_scores, score.unsqueeze(0)), dim=0)
                    pred_labels = torch.cat(
                        (pred_labels, torch.tensor(self.mapping[class_name], device=device).unsqueeze(0)), dim=0
                    )

        return [{"pred_boxes": pred_boxes, "pred_scores": pred_scores, "pred_labels": pred_labels}]

    def inference(self, points: np.ndarray) -> dict[str, torch.Tensor]:
        """Inference single frame point cloud.

        Parameters
        ----------
        points : np.ndarray
            (N, 4) array. Each point is represented by (x, y, z, intensity).

        Returns
        -------
        pred_dict : dict[str, torch.Tensor]
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
            pred_dicts = self._map_labels(pred_dicts)

        return pred_dicts[0]

    @staticmethod
    def filter(pred_dict: dict[str, torch.Tensor], score_threshold: float = 0.0) -> dict[str, torch.Tensor]:
        """Filter boxes with score threshold.

        Parameters
        ----------
        pred_dicts : list[dict[str, torch.Tensor]]
            Prediction results. Each dict contains:
                - pred_boxes: (M, 9) array in (x, y, z, dx, dy, dz, yaw, pitch, roll) format.
                - pred_scores: (M,) array. Score of each box.
                - pred_labels: (M,) array. Label of each box.
        score_threshold : float
            Score threshold.

        Returns
        -------
        filtered_pred_dicts : list[dict[str, torch.Tensor]]
            Filtered prediction results.
        """
        filtered_pred_dict = {}
        scores = pred_dict["pred_scores"].cpu().numpy()
        mask = scores >= score_threshold

        filtered_pred_dict = {
            "pred_boxes": pred_dict["pred_boxes"][mask],
            "pred_scores": pred_dict["pred_scores"][mask],
            "pred_labels": pred_dict["pred_labels"][mask],
        }

        return filtered_pred_dict


def parse_config() -> NameSpace:
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("input_dir", type=Path, help="specify the point cloud data file or directory")
    parser.add_argument("output_dir", type=Path, help="specify the output directory for results")
    parser.add_argument("config", type=Path, help="specify the config for demo")
    parser.add_argument("checkpoint", type=Path, help="specify the pretrained model")
    parser.add_argument("--mapping_json", type=Path, default=None, help="specify the class mapping file")
    parser.add_argument("--score_threshold", type=float, default=0.0, help="specify the score threshold")
    parser.add_argument(
        "--output_format",
        type=str,
        default="CSV",
        choices=["CSV", "ROS1", "ROS2", "VISUALIZE"],
        help="specify the output format",
    )
    parser.add_argument(
        "--input_topic_name",
        type=str,
        default=None,
        help="specify the topic names to read point clouds from when input_dir is a rosbag file",
    )
    parser.add_argument(
        "--output_topic_name",
        type=str,
        default=None,
        help="specify the topic name to write point clouds to when output_format is ROS1 or ROS2",
    )

    args = parser.parse_args(namespace=NameSpace)

    return args


def verify_args(args: NameSpace) -> None:
    if args.output_format == "ROS1":
        if args.output_dir.suffix != ".bag":
            raise ValueError("When output_format is 'ros1', output_dir must be a directory ending with .bag")

    if args.output_format in ("ROS1", "ROS2"):
        if not args.output_topic_name:
            raise ValueError("When output_format is 'ros1' or 'ros2', output_topic_name must be specified")

    if not 0 <= args.score_threshold <= 1:
        raise ValueError("score_threshold must be between 0 and 1")


def main() -> None:
    args = parse_config()
    verify_args(args)

    # pre-process
    simple_inference = SimpleInference(config=args.config, checkpoint=args.checkpoint, mapping_json=args.mapping_json)
    if args.output_format in ("ROS1", "ROS2"):
        from rosbag_writer import RosbagWriter

        rosbag_writer = RosbagWriter(args.output_format, args.output_dir)
        pointcloud_loader = PointCloudLoader(
            file_path=args.input_dir,
            use_topic_list=[args.input_topic_name] if args.input_topic_name else None,
            rosbag_writer=rosbag_writer,
        )
    else:
        pointcloud_loader = PointCloudLoader(
            file_path=args.input_dir, use_topic_list=[args.input_topic_name] if args.input_topic_name else None
        )

    # main-process
    for points, timestamp in tqdm(pointcloud_loader.get_pointcloud(), total=len(pointcloud_loader)):
        # Inference
        pred_dict = simple_inference.inference(points)
        result = simple_inference.filter(pred_dict, args.score_threshold)

        if args.output_format == "CSV":
            OutputManager.save_results_to_csv(result=result, timestamp=timestamp, output_dir=args.output_dir)
        elif args.output_format in ("ROS1", "ROS2"):
            OutputManager.save_results_to_rosbag(
                result=result,
                topic_name=args.output_topic_name,
                timestamp=timestamp,
                rosbag_writer=rosbag_writer,
                frame_id=pointcloud_loader.frame_id,
            )
        elif args.output_format == "VISUALIZE":
            OutputManager.visualize_results(points=points, result=result, score_threshold=args.score_threshold)

    # post-process
    if args.output_format in ("ROS1", "ROS2"):
        rosbag_writer.close()


if __name__ == "__main__":
    main()
