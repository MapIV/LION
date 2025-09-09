import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import open3d  # noqa: F401
import pandas as pd
import torch
from pointcloud_loader import PointCloudLoader
from tqdm import tqdm
from visual_utils import open3d_vis_utils as V  # noqa: F401

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
    mapping_json: Union[Path, None] = None


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
                    pred_labels = torch.cat((pred_labels, torch.tensor(self.mapping[class_name], device=device).unsqueeze(0)), dim=0)

        return [{"pred_boxes": pred_boxes, "pred_scores": pred_scores, "pred_labels": pred_labels}]

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
            pred_dicts = self._map_labels(pred_dicts)

        return pred_dicts


def parse_config() -> NameSpace:
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("input_dir", type=Path, help="specify the point cloud data file or directory")
    parser.add_argument("output_dir", type=Path, help="specify the output directory for results")
    parser.add_argument("config", type=Path, help="specify the config for demo")
    parser.add_argument("checkpoint", type=Path, help="specify the pretrained model")
    parser.add_argument("--mapping_json", type=Path, default=None, help="specify the class mapping file")

    args = parser.parse_args(namespace=NameSpace)

    return args


def main() -> None:
    args = parse_config()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    simple_inference = SimpleInference(config=args.config, checkpoint=args.checkpoint, mapping_json=args.mapping_json)
    pointcloud_loader = PointCloudLoader(file_path=args.input_dir, use_topic_list=["/lidar0/pandar_packets"])

    for points, timestamp in tqdm(pointcloud_loader.get_pointcloud(), total=len(pointcloud_loader)):
        # Inference
        pred_dicts = simple_inference.inference(points)

        # Extract predictions from the first (and only) batch element
        pred_boxes = pred_dicts[0]["pred_boxes"].cpu().numpy()
        pred_scores = pred_dicts[0]["pred_scores"].cpu().numpy()
        pred_labels = pred_dicts[0]["pred_labels"].cpu().numpy()

        # Create DataFrame and save to CSV
        results = np.hstack((pred_boxes, pred_scores[:, None]))
        df = pd.DataFrame(results, columns=["x", "y", "z", "l", "w", "h", "yaw", "pitch", "roll", "score"])
        df.insert(0, "timestamp", np.full((len(pred_labels), 1), timestamp / 10**9))
        df.insert(1, "class", pred_labels)
        df = df.sort_values(by="score", ascending=False)
        df.to_csv(args.output_dir / f"{timestamp}.csv", index=False, float_format="%.9f")

        # V.draw_scenes(
        #     points=points,
        #     ref_boxes=pred_dicts[0]["pred_boxes"],
        #     ref_scores=pred_dicts[0]["pred_scores"],
        #     ref_labels=pred_dicts[0]["pred_labels"],
        #     threshold=0.5,
        # )


if __name__ == "__main__":
    main()
