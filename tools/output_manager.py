from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rosbag_writer import RosbagWriter
from visual_utils import open3d_vis_utils as V  # noqa: F401


class OutputManager:
    @staticmethod
    def save_results_to_csv(result: dict[str, torch.Tensor], timestamp: int, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract predictions from the first (and only) batch element
        pred_boxes = result["pred_boxes"].cpu().numpy()
        pred_scores = result["pred_scores"].cpu().numpy()
        pred_labels = result["pred_labels"].cpu().numpy()

        # Create DataFrame and save to CSV
        csv_result = np.hstack((pred_boxes, pred_scores[:, None]))
        df = pd.DataFrame(csv_result, columns=["x", "y", "z", "l", "w", "h", "yaw", "pitch", "roll", "score"])
        df.insert(0, "timestamp", np.full((len(pred_labels), 1), timestamp / 10**9))
        df.insert(1, "class", pred_labels)
        df = df.sort_values(by="score", ascending=False)
        df.to_csv(output_dir / f"{timestamp}.csv", index=False, float_format="%.9f")
    
    @staticmethod
    def save_results_to_rosbag(result: dict[str, torch.Tensor], topic_name: str, timestamp: int, rosbag_writer: RosbagWriter, frame_id: str) -> None:
        rosbag_writer.add_connections(topic_name)
        rosbag_writer.write(result, topic_name, timestamp, frame_id=frame_id)

    @staticmethod
    def visualize_results(
        points: np.ndarray, result: dict[str, torch.Tensor], score_threshold: float
    ) -> None:
        V.draw_scenes(
            points=points,
            ref_boxes=result["pred_boxes"],
            ref_scores=result["pred_scores"],
            ref_labels=result["pred_labels"],
            threshold=score_threshold,
        )