from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from rosbag_writer import RosbagWriter
from pointcloud_loader import PointCloudData
from visual_utils import open3d_vis_utils as V  # noqa: F401


class OutputManager:
    @staticmethod
    def save_results_to_csv(result: dict[str, torch.Tensor], timestamp: int, output_dir: Path) -> None:
        """Save detection results to a CSV file.

        Parameters
        ----------
        result : dict[str, torch.Tensor]
            Inference results containing bounding boxes, scores, and labels.
        timestamp : int
            Timestamp in nanoseconds.
        output_dir : Path
            Directory to save the output CSV file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract predictions from the first (and only) batch element
        pred_boxes = result["pred_boxes"].cpu().numpy()
        pred_scores = result["pred_scores"].cpu().numpy()
        pred_labels = result["pred_labels"].cpu().numpy()

        # Convert Euler angles (yaw, pitch, roll) to quaternions
        positions = pred_boxes[:, :3]  # x, y, z positions
        dimensions = pred_boxes[:, 3:6]  # x, y, z dimensions
        euler_angles = pred_boxes[:, 6:9]  # yaw, pitch, roll
        euler_angles[:, 1:3] = 0  # pitch and roll are set to zero for accuracy
        if euler_angles.shape[0] != 0:
            rotations = Rotation.from_euler("zyx", euler_angles)
            quaternions = rotations.as_quat()  # Returns [x, y, z, w] format
        else:
            quaternions = np.empty((0, 4))

        # Create DataFrame and save to CSV
        csv_result = np.hstack((positions, dimensions, quaternions, pred_scores[:, None]))
        df = pd.DataFrame(
            csv_result,
            columns=[
                "x_position",
                "y_position",
                "z_position",
                "x_dimension",
                "y_dimension",
                "z_dimension",
                "quaternion_x",
                "quaternion_y",
                "quaternion_z",
                "quaternion_w",
                "confidence",
            ],
        )
        df.insert(0, "timestamp", np.full((len(pred_labels), 1), timestamp / 10**9))
        df.insert(1, "object_class", pred_labels)
        df = df.sort_values(by="confidence", ascending=False)
        df.to_csv(output_dir / f"{timestamp}.csv", index=False, float_format="%.9f")

    @staticmethod
    def save_results_to_rosbag(
        result: dict[str, torch.Tensor],
        pointcloud: PointCloudData,
        topic_name: str,
        rosbag_writer: RosbagWriter,
    ) -> None:
        """Save detection results to a ROS bag.

        Parameters
        ----------
        result : dict[str, torch.Tensor]
            Inference results containing bounding boxes, scores, and labels.
        pointcloud : PointCloudData
            Point cloud data to associate with the detections.
        topic_name : str
            Topic name for the detections.
        rosbag_writer : RosbagWriter
            ROS bag writer instance.
        """
        # Add pointcloud to rosbag
        rosbag_writer.add_pointcloud_connection(pointcloud.topic_name)
        rosbag_writer.write_pointcloud(pointcloud)

        # Add detections to rosbag
        rosbag_writer.add_connections(topic_name)
        rosbag_writer.write(result, topic_name, pointcloud.timestamp, frame_id=pointcloud.frame_id)

    @staticmethod
    def visualize_results(points: np.ndarray, result: dict[str, torch.Tensor]) -> None:
        """Visualize point cloud and detection results.

        Parameters
        ----------
        points : np.ndarray
            (N, 4) array. Each point is represented by (x, y, z, intensity).
        result : dict[str, torch.Tensor]
            Inference results containing bounding boxes, scores, and labels.
        """
        V.draw_scenes(
            points=points,
            ref_boxes=result["pred_boxes"],
            ref_scores=result["pred_scores"],
            ref_labels=result["pred_labels"],
        )
