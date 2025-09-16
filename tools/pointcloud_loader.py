import glob
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from pypcd4 import PointCloud
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.stores.ros2_foxy import sensor_msgs__msg__PointCloud2


@dataclass
class PointCloudData:
    data: np.ndarray
    timestamp: int
    frame_id: str = "base_link"
    topic_name: str = "/lidar/panda_packets"


class PointCloudLoader:
    def __init__(
        self,
        file_path: Path,
        use_topic_list: Union[list[str], None] = None,
    ) -> None:
        self.AVAILABLE_EXTENSIONS = [".bin", ".npy", ".pcd"]

        self.file_path = file_path
        self.type = "file"
        self.pointcloud_path_list = []
        self.use_topic_list = use_topic_list
        self.data_size = 0

        # Get list of point cloud files
        if self.file_path.is_dir():
            for ext in self.AVAILABLE_EXTENSIONS:
                self.pointcloud_path_list.extend(glob.glob(str(self.file_path / f"*{ext}")))
            self.pointcloud_path_list = sorted(self.pointcloud_path_list)
        else:  # noqa: PLR5501
            if self.file_path.suffix.lower() in self.AVAILABLE_EXTENSIONS:
                self.pointcloud_path_list = [self.file_path]
            else:
                self.pointcloud_path_list = []

        # Check if the file is a rosbag
        if len(self.pointcloud_path_list) == 0:
            try:
                with AnyReader([self.file_path]) as reader:
                    # Count messages
                    for x in reader.connections:
                        if x.msgtype == "sensor_msgs/msg/PointCloud2":
                            if self.use_topic_list is None or x.topic in self.use_topic_list:
                                self.data_size += x.msgcount

                    self.type = "rosbag"
            except Exception:
                self.pointcloud_path_list = []
        else:
            self.data_size = len(self.pointcloud_path_list)

    def get_pointcloud(self) -> Generator[PointCloudData, None, None]:
        """Load point cloud from file.

        Returns
        -------
        pointcloud : PointCloudData
            Pointcloud data. This contains:
                - data: (N, 4) array. Each point is represented by (x, y, z, intensity).
                - timestamp: int. Timestamp in nanoseconds.
                - pred_labels: str. Frame ID of the point cloud.
        """
        if self.type == "file":
            for pointcloud_path in self.pointcloud_path_list:
                pointcloud_path = Path(pointcloud_path)
                file_extension = pointcloud_path.suffix.lower()
                timestamp = pointcloud_path.stem.split("_")[-1]
                if "." in timestamp:
                    sec = int(timestamp.split(".")[0])
                    nsec = int(timestamp.split(".")[1].ljust(9, "0"))
                    timestamp = sec * 10**9 + nsec
                else:
                    timestamp = int(timestamp.ljust(19, "0"))

                if file_extension == ".bin":
                    point = self.load_bin(pointcloud_path)
                elif file_extension == ".npy":
                    point = self.load_npy(pointcloud_path)
                elif file_extension == ".pcd":
                    point = self.load_pcd(pointcloud_path)

                yield PointCloudData(data=point, timestamp=timestamp)

        elif self.type == "rosbag":
            typestore = get_typestore(Stores.ROS2_FOXY)
            with AnyReader([self.file_path], default_typestore=typestore) as reader:
                # Filter by topic name
                connections = []
                for x in reader.connections:
                    if x.msgtype == "sensor_msgs/msg/PointCloud2":
                        if self.use_topic_list is None or x.topic in self.use_topic_list:
                            connections.append(x)

                # Read messages
                for connection, timestamp, rawdata in reader.messages(connections=connections):
                    msg: sensor_msgs__msg__PointCloud2 = reader.deserialize(rawdata, connection.msgtype)

                    pc = PointCloud.from_msg(msg)
                    points = pc.numpy(("x", "y", "z", "intensity"))
                    timestamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec

                    yield PointCloudData(
                        data=points, timestamp=timestamp, frame_id=msg.header.frame_id, topic_name=connection.topic
                    )

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

    def __len__(self) -> int:
        return self.data_size
