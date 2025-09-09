import glob
from collections.abc import Generator
from pathlib import Path
from typing import Union

import numpy as np
from pypcd4 import PointCloud
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.stores.ros2_foxy import sensor_msgs__msg__PointCloud2


class PointCloudLoader:
    def __init__(self, file_path: Path, use_topic_list: Union[list[str], None] = None) -> None:
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

    def get_pointcloud(self) -> Generator[list[np.ndarray, float], None, None]:
        """Load point cloud from file.

        Returns
        -------
        points : np.ndarray
            (N, 4) array. Each point is represented by (x, y, z, intensity).
        """
        cnt = 0
        if self.type == "file":
            for pointcloud_path in self.pointcloud_path_list:
                file_extension = Path(pointcloud_path).suffix.lower()
                if file_extension == ".bin":
                    point = self.load_bin(pointcloud_path)
                elif file_extension == ".npy":
                    point = self.load_npy(pointcloud_path)
                elif file_extension == ".pcd":
                    point = self.load_pcd(pointcloud_path)
                cnt += 1

                yield point, cnt

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
                    msg: sensor_msgs__msg__PointCloud2 = reader.deserialize(
                        rawdata, connection.msgtype
                    )

                    pc = PointCloud.from_msg(msg)
                    points = pc.numpy(("x", "y", "z", "intensity"))
                    yield points, timestamp

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
