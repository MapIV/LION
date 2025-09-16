from pathlib import Path

import numpy as np
import torch
from rosbags.interfaces import Connection
from rosbags.rosbag1 import Writer as Writer1
from rosbags.rosbag2 import Writer as Writer2
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from scipy.spatial.transform import Rotation

from pointcloud_loader import PointCloudData


class RosbagWriter:
    def __init__(self, ros_type: str, output_path: Path) -> None:
        if ros_type == "ROS1":
            self.writer = Writer1(output_path)
            self.writer.open()
            self.typestore = get_typestore(Stores.ROS1_NOETIC)
            self.ros = ros_type
        elif ros_type == "ROS2":
            self.writer = Writer2(output_path)
            self.writer.open()
            self.typestore = get_typestore(Stores.LATEST)
            self.ros = ros_type

        self.load_custom_msgs(
            [
                "autoware_perception_msgs/msg/DetectedObjects.msg",
                "autoware_perception_msgs/msg/DetectedObject.msg",
                "autoware_perception_msgs/msg/DetectedObjectKinematics.msg",
                "autoware_perception_msgs/msg/ObjectClassification.msg",
                "autoware_perception_msgs/msg/Shape.msg",
            ]
        )

        self.DetectedObjects = self.typestore.types["autoware_perception_msgs/msg/DetectedObjects"]
        self.PointCloud2 = self.typestore.types["sensor_msgs/msg/PointCloud2"]
        self.msgtype = self.DetectedObjects.__msgtype__
        self.pointcloud_msgtype = self.PointCloud2.__msgtype__
        self.connection_dict: dict[str, Connection] = {}
        self.seq = -1

    def _get_seq(self) -> int:
        self.seq += 1
        return self.seq

    def load_custom_msgs(self, custom_msg_path_list: list[str]) -> None:
        def guess_msgtype(path: Path) -> str:
            """Guess message type name from path."""
            name = path.relative_to(path.parents[2]).with_suffix("")
            if "msg" not in name.parts:
                name = name.parent / "msg" / name.name
            return str(name)

        add_types = {}
        for msg_path in custom_msg_path_list:
            msg_path = Path(msg_path)
            msg_def = msg_path.read_text(encoding="utf-8")
            add_types.update(get_types_from_msg(msg_def, guess_msgtype(msg_path)))

        self.typestore.register(add_types)

    def add_connections(self, topic_name: str) -> None:
        if topic_name not in self.connection_dict:
            connection = self.writer.add_connection(topic_name, self.msgtype, typestore=self.typestore)

            self.connection_dict[topic_name] = connection

    def add_pointcloud_connection(self, topic_name: str) -> None:
        if topic_name not in self.connection_dict:
            connection = self.writer.add_connection(topic_name, self.pointcloud_msgtype, typestore=self.typestore)

            self.connection_dict[topic_name] = connection

    def write(
        self, result: dict[str, torch.Tensor], topic_name: str, timestamp: int, frame_id: str = "base_link"
    ) -> None:
        # writer setup
        Header = self.typestore.types["std_msgs/msg/Header"]
        Time = self.typestore.types["builtin_interfaces/msg/Time"]
        DetectedObject = self.typestore.types["autoware_perception_msgs/msg/DetectedObject"]
        DetectedObjectKinematics = self.typestore.types["autoware_perception_msgs/msg/DetectedObjectKinematics"]
        TwistWithCovariance = self.typestore.types["geometry_msgs/msg/TwistWithCovariance"]
        Twist = self.typestore.types["geometry_msgs/msg/Twist"]
        ObjectClassification = self.typestore.types["autoware_perception_msgs/msg/ObjectClassification"]
        Pose = self.typestore.types["geometry_msgs/msg/Pose"]
        Point = self.typestore.types["geometry_msgs/msg/Point"]
        Quaternion = self.typestore.types["geometry_msgs/msg/Quaternion"]
        PoseWithCovariance = self.typestore.types["geometry_msgs/msg/PoseWithCovariance"]
        Shape = self.typestore.types["autoware_perception_msgs/msg/Shape"]
        Polygon = self.typestore.types["geometry_msgs/msg/Polygon"]
        Vector3 = self.typestore.types["geometry_msgs/msg/Vector3"]

        pred_boxes = result["pred_boxes"].cpu().numpy()
        pred_scores = result["pred_scores"].cpu().numpy()
        pred_labels = result["pred_labels"].cpu().numpy()

        detected_object_list = []
        for pred_box, pred_score, pred_label in zip(pred_boxes, pred_scores, pred_labels):
            classification = ObjectClassification(label=pred_label, probability=pred_score)

            r = Rotation.from_euler("zyx", [pred_box[6], pred_box[7], pred_box[8]])
            q = r.as_quat()
            pose = Pose(
                position=Point(x=pred_box[0], y=pred_box[1], z=pred_box[2]),
                orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
            )
            kinematics = DetectedObjectKinematics(
                pose_with_covariance=PoseWithCovariance(pose=pose, covariance=np.array([0.0] * 36)),
                orientation_availability=DetectedObjectKinematics.UNAVAILABLE,
                has_position_covariance=False,
                twist_with_covariance=TwistWithCovariance(
                    twist=Twist(linear=Vector3(x=0.0, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=0.0)),
                    covariance=np.array([0.0] * 36),
                ),
                has_twist=False,
                has_twist_covariance=False,
            )

            shape = Shape(
                type=Shape.BOUNDING_BOX,
                footprint=Polygon(points=np.array([])),
                dimensions=Vector3(x=pred_box[3], y=pred_box[4], z=pred_box[5]),
            )

            detected_object_list.append(
                DetectedObject(
                    existence_probability=pred_score,
                    classification=[classification],
                    kinematics=kinematics,
                    shape=shape,
                )
            )

        if self.ros == "ROS1":
            message = self.DetectedObjects(
                header=Header(
                    stamp=Time(sec=int(timestamp // 10**9), nanosec=int(timestamp % 10**9)),
                    frame_id=frame_id,
                    seq=self._get_seq(),
                ),
                objects=detected_object_list,
            )

            self.writer.write(
                self.connection_dict[topic_name],
                timestamp,
                self.typestore.serialize_ros1(message, self.msgtype),
            )
        elif self.ros == "ROS2":
            message = self.DetectedObjects(
                header=Header(
                    stamp=Time(sec=int(timestamp // 10**9), nanosec=int(timestamp % 10**9)),
                    frame_id=frame_id,
                ),
                objects=detected_object_list,
            )

            self.writer.write(
                self.connection_dict[topic_name],
                timestamp,
                self.typestore.serialize_cdr(message, self.msgtype),
            )

    def write_pointcloud(self, pointcloud: PointCloudData) -> None:
        # writer setup
        Header = self.typestore.types["std_msgs/msg/Header"]
        Time = self.typestore.types["builtin_interfaces/msg/Time"]
        PointField = self.typestore.types["sensor_msgs/msg/PointField"]
        
        message = self.PointCloud2(
            header=Header(
                stamp=Time(sec=int(pointcloud.timestamp // 10**9), nanosec=int(pointcloud.timestamp % 10**9)),
                frame_id=pointcloud.frame_id,
                seq=self._get_seq(),
            ),
            height=1,
            width=pointcloud.data.shape[0],
            is_bigendian=False,
            point_step=16,
            row_step=pointcloud.data.shape[0] * 16,
            is_dense=True,
            data=pointcloud.data.reshape(-1).view(np.uint8),
            fields=[
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            ],
        )

        if self.ros == "ROS1":
            message = self.PointCloud2(
                header=Header(
                    stamp=Time(sec=int(pointcloud.timestamp // 10**9), nanosec=int(pointcloud.timestamp % 10**9)),
                    frame_id=pointcloud.frame_id,
                    seq=self._get_seq(),
                ),
                height=1,
                width=pointcloud.data.shape[0],
                is_bigendian=False,
                point_step=16,
                row_step=pointcloud.data.shape[0] * 16,
                is_dense=True,
                data=pointcloud.data.reshape(-1).view(np.uint8),
                fields=[
                    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
                ],
            )

            self.writer.write(
                self.connection_dict[pointcloud.topic_name],
                pointcloud.timestamp,
                self.typestore.serialize_ros1(message, self.pointcloud_msgtype),
            )
        elif self.ros == "ROS2":
            message = self.PointCloud2(
                header=Header(
                    stamp=Time(sec=int(pointcloud.timestamp // 10**9), nanosec=int(pointcloud.timestamp % 10**9)),
                    frame_id=pointcloud.frame_id,
                ),
                height=1,
                width=pointcloud.data.shape[0],
                is_bigendian=False,
                point_step=16,
                row_step=pointcloud.data.shape[0] * 16,
                is_dense=True,
                data=pointcloud.data.reshape(-1).view(np.uint8),
                fields=[
                    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
                ],
            )

            self.writer.write(
                self.connection_dict[pointcloud.topic_name],
                pointcloud.timestamp,
                self.typestore.serialize_cdr(message, self.pointcloud_msgtype),
            )

    def close(self) -> None:
        self.writer.close()
