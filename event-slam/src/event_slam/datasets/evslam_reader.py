from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from event_slam.core.imu import ImuData, prepare_imu_gyro_samples
from event_slam.core.types import CameraId, EventBatch

import rosbag


DEFAULT_LEFT_EVENT_TOPIC = "/dvxplorer_left/events"
DEFAULT_RIGHT_EVENT_TOPIC = "/dvxplorer_right/events"
DEFAULT_LEFT_IMU_TOPIC = "/dvxplorer_left/imu"


@dataclass
class ImuSample:
    """
    A single IMU measurement converted from sensor_msgs/Imu.

    The timestamp is stored in seconds as float64.
    Quaternion order is [qx, qy, qz, qw].
    """

    timestamp: float
    topic: str
    frame_id: str | None

    orientation_xyzw: np.ndarray
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray


class EvSlamRosbagReader:
    """
    Offline streaming reader for EvSLAM ROS bag files.

    This class is intentionally limited to I/O and message conversion.
    It does not synchronize left/right cameras, does not aggregate events into
    frames, and does not estimate motion.

    Each generator opens the bag, iterates over selected messages, converts them
    to project-level data structures, and yields them one-by-one. This prevents
    loading the whole sequence into RAM.
    """

    def __init__(
        self,
        bag_path: str | Path,
        left_event_topic: str = DEFAULT_LEFT_EVENT_TOPIC,
        right_event_topic: str = DEFAULT_RIGHT_EVENT_TOPIC,
        imu_topic: str = DEFAULT_LEFT_IMU_TOPIC,
    ) -> None:
        self.bag_path = Path(bag_path)

        self.left_event_topic = left_event_topic
        self.right_event_topic = right_event_topic
        self.imu_topic = imu_topic

        if not self.bag_path.exists():
            raise FileNotFoundError(f"ROS bag file does not exist: {self.bag_path}")

    def iter_messages(
        self,
        topics=None,
        start_time: float | None = None,
        end_time: float | None = None,
    ):
        """
        Iterate over raw ROS messages from the bag.

        Parameters:
        topics:
            None, one topic string, or an iterable of topic strings.
        start_time:
            Optional start time in seconds.
        end_time:
            Optional end time in seconds.

        Yields:
        tuple
            (topic, msg, bag_timestamp)
        """

        normalized_topics = _normalize_topics(topics)
        ros_start_time = _seconds_to_ros_time(start_time)
        ros_end_time = _seconds_to_ros_time(end_time)

        with rosbag.Bag(str(self.bag_path), "r") as bag:
            for topic, msg, bag_timestamp in bag.read_messages(
                topics=normalized_topics,
                start_time=ros_start_time,
                end_time=ros_end_time,
            ):
                yield topic, msg, bag_timestamp

    def iter_event_batches(
        self,
        camera: CameraId | str,
        start_time: float | None = None,
        end_time: float | None = None,
        max_batches: int | None = None,
        include_empty_batches: bool = False,
    ):
        """
        Iterate over EventBatch objects for one selected camera.

        This generator reads only one event topic and converts each ROS
        dvs_msgs/EventArray message to an EventBatch.
        """
        camera = CameraId(camera)
        topic = self._event_topic_from_camera(camera)

        yielded_batches = 0

        for _, msg, _ in self.iter_messages(
            topics=topic,
            start_time=start_time,
            end_time=end_time,
        ):
            batch = _event_array_to_batch(msg, camera)

            if len(batch) > 0 or include_empty_batches:
                yield batch
                yielded_batches += 1

            if max_batches is not None and yielded_batches >= max_batches:
                break

    def iter_left_event_batches(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        max_batches: int | None = None,
        include_empty_batches: bool = False,
    ):
        """
        Iterate over left-camera EventBatch objects.
        """
        yield from self.iter_event_batches(
            camera=CameraId.LEFT,
            start_time=start_time,
            end_time=end_time,
            max_batches=max_batches,
            include_empty_batches=include_empty_batches,
        )

    def iter_right_event_batches(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        max_batches: int | None = None,
        include_empty_batches: bool = False,
    ):
        """
        Iterate over right-camera EventBatch objects.
        """
        yield from self.iter_event_batches(
            camera=CameraId.RIGHT,
            start_time=start_time,
            end_time=end_time,
            max_batches=max_batches,
            include_empty_batches=include_empty_batches,
        )

    def iter_imu_samples(
        self,
        topic: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        max_samples: int | None = None,
    ):
        """
        Iterate over IMU measurements.

        The stream is used for gyro integration and rotational event
        motion compensation.
        """
        imu_topic = self.imu_topic if topic is None else topic

        yielded_samples = 0

        for topic_name, msg, _ in self.iter_messages(
            topics=imu_topic,
            start_time=start_time,
            end_time=end_time,
        ):
            yield _imu_msg_to_sample(msg, topic_name)
            yielded_samples += 1

            if max_samples is not None and yielded_samples >= max_samples:
                break

    def load_imu_gyro(self, topic: str | None = None) -> tuple:
        """
        Load, sort and deduplicate gyroscope samples from one IMU topic.
        """
        imu_topic = self.imu_topic if topic is None else topic
        timestamps = []
        angular_velocities = []

        for sample in self.iter_imu_samples(topic=imu_topic):
            timestamps.append(sample.timestamp)
            angular_velocities.append(sample.angular_velocity)

        if len(timestamps) < 2:
            raise ValueError(
                f"Need at least two IMU samples on topic {imu_topic}, "
                f"got {len(timestamps)}"
            )

        return prepare_imu_gyro_samples(
            timestamps=np.asarray(timestamps, dtype=np.float64),
            angular_velocities=np.asarray(angular_velocities, dtype=np.float64),
        )

    def load_imu(self, topic: str | None = None) -> ImuData:
        """Load the complete IMU stream once for visual-inertial estimation."""
        imu_topic = self.imu_topic if topic is None else topic
        samples = list(self.iter_imu_samples(topic=imu_topic))
        if len(samples) < 2:
            raise ValueError(
                f"Need at least two IMU samples on topic {imu_topic}, got {len(samples)}"
            )
        return ImuData(
            timestamps=np.asarray([sample.timestamp for sample in samples]),
            angular_velocities=np.asarray(
                [sample.angular_velocity for sample in samples]
            ),
            linear_accelerations=np.asarray(
                [sample.linear_acceleration for sample in samples]
            ),
        )

    def get_time_range(self) -> tuple[float, float]:
        """
        Return bag start and end time in seconds.

        This uses the bag index metadata and does not read all messages.
        """

        with rosbag.Bag(str(self.bag_path), "r") as bag:
            return float(bag.get_start_time()), float(bag.get_end_time())

    def get_duration(self) -> float:
        """
        Return bag duration in seconds.
        """
        start_time, end_time = self.get_time_range()
        return end_time - start_time

    def get_topics_summary(self) -> dict:
        """
        Return a lightweight summary of topics stored in the bag.

        The summary is read from bag metadata and does not load messages into RAM.
        """

        with rosbag.Bag(str(self.bag_path), "r") as bag:
            _, topics_info = bag.get_type_and_topic_info()

        summary = {}

        for topic_name, topic_info in topics_info.items():
            summary[topic_name] = {
                "msg_type": getattr(topic_info, "msg_type", None),
                "message_count": getattr(topic_info, "message_count", None),
                "connections": getattr(topic_info, "connections", None),
                "frequency": getattr(topic_info, "frequency", None),
            }

        return summary

    def _event_topic_from_camera(self, camera: CameraId) -> str:
        if camera == CameraId.LEFT:
            return self.left_event_topic

        if camera == CameraId.RIGHT:
            return self.right_event_topic

        raise ValueError(f"Unsupported camera id: {camera}")


def _event_array_to_batch(msg, camera: CameraId) -> EventBatch:
    if not hasattr(msg, "events"):
        raise ValueError(
            "Expected a dvs_msgs/EventArray-like message with an 'events' field"
        )

    events = msg.events
    event_count = len(events)

    t = np.empty(event_count, dtype=np.float64)
    x = np.empty(event_count, dtype=np.int32)
    y = np.empty(event_count, dtype=np.int32)
    p = np.empty(event_count, dtype=np.bool_)

    for index, event in enumerate(events):
        t[index] = _ros_time_to_seconds(event.ts)
        x[index] = int(event.x)
        y[index] = int(event.y)
        p[index] = bool(event.polarity)

    return EventBatch(
        t=t,
        x=x,
        y=y,
        p=p,
        camera=camera,
    )


def _imu_msg_to_sample(msg, topic: str) -> ImuSample:
    timestamp = _ros_time_to_seconds(msg.header.stamp)

    frame_id = None
    if hasattr(msg, "header") and hasattr(msg.header, "frame_id"):
        frame_id = str(msg.header.frame_id)

    orientation_xyzw = np.array(
        [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ],
        dtype=np.float64,
    )

    angular_velocity = np.array(
        [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ],
        dtype=np.float64,
    )

    linear_acceleration = np.array(
        [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ],
        dtype=np.float64,
    )

    return ImuSample(
        timestamp=timestamp,
        topic=topic,
        frame_id=frame_id,
        orientation_xyzw=orientation_xyzw,
        angular_velocity=angular_velocity,
        linear_acceleration=linear_acceleration,
    )


def _ros_time_to_seconds(ros_time) -> float:
    if hasattr(ros_time, "to_sec"):
        return float(ros_time.to_sec())

    if hasattr(ros_time, "secs") and hasattr(ros_time, "nsecs"):
        return float(ros_time.secs) + float(ros_time.nsecs) * 1e-9

    return float(ros_time)


def _seconds_to_ros_time(seconds: float | None):
    if seconds is None:
        return None

    try:
        import genpy
    except ImportError as exc:
        raise ImportError(
            "genpy is required to use start_time/end_time filters. "
            "Run this code inside a ROS Noetic environment."
        ) from exc

    return genpy.Time.from_sec(float(seconds))


def _normalize_topics(topics):
    if topics is None:
        return None

    if isinstance(topics, str):
        return [topics]

    return list(topics)
