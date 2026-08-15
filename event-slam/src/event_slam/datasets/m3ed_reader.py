from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import h5py
import numpy as np

from event_slam.core.imu import prepare_imu_gyro_samples
from event_slam.core.types import CameraId, EventBatch, StereoEventWindow


HDF5_CACHE_BYTES = 64 * 1024 * 1024
MICROSECONDS_PER_SECOND = 1_000_000
SUPPORTED_VERSION_PREFIX = "v1.2"


@dataclass
class _EventStream:
    group: object
    ms_map_idx: np.ndarray
    event_count: int
    first_time_us: int
    last_time_us: int
    width: int
    height: int


class M3edH5Reader:
    """Stream synchronized stereo event windows directly from an M3ED H5 file."""

    def __init__(
        self,
        path,
        time_window: float,
        t_start: float = None,
        t_end: float = None,
        drop_empty_windows: bool = True,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"M3ED H5 file does not exist: {self.path}")

        self.time_window_us = _seconds_to_us(time_window, "time_window")
        if self.time_window_us <= 0:
            raise ValueError(f"time_window must be positive, got {time_window}")

        self.t_start_us = _optional_seconds_to_us(t_start, "t_start")
        self.t_end_us = _optional_seconds_to_us(t_end, "t_end")
        if (
            self.t_start_us is not None
            and self.t_end_us is not None
            and self.t_end_us <= self.t_start_us
        ):
            raise ValueError(f"t_end must be greater than t_start, got {t_start} -> {t_end}")

        self.time_window = self.time_window_us / MICROSECONDS_PER_SECOND
        self.drop_empty_windows = bool(drop_empty_windows)
        self.read_seconds = 0.0
        self.read_windows = 0

    def iter_windows(self):
        """Yield half-open stereo windows while keeping the H5 file open once."""
        started = time.perf_counter()
        h5_file = _open_h5(self.path)
        try:
            left, right = _load_event_streams(h5_file)
            run_start_us, run_end_us = self._common_time_range(left, right)
            left_start = _find_event_index(left, run_start_us)
            right_start = _find_event_index(right, run_start_us)
            window_start_us = run_start_us
            self.read_seconds += time.perf_counter() - started

            while window_start_us < run_end_us:
                started = time.perf_counter()
                window_end_us = min(
                    window_start_us + self.time_window_us,
                    run_end_us,
                )
                left_end = _find_event_index(left, window_end_us)
                right_end = _find_event_index(right, window_end_us)
                window = StereoEventWindow(
                    t_start=_us_to_seconds(window_start_us),
                    t_end=_us_to_seconds(window_end_us),
                    left=_read_event_batch(
                        left, left_start, left_end, CameraId.LEFT
                    ),
                    right=_read_event_batch(
                        right, right_start, right_end, CameraId.RIGHT
                    ),
                )

                left_start, right_start = left_end, right_end
                window_start_us = window_end_us
                self.read_seconds += time.perf_counter() - started
                if not (window.is_empty and self.drop_empty_windows):
                    self.read_windows += 1
                    yield window
        finally:
            h5_file.close()

    def load_imu_gyro(self) -> tuple:
        """Load the small M3ED OVC gyroscope stream in seconds and rad/s."""
        with _open_h5(self.path) as h5_file:
            _validate_version(h5_file)
            _require_paths(h5_file, ("ovc/imu/ts", "ovc/imu/omega"))
            timestamps = h5_file["ovc/imu/ts"]
            angular_velocities = h5_file["ovc/imu/omega"]

            if timestamps.ndim != 1 or timestamps.dtype != np.dtype(np.int64):
                raise ValueError("/ovc/imu/ts must be a one-dimensional int64 dataset")
            if angular_velocities.shape != (len(timestamps), 3):
                raise ValueError(
                    "/ovc/imu/omega must have shape "
                    f"({len(timestamps)}, 3), got {angular_velocities.shape}"
                )
            if not np.issubdtype(angular_velocities.dtype, np.floating):
                raise ValueError("/ovc/imu/omega must contain floating-point values")

            timestamps_us = np.asarray(timestamps[()], dtype=np.int64)
            omega = np.asarray(angular_velocities[()], dtype=np.float64)

        valid = (timestamps_us >= 0) & np.all(np.isfinite(omega), axis=1)
        timestamps_us = timestamps_us[valid]
        omega = omega[valid]
        if len(timestamps_us) < 2:
            raise ValueError("M3ED H5 contains fewer than two valid IMU samples")
        return prepare_imu_gyro_samples(
            timestamps_us.astype(np.float64) / MICROSECONDS_PER_SECOND,
            omega,
        )

    def get_time_range(self) -> tuple:
        """Return the configured common stereo interval in seconds."""
        with _open_h5(self.path) as h5_file:
            left, right = _load_event_streams(h5_file)
            start_us, end_us = self._common_time_range(left, right)
        return _us_to_seconds(start_us), _us_to_seconds(end_us)

    def _common_time_range(
        self,
        left: _EventStream,
        right: _EventStream,
    ) -> tuple:
        start_us = max(left.first_time_us, right.first_time_us)
        end_us = min(left.last_time_us, right.last_time_us)
        if self.t_start_us is not None:
            start_us = max(start_us, self.t_start_us)
        if self.t_end_us is not None:
            end_us = min(end_us, self.t_end_us)
        if end_us <= start_us:
            raise ValueError(
                "Requested interval does not overlap both M3ED cameras: "
                f"[{_us_to_seconds(start_us):.6f}, {_us_to_seconds(end_us):.6f}) s"
            )
        return start_us, end_us


def _open_h5(path: Path):
    return h5py.File(
        str(path),
        "r",
        rdcc_nbytes=HDF5_CACHE_BYTES,
        rdcc_nslots=10007,
        rdcc_w0=0.75,
    )


def _load_event_streams(h5_file) -> tuple:
    _validate_version(h5_file)
    left = _load_event_stream(h5_file, "left")
    right = _load_event_stream(h5_file, "right")
    if (left.width, left.height) != (right.width, right.height):
        raise ValueError(
            "M3ED stereo camera resolutions differ: "
            f"left={left.width}x{left.height}, right={right.width}x{right.height}"
        )
    return left, right


def _load_event_stream(h5_file, side: str) -> _EventStream:
    root = f"prophesee/{side}"
    paths = tuple(f"{root}/{name}" for name in ("x", "y", "p", "t", "ms_map_idx"))
    _require_paths(h5_file, paths + (f"{root}/calib/resolution",))
    group = h5_file[root]
    event_count = len(group["t"])

    expected_dtypes = {
        "x": np.dtype(np.uint16),
        "y": np.dtype(np.uint16),
        "p": np.dtype(np.int8),
        "t": np.dtype(np.int64),
    }
    for name, expected_dtype in expected_dtypes.items():
        dataset = group[name]
        if dataset.ndim != 1 or len(dataset) != event_count:
            raise ValueError(
                f"/{root}/{name} must be one-dimensional and contain "
                f"{event_count} values, got shape {dataset.shape}"
            )
        if dataset.dtype != expected_dtype:
            raise ValueError(
                f"/{root}/{name} must have dtype {expected_dtype}, got {dataset.dtype}"
            )
    if event_count == 0:
        raise ValueError(f"/{root} contains no events")

    ms_map_dataset = group["ms_map_idx"]
    if ms_map_dataset.ndim != 1 or not np.issubdtype(
        ms_map_dataset.dtype, np.unsignedinteger
    ):
        raise ValueError(f"/{root}/ms_map_idx must be a one-dimensional unsigned array")
    ms_map_idx = np.asarray(ms_map_dataset[()], dtype=np.int64)
    if len(ms_map_idx) == 0 or np.any(np.diff(ms_map_idx) < 0):
        raise ValueError(f"/{root}/ms_map_idx must be non-empty and non-decreasing")
    if ms_map_idx[0] < 0 or ms_map_idx[-1] >= event_count:
        raise ValueError(f"/{root}/ms_map_idx contains an out-of-range event index")

    resolution = np.asarray(group["calib/resolution"][()], dtype=np.int64).reshape(-1)
    if resolution.shape != (2,) or np.any(resolution <= 0):
        raise ValueError(f"/{root}/calib/resolution must contain [width, height]")
    width, height = int(resolution[0]), int(resolution[1])

    boundary = np.r_[0 : min(2, event_count), max(0, event_count - 2) : event_count]
    sample_x = np.asarray(group["x"][boundary])
    sample_y = np.asarray(group["y"][boundary])
    sample_p = np.asarray(group["p"][boundary])
    sample_t = np.asarray(group["t"][boundary], dtype=np.int64)
    if np.any(sample_x >= width) or np.any(sample_y >= height):
        raise ValueError(f"/{root} boundary event coordinates exceed {width}x{height}")
    if np.any((sample_p != 0) & (sample_p != 1)):
        raise ValueError(f"/{root}/p boundary samples must contain only 0 or 1")
    if sample_t[0] > sample_t[-1] or np.any(np.diff(sample_t[:2]) < 0) or np.any(
        np.diff(sample_t[-2:]) < 0
    ):
        raise ValueError(f"/{root}/t boundary samples are not non-decreasing")

    return _EventStream(
        group=group,
        ms_map_idx=ms_map_idx,
        event_count=event_count,
        first_time_us=int(sample_t[0]),
        last_time_us=int(sample_t[-1]),
        width=width,
        height=height,
    )


def _find_event_index(stream: _EventStream, timestamp_us: int) -> int:
    """Locate the first event at or after timestamp_us from a small time slice."""
    millisecond = max(0, int(timestamp_us) // 1000)
    map_size = len(stream.ms_map_idx)
    lower_key = max(0, min(map_size - 1, millisecond - 1))
    upper_key = millisecond + 2
    lower = 0 if millisecond == 0 else int(stream.ms_map_idx[lower_key])
    upper = (
        int(stream.ms_map_idx[upper_key])
        if upper_key < map_size
        else stream.event_count
    )
    times = np.asarray(stream.group["t"][lower:upper], dtype=np.int64)
    return lower + int(np.searchsorted(times, timestamp_us, side="left"))


def _read_event_batch(
    stream: _EventStream,
    start: int,
    end: int,
    camera: CameraId,
) -> EventBatch:
    if end <= start:
        return EventBatch.empty(camera)
    event_count = end - start
    source_slice = np.s_[start:end]
    timestamps = np.empty(event_count, dtype=np.float64)
    x = np.empty(event_count, dtype=np.int32)
    y = np.empty(event_count, dtype=np.int32)
    p = np.empty(event_count, dtype=np.bool_)
    stream.group["t"].read_direct(timestamps, source_sel=source_slice)
    stream.group["x"].read_direct(x, source_sel=source_slice)
    stream.group["y"].read_direct(y, source_sel=source_slice)
    stream.group["p"].read_direct(p, source_sel=source_slice)
    timestamps /= MICROSECONDS_PER_SECOND
    return EventBatch(
        t=timestamps,
        x=x,
        y=y,
        p=p,
        camera=camera,
    )


def _validate_version(h5_file) -> None:
    version = h5_file.attrs.get("version")
    if isinstance(version, bytes):
        version = version.decode("utf-8")
    if not isinstance(version, str) or not version.startswith(SUPPORTED_VERSION_PREFIX):
        raise ValueError(
            f"Unsupported M3ED version {version!r}; expected {SUPPORTED_VERSION_PREFIX}.x"
        )


def _require_paths(h5_file, paths: tuple) -> None:
    missing = [f"/{path}" for path in paths if path not in h5_file]
    if missing:
        raise KeyError("Missing required M3ED H5 paths: " + ", ".join(missing))


def _seconds_to_us(value: float, name: str) -> int:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number in seconds") from exc
    if not np.isfinite(seconds):
        raise ValueError(f"{name} must be a finite number in seconds")
    return int(round(seconds * MICROSECONDS_PER_SECOND))


def _optional_seconds_to_us(value, name: str):
    return None if value is None else _seconds_to_us(value, name)


def _us_to_seconds(timestamp_us: int) -> float:
    return int(timestamp_us) / MICROSECONDS_PER_SECOND
