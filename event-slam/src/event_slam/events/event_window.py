from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from event_slam.core.types import CameraId, EventBatch, StereoEventWindow


@dataclass
class StereoEventWindowStats:
    """
    Runtime statistics for StereoEventWindowBuilder.

    The counters are intentionally simple and are useful for debugging stream
    synchronization and memory behavior.
    """

    generated_windows: int = 0
    emitted_windows: int = 0
    empty_windows: int = 0
    dropped_empty_windows: int = 0

    left_events: int = 0
    right_events: int = 0

    first_window_start: float | None = None
    last_window_end: float | None = None


class StereoEventWindowBuilder:
    """
    Build fixed-duration stereo event windows from left/right EventBatch streams.

    This class does not load the whole sequence into RAM. It keeps only short
    per-camera buffers that contain events around the current window.

    Window convention:
        [window_start, window_end)

    This half-open interval avoids assigning an event lying exactly on a boundary
    to two neighboring windows.
    """

    def __init__(
        self,
        left_batches,
        right_batches,
        time_window: float,
        t_start: float | None = None,
        t_end: float | None = None,
        drop_empty_windows: bool = True,
    ) -> None:
        if time_window <= 0.0:
            raise ValueError(f"time_window must be positive, got {time_window}")

        if t_start is not None and t_end is not None and t_end <= t_start:
            raise ValueError(f"t_end must be greater than t_start, got {t_start} -> {t_end}")

        self.left_batches = iter(left_batches)
        self.right_batches = iter(right_batches)

        self.time_window = float(time_window)
        self.t_start = None if t_start is None else float(t_start)
        self.t_end = None if t_end is None else float(t_end)

        self.drop_empty_windows = bool(drop_empty_windows)

        self.left_buffer = _EventBatchBuffer(CameraId.LEFT)
        self.right_buffer = _EventBatchBuffer(CameraId.RIGHT)

        self.stats = StereoEventWindowStats()

        self._initialized = False
        self._finished = False
        self._current_window_start = None

    @classmethod
    def from_reader(
        cls,
        reader,
        time_window: float,
        t_start: float | None = None,
        t_end: float | None = None,
        drop_empty_windows: bool = True,
        read_margin: float | None = None,
    ) -> StereoEventWindowBuilder:
        """
        Create the builder directly from an EvSlamRosbagReader.

        The reader still owns bag access. This builder only consumes the left
        and right EventBatch generators.

        read_margin is used when t_start is provided. ROS bag filtering uses
        message timestamps, while event windows use per-event timestamps.
        Reading a small margin before t_start avoids accidentally skipping an
        EventArray message that contains events near the requested start time.
        """
        if read_margin is None:
            read_margin = time_window

        if read_margin < 0.0:
            raise ValueError(f"read_margin must be non-negative, got {read_margin}")

        read_start_time = None

        if t_start is not None:
            read_start_time = max(0.0, float(t_start) - float(read_margin))

        left_batches = reader.iter_left_event_batches(
            start_time=read_start_time,
            end_time=t_end,
            include_empty_batches=False,
        )

        right_batches = reader.iter_right_event_batches(
            start_time=read_start_time,
            end_time=t_end,
            include_empty_batches=False,
        )

        return cls(
            left_batches=left_batches,
            right_batches=right_batches,
            time_window=time_window,
            t_start=t_start,
            t_end=t_end,
            drop_empty_windows=drop_empty_windows,
        )

    def iter_windows(self):
        """
        Yield StereoEventWindow objects.

        The method is a generator. It consumes left/right streams progressively,
        fills buffers only as far as required for the current window, emits a
        window, and removes old events from the buffers.
        """
        if self._finished:
            return

        if not self._initialized:
            self._initialize()

        if self._current_window_start is None:
            self._finished = True
            return

        while True:
            window_start = self._current_window_start

            if self.t_end is not None and window_start >= self.t_end:
                self._finished = True
                break

            window_end = window_start + self.time_window

            if self.t_end is not None:
                window_end = min(window_end, self.t_end)

            left_has_coverage = self._fill_until(
                buffer=self.left_buffer,
                batches=self.left_batches,
                target_time=window_end,
            )

            right_has_coverage = self._fill_until(
                buffer=self.right_buffer,
                batches=self.right_batches,
                target_time=window_end,
            )

            if not left_has_coverage or not right_has_coverage:
                self._finished = True
                break

            left_window_batch = self.left_buffer.pop_window(window_start, window_end)
            right_window_batch = self.right_buffer.pop_window(window_start, window_end)

            window = StereoEventWindow(
                t_start=window_start,
                t_end=window_end,
                left=left_window_batch,
                right=right_window_batch,
            )

            self._update_stats(window)

            self._current_window_start = window_end

            if window.is_empty and self.drop_empty_windows:
                self.stats.dropped_empty_windows += 1
                continue

            self.stats.emitted_windows += 1
            yield window

    def _initialize(self) -> None:
        if self.t_start is not None:
            self._current_window_start = self.t_start
            self._initialized = True
            return

        left_has_data = self._fill_until_first_event(
            buffer=self.left_buffer,
            batches=self.left_batches,
        )

        right_has_data = self._fill_until_first_event(
            buffer=self.right_buffer,
            batches=self.right_batches,
        )

        if not left_has_data or not right_has_data:
            self._current_window_start = None
            self._initialized = True
            return

        first_left_time = self.left_buffer.first_time
        first_right_time = self.right_buffer.first_time

        if first_left_time is None or first_right_time is None:
            self._current_window_start = None
            self._initialized = True
            return

        self._current_window_start = max(first_left_time, first_right_time)

        self.left_buffer.drop_before(self._current_window_start)
        self.right_buffer.drop_before(self._current_window_start)

        self._initialized = True

    def _fill_until(self, buffer, batches, target_time: float) -> bool:
        """
        Fill one camera buffer until it has observed events reaching target_time.

        Returns True if the buffer is known to cover target_time, otherwise False.
        """
        while not buffer.has_reached(target_time):
            try:
                batch = next(batches)
            except StopIteration:
                buffer.mark_exhausted()
                break

            buffer.append(batch)

        return buffer.has_reached(target_time)

    def _fill_until_first_event(self, buffer, batches) -> bool:
        """
        Fill one camera buffer until at least one event is available.
        """
        while buffer.is_empty:
            try:
                batch = next(batches)
            except StopIteration:
                buffer.mark_exhausted()
                break

            buffer.append(batch)

        return not buffer.is_empty

    def _update_stats(self, window: StereoEventWindow) -> None:
        self.stats.generated_windows += 1

        if window.is_empty:
            self.stats.empty_windows += 1

        self.stats.left_events += len(window.left)
        self.stats.right_events += len(window.right)

        if self.stats.first_window_start is None:
            self.stats.first_window_start = window.t_start

        self.stats.last_window_end = window.t_end


class _EventBatchBuffer:
    """
    Small internal event buffer for one camera.

    The buffer stores only events that have not yet been assigned to a finished
    window. It is deliberately private because synchronization policy belongs to
    StereoEventWindowBuilder.
    """

    def __init__(self, camera: CameraId) -> None:
        self.camera = CameraId(camera)

        self.t = np.empty(0, dtype=np.float64)
        self.x = np.empty(0, dtype=np.int32)
        self.y = np.empty(0, dtype=np.int32)
        self.p = np.empty(0, dtype=np.bool_)

        self.exhausted = False
        self.last_seen_time = None

    @property
    def is_empty(self) -> bool:
        return len(self.t) == 0

    @property
    def first_time(self) -> float | None:
        if self.is_empty:
            return None

        return float(self.t[0])

    @property
    def last_time(self) -> float | None:
        if self.is_empty:
            return self.last_seen_time

        return float(self.t[-1])

    def append(self, batch: EventBatch) -> None:
        if batch.camera != self.camera:
            raise ValueError(
                f"Cannot append {batch.camera.value} batch to {self.camera.value} buffer"
            )

        if len(batch) == 0:
            return

        if self.last_seen_time is not None and batch.t[0] < self.last_seen_time:
            raise ValueError(
                f"Event timestamps must be non-decreasing for {self.camera.value}. "
                f"Last seen={self.last_seen_time}, new first={batch.t[0]}"
            )

        self.t = np.concatenate((self.t, batch.t))
        self.x = np.concatenate((self.x, batch.x))
        self.y = np.concatenate((self.y, batch.y))
        self.p = np.concatenate((self.p, batch.p))

        self.last_seen_time = float(batch.t[-1])

    def has_reached(self, timestamp: float) -> bool:
        if self.last_seen_time is None:
            return False

        return self.last_seen_time >= timestamp

    def pop_window(self, t_start: float, t_end: float) -> EventBatch:
        """
        Return events from [t_start, t_end) and remove all events older than t_end.
        """
        if self.is_empty:
            return EventBatch.empty(self.camera)

        in_window = (self.t >= t_start) & (self.t < t_end)

        batch = EventBatch(
            t=self.t[in_window],
            x=self.x[in_window],
            y=self.y[in_window],
            p=self.p[in_window],
            camera=self.camera,
        )

        keep = self.t >= t_end

        self.t = self.t[keep]
        self.x = self.x[keep]
        self.y = self.y[keep]
        self.p = self.p[keep]

        return batch

    def drop_before(self, timestamp: float) -> None:
        """
        Remove events older than timestamp.
        """
        if self.is_empty:
            return

        keep = self.t >= timestamp

        self.t = self.t[keep]
        self.x = self.x[keep]
        self.y = self.y[keep]
        self.p = self.p[keep]

    def mark_exhausted(self) -> None:
        self.exhausted = True