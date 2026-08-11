from __future__ import annotations

import numpy as np

from event_slam.core.types import EventBatch, StereoEventWindow


EVENT_CHUNK_SIZE = 2_000
NO_EVENT = np.iinfo(np.int64).max


class BackgroundActivityFilter:
    """
    Simple nearest-neighbor Background Activity Filter.

    An event is kept only if there was at least min_neighbors recent event in
    its spatial neighborhood. The filter keeps an internal timestamp surface, so
    it can be applied continuously across consecutive event windows.
    """

    def __init__(
        self,
        image_shape: tuple,
        time_window: float = 1.0 / 24.0,
        radius: int = 2,
        min_neighbors: int = 1,
    ) -> None:
        self.height = int(image_shape[0])
        self.width = int(image_shape[1])
        self.time_window = float(time_window)
        self.radius = int(radius)
        self.min_neighbors = int(min_neighbors)

        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"Invalid image_shape: {image_shape}")

        if self.time_window <= 0.0:
            raise ValueError(f"time_window must be positive, got {self.time_window}")

        if self.radius < 1:
            raise ValueError(f"radius must be at least 1, got {self.radius}")

        if self.min_neighbors < 1:
            raise ValueError(f"min_neighbors must be at least 1, got {self.min_neighbors}")

        self.last_timestamp = np.full(
            (self.height, self.width),
            -np.inf,
            dtype=np.float64,
        )
        offsets = np.array(
            [
                (dx, dy)
                for dy in range(-self.radius, self.radius + 1)
                for dx in range(-self.radius, self.radius + 1)
                if dx != 0 or dy != 0
            ],
            dtype=np.int64,
        )
        self.neighbor_dx = offsets[:, 0]
        self.neighbor_dy = offsets[:, 1]
        self.first_event_by_pixel = np.full(
            self.height * self.width,
            NO_EVENT,
            dtype=np.int64,
        )

    def reset(self) -> None:
        self.last_timestamp.fill(-np.inf)
        self.first_event_by_pixel.fill(NO_EVENT)

    def filter(self, batch: EventBatch) -> EventBatch:
        """
        Filter one EventBatch and return a new EventBatch.

        Invalid out-of-image events are dropped.
        """
        if len(batch) == 0:
            return EventBatch.empty(batch.camera)

        inside = (
            (batch.x >= 0)
            & (batch.x < self.width)
            & (batch.y >= 0)
            & (batch.y < self.height)
        )
        event_indices = np.flatnonzero(inside)
        keep = np.zeros(len(batch), dtype=np.bool_)
        if len(event_indices) == 0:
            return EventBatch.empty(batch.camera)

        x = batch.x[event_indices].astype(np.int64, copy=False)
        y = batch.y[event_indices].astype(np.int64, copy=False)
        t = batch.t[event_indices]
        pixels = y.astype(np.int64) * self.width + x
        timestamp_surface = self.last_timestamp.reshape(-1)
        block_start = 0

        # Inside a block shorter than time_window, every earlier event is recent.
        # Recording its first index per pixel reproduces the sequential filter
        # while evaluating events in small NumPy chunks instead of a Python loop.
        while block_start < len(event_indices):
            block_end = np.searchsorted(
                t,
                t[block_start] + self.time_window,
                side="left",
            )
            block_events = event_indices[block_start:block_end]
            block_pixels = pixels[block_start:block_end]
            np.minimum.at(
                self.first_event_by_pixel,
                block_pixels,
                block_events,
            )

            for chunk_start in range(block_start, block_end, EVENT_CHUNK_SIZE):
                chunk_end = min(chunk_start + EVENT_CHUNK_SIZE, block_end)
                query_events = event_indices[chunk_start:chunk_end]
                neighbor_x = x[chunk_start:chunk_end, None] + self.neighbor_dx
                neighbor_y = y[chunk_start:chunk_end, None] + self.neighbor_dy
                neighbor_inside = (
                    (neighbor_x >= 0)
                    & (neighbor_x < self.width)
                    & (neighbor_y >= 0)
                    & (neighbor_y < self.height)
                )
                neighbor_pixels = np.where(
                    neighbor_inside,
                    neighbor_y * self.width + neighbor_x,
                    0,
                )
                recent = (
                    timestamp_surface[neighbor_pixels]
                    >= batch.t[query_events, None] - self.time_window
                ) | (
                    self.first_event_by_pixel[neighbor_pixels]
                    < query_events[:, None]
                )
                keep[query_events] = np.count_nonzero(
                    recent & neighbor_inside,
                    axis=1,
                ) >= self.min_neighbors

            np.maximum.at(
                timestamp_surface,
                block_pixels,
                batch.t[block_events],
            )
            self.first_event_by_pixel[block_pixels] = NO_EVENT
            block_start = block_end

        return EventBatch(
            t=batch.t[keep],
            x=batch.x[keep],
            y=batch.y[keep],
            p=batch.p[keep],
            camera=batch.camera,
        )


class StereoBackgroundActivityFilter:
    """Apply independent stateful BAF filters to a stereo event stream."""

    def __init__(
        self,
        image_shape: tuple,
        time_window: float,
        radius: int,
        min_neighbors: int,
    ) -> None:
        params = {
            "image_shape": image_shape,
            "time_window": time_window,
            "radius": radius,
            "min_neighbors": min_neighbors,
        }
        self.left = BackgroundActivityFilter(**params)
        self.right = BackgroundActivityFilter(**params)

    def reset(self) -> None:
        self.left.reset()
        self.right.reset()

    def filter(self, window: StereoEventWindow) -> StereoEventWindow:
        return StereoEventWindow(
            t_start=window.t_start,
            t_end=window.t_end,
            left=self.left.filter(window.left),
            right=self.right.filter(window.right),
        )
