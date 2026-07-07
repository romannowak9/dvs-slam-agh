from __future__ import annotations

import numpy as np

from event_slam.core.types import EventBatch


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

    def reset(self) -> None:
        self.last_timestamp.fill(-np.inf)

    def filter(self, batch: EventBatch) -> EventBatch:
        """
        Filter one EventBatch and return a new EventBatch.

        Invalid out-of-image events are dropped.
        """
        if len(batch) == 0:
            return EventBatch.empty(batch.camera)

        keep = np.zeros(len(batch), dtype=np.bool_)

        for index in range(len(batch)):
            x = int(batch.x[index])
            y = int(batch.y[index])
            t = float(batch.t[index])

            if not self._is_inside_image(x, y):
                continue

            if self._has_recent_neighbor(x, y, t):
                keep[index] = True

            self.last_timestamp[y, x] = t

        return EventBatch(
            t=batch.t[keep],
            x=batch.x[keep],
            y=batch.y[keep],
            p=batch.p[keep],
            camera=batch.camera,
        )

    def _has_recent_neighbor(self, x: int, y: int, timestamp: float) -> bool:
        x0 = max(0, x - self.radius)
        x1 = min(self.width, x + self.radius + 1)

        y0 = max(0, y - self.radius)
        y1 = min(self.height, y + self.radius + 1)

        neighborhood = self.last_timestamp[y0:y1, x0:x1]
        recent = neighborhood >= timestamp - self.time_window

        center_y = y - y0
        center_x = x - x0

        if 0 <= center_y < recent.shape[0] and 0 <= center_x < recent.shape[1]:
            recent = recent.copy()
            recent[center_y, center_x] = False

        return int(np.count_nonzero(recent)) >= self.min_neighbors

    def _is_inside_image(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height


def filter_background_activity(
    batch: EventBatch,
    image_shape: tuple,
    time_window: float = 1.0 / 24.0,
    radius: int = 2,
    min_neighbors: int = 1,
) -> EventBatch:
    """
    Convenience stateless BAF call for one batch.

    For continuous processing, prefer BackgroundActivityFilter because it keeps
    timestamp history across consecutive windows.
    """
    event_filter = BackgroundActivityFilter(
        image_shape=image_shape,
        time_window=time_window,
        radius=radius,
        min_neighbors=min_neighbors,
    )

    return event_filter.filter(batch)