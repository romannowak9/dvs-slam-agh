from __future__ import annotations

from enum import Enum

import numpy as np

from event_slam.core.types import EventBatch, EventFrame, StereoEventFrame, StereoEventWindow


BACKGROUND_INTENSITY = 127
POSITIVE_INTENSITY = 255
NEGATIVE_INTENSITY = 0


class EventFrameMode(str, Enum):
    STANDARD = "standard"
    EXPONENTIAL = "exponential"


class PolarityMode(str, Enum):
    BOTH = "both"
    POSITIVE = "positive"
    NEGATIVE = "negative"


class EventFrameAggregator:
    """
    Convert event batches or stereo event windows into uint8 event frames.

    Intensity convention:
        127 -> background
        255 -> positive events
          0 -> negative events
    """

    def __init__(
        self,
        image_shape: tuple,
        mode: EventFrameMode | str = EventFrameMode.STANDARD,
        polarity_mode: PolarityMode | str = PolarityMode.BOTH,
        tau: float = 0.03,
    ) -> None:
        self.height = int(image_shape[0])
        self.width = int(image_shape[1])

        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"Invalid image_shape: {image_shape}")

        self.mode = EventFrameMode(mode)
        self.polarity_mode = PolarityMode(polarity_mode)
        self.tau = float(tau)

        if self.tau <= 0.0:
            raise ValueError(f"tau must be positive, got {self.tau}")

    def aggregate_batch(
        self,
        batch: EventBatch,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> EventFrame:
        """
        Aggregate one EventBatch into one EventFrame.
        """
        t_start, t_end = self._resolve_time_range(batch, t_start, t_end)

        if self.mode == EventFrameMode.STANDARD:
            image = self._make_standard_frame(batch)
        elif self.mode == EventFrameMode.EXPONENTIAL:
            image = self._make_exponential_frame(batch, t_ref=t_end)
        else:
            raise ValueError(f"Unsupported event frame mode: {self.mode}")

        return EventFrame(
            image=image,
            t_start=t_start,
            t_end=t_end,
            camera=batch.camera,
            frame_type=f"{self.mode.value}_{self.polarity_mode.value}",
        )

    def aggregate_stereo_window(self, window: StereoEventWindow) -> StereoEventFrame:
        """
        Aggregate a StereoEventWindow into left/right EventFrame objects.
        """
        left_frame = self.aggregate_batch(
            batch=window.left,
            t_start=window.t_start,
            t_end=window.t_end,
        )

        right_frame = self.aggregate_batch(
            batch=window.right,
            t_start=window.t_start,
            t_end=window.t_end,
        )

        return StereoEventFrame(
            left=left_frame,
            right=right_frame,
        )

    def _make_standard_frame(self, batch: EventBatch) -> np.ndarray:
        image = np.full(
            (self.height, self.width),
            BACKGROUND_INTENSITY,
            dtype=np.uint8,
        )

        mask = self._valid_event_mask(batch)

        if not np.any(mask):
            return image

        linear_idx, last_event_idx = self._last_event_per_pixel(batch, mask)

        p = batch.p[last_event_idx]

        image_flat = image.reshape(-1)

        positive_idx = linear_idx[p]
        negative_idx = linear_idx[~p]

        image_flat[positive_idx] = POSITIVE_INTENSITY
        image_flat[negative_idx] = NEGATIVE_INTENSITY

        return image

    def _make_exponential_frame(
        self,
        batch: EventBatch,
        t_ref: float,
    ) -> np.ndarray:
        mask = self._valid_event_mask(batch)

        image = np.full(
            (self.height, self.width),
            float(BACKGROUND_INTENSITY),
            dtype=np.float64,
        )

        if not np.any(mask):
            return image.astype(np.uint8)

        linear_idx, last_event_idx = self._last_event_per_pixel(batch, mask)

        t = batch.t[last_event_idx]
        p = batch.p[last_event_idx]

        age = np.maximum(0.0, float(t_ref) - t)
        weight = np.exp(-age / self.tau)

        values = np.full(weight.shape, float(BACKGROUND_INTENSITY), dtype=np.float64)
        values[p] = BACKGROUND_INTENSITY + 128.0 * weight[p]
        values[~p] = BACKGROUND_INTENSITY - 127.0 * weight[~p]

        image_flat = image.reshape(-1)
        image_flat[linear_idx] = values

        return np.clip(np.rint(image), 0, 255).astype(np.uint8)

    def _last_event_per_pixel(
        self,
        batch: EventBatch,
        mask: np.ndarray,
    ) -> tuple:
        event_indices = np.flatnonzero(mask)
        linear_idx = (
            batch.y[event_indices].astype(np.int64) * self.width
            + batch.x[event_indices]
        )
        last_event_idx = np.full(self.height * self.width, -1, dtype=np.int64)
        np.maximum.at(last_event_idx, linear_idx, event_indices)
        pixels = np.flatnonzero(last_event_idx >= 0)
        return pixels, last_event_idx[pixels]

    def _valid_event_mask(self, batch: EventBatch) -> np.ndarray:
        mask = (
            (batch.x >= 0)
            & (batch.x < self.width)
            & (batch.y >= 0)
            & (batch.y < self.height)
        )

        if self.polarity_mode == PolarityMode.POSITIVE:
            mask = mask & batch.p
        elif self.polarity_mode == PolarityMode.NEGATIVE:
            mask = mask & (~batch.p)

        return mask

    def _resolve_time_range(
        self,
        batch: EventBatch,
        t_start: float | None,
        t_end: float | None,
    ) -> tuple:
        if t_start is not None and t_end is not None:
            return float(t_start), float(t_end)

        batch_time_range = batch.time_range

        if batch_time_range is None:
            raise ValueError(
                "Cannot infer frame time range from an empty EventBatch. "
                "Pass t_start and t_end explicitly."
            )

        batch_t_start, batch_t_end = batch_time_range

        if t_start is None:
            t_start = batch_t_start

        if t_end is None:
            t_end = batch_t_end

        return float(t_start), float(t_end)
