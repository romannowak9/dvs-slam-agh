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

        if len(batch) == 0:
            return image

        mask = self._valid_event_mask(batch)

        if not np.any(mask):
            return image

        for x, y, p in zip(
            batch.x[mask],
            batch.y[mask],
            batch.p[mask],
        ):
            if bool(p):
                image[int(y), int(x)] = POSITIVE_INTENSITY
            else:
                image[int(y), int(x)] = NEGATIVE_INTENSITY

        return image

    def _make_exponential_frame(
        self,
        batch: EventBatch,
        t_ref: float,
    ) -> np.ndarray:
        last_t = np.full(
            (self.height, self.width),
            -np.inf,
            dtype=np.float64,
        )

        last_p = np.zeros(
            (self.height, self.width),
            dtype=np.bool_,
        )

        mask = self._valid_event_mask(batch)

        for x, y, t, p in zip(
            batch.x[mask],
            batch.y[mask],
            batch.t[mask],
            batch.p[mask],
        ):
            last_t[int(y), int(x)] = float(t)
            last_p[int(y), int(x)] = bool(p)

        image = np.full(
            (self.height, self.width),
            float(BACKGROUND_INTENSITY),
            dtype=np.float64,
        )

        active = np.isfinite(last_t)

        if self.polarity_mode == PolarityMode.POSITIVE:
            active = active & last_p
        elif self.polarity_mode == PolarityMode.NEGATIVE:
            active = active & (~last_p)

        if not np.any(active):
            return image.astype(np.uint8)

        age = np.maximum(0.0, float(t_ref) - last_t[active])
        weight = np.exp(-age / self.tau)

        active_polarity = last_p[active]

        values = np.full(weight.shape, float(BACKGROUND_INTENSITY), dtype=np.float64)
        values[active_polarity] = BACKGROUND_INTENSITY + 128.0 * weight[active_polarity]
        values[~active_polarity] = BACKGROUND_INTENSITY - 127.0 * weight[~active_polarity]

        image[active] = values

        return np.clip(np.rint(image), 0, 255).astype(np.uint8)

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