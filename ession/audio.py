from __future__ import annotations

from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


class Audio:
    """Base class for audio data."""

    DEFAULT_SAMPLE_RATE = 44100

    def __init__(
        self,
        data: np.ndarray,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        bit_depth: int = 16,
    ) -> None:
        if data.ndim == 1:
            data = data[:, np.newaxis]
        if data.ndim != 2 or data.shape[1] not in (1, 2):
            raise ValueError(f"Expected mono or stereo, got {data.shape}")
        self.data = data.astype(np.float64)
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth

    # ── Factory methods ──────────────────────────────────────────

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        bit_depth: int = 16,
    ) -> Self:
        return cls(data, sample_rate, bit_depth)

    @classmethod
    def silence(
        cls,
        duration: float,
        channels: int = 1,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        bit_depth: int = 16,
    ) -> Self:
        n_samples = int(duration * sample_rate)
        data = np.zeros((n_samples, channels), dtype=np.float64)
        return cls(data, sample_rate, bit_depth)

    # ── Properties ───────────────────────────────────────────────

    @property
    def duration(self) -> float:
        return len(self) / self.sample_rate

    @property
    def channels(self) -> int:
        return self.data.shape[1]

    @property
    def is_mono(self) -> bool:
        return self.channels == 1

    @property
    def is_stereo(self) -> bool:
        return self.channels == 2

    # ── I/O ──────────────────────────────────────────────────────

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        subtype_map = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}
        subtype = subtype_map.get(self.bit_depth)
        if subtype is None:
            raise ValueError(f"Bad bit_depth {self.bit_depth}; use 16/24/32")
        sf.write(str(path), self.data, self.sample_rate, subtype=subtype)
        return path

    # ── Manipulation ─────────────────────────────────────────────

    def normalize(self, peak: float = 1.0) -> Self:
        max_val = np.max(np.abs(self.data))
        if max_val == 0:
            return self
        self.data = self.data * (peak / max_val)
        return self

    def trim(self, start: float = 0.0, end: float | None = None) -> Audio:
        start_sample = int(start * self.sample_rate)
        if end is not None:
            end_sample = int(end * self.sample_rate)
        else:
            end_sample = len(self)
        return Audio(
            self.data[start_sample:end_sample].copy(),
            self.sample_rate,
            self.bit_depth,
        )

    @staticmethod
    def concatenate(*clips: Audio) -> Audio:
        if not clips:
            raise ValueError("Need at least one Audio to concatenate")
        sr = clips[0].sample_rate
        bd = clips[0].bit_depth
        ch = clips[0].channels
        for c in clips[1:]:
            if c.sample_rate != sr:
                raise ValueError("Sample rates must match")
            if c.channels != ch:
                raise ValueError("Channel counts must match")
        data = np.concatenate(
            [c.data for c in clips],
            axis=0,
        )
        return Audio(data, sr, bd)

    @staticmethod
    def mix(*clips: Audio, levels: list[float] | None = None) -> Audio:
        if not clips:
            raise ValueError("Need at least one Audio to mix")
        sr = clips[0].sample_rate
        bd = clips[0].bit_depth
        ch = clips[0].channels
        for c in clips[1:]:
            if c.sample_rate != sr:
                raise ValueError("Sample rates must match")
            if c.channels != ch:
                raise ValueError("Channel counts must match")
        if levels is None:
            levels = [1.0] * len(clips)
        max_len = max(len(c) for c in clips)
        mixed = np.zeros((max_len, ch), dtype=np.float64)
        for clip, level in zip(clips, levels):
            mixed[: len(clip)] += clip.data * level
        return Audio(mixed, sr, bd)

    # ── Visualization ────────────────────────────────────────────

    def plot(
        self,
        title: str = "Waveform",
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        t = np.arange(len(self)) / self.sample_rate
        show = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        else:
            fig = ax.get_figure()
        for ch in range(self.channels):
            if self.is_mono:
                label = "mono"
            else:
                label = "left" if ch == 0 else "right"
            ax.plot(t, self.data[:, ch], linewidth=0.5, label=label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.set_xlim(0, self.duration)
        ax.set_ylim(-1.05, 1.05)
        if self.is_stereo:
            ax.legend(loc="upper right")
        if show:
            plt.tight_layout()
            plt.show()
        return fig

    # ── Dunder methods ───────────────────────────────────────────

    def __len__(self) -> int:
        return self.data.shape[0]

    def __repr__(self) -> str:
        ch_str = "mono" if self.is_mono else "stereo"
        return (
            f"Audio({self.duration:.3f}s, {self.sample_rate}Hz, "
            f"{ch_str}, {self.bit_depth}-bit)"
        )
