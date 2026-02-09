from __future__ import annotations

from typing import Callable

import numpy as np

from ession.audio import Audio

Waveform = Callable[[np.ndarray], np.ndarray]


def sine(phase: np.ndarray) -> np.ndarray:
    """Sine waveform."""
    return np.sin(phase)


def square(phase: np.ndarray) -> np.ndarray:
    """Square waveform (band-unlimited)."""
    return np.sign(np.sin(phase))


def sawtooth(phase: np.ndarray) -> np.ndarray:
    """Sawtooth waveform (rising, band-unlimited)."""
    return 2.0 * (phase / (2.0 * np.pi) % 1.0) - 1.0


def triangle(phase: np.ndarray) -> np.ndarray:
    """Triangle waveform (band-unlimited)."""
    return 2.0 * np.abs(sawtooth(phase)) - 1.0


_BUILTIN: dict[str, Waveform] = {
    "sine": sine,
    "square": square,
    "sawtooth": sawtooth,
    "saw": sawtooth,
    "triangle": triangle,
    "tri": triangle,
}


class Oscillator:
    """Generates audio from a waveform function.

    Parameters
    ----------
    waveform:
        A waveform name ('sine', 'square', 'sawtooth',
        'triangle') or a callable ``(phase) -> samples``.
    attack:
        Attack time in seconds for click-free envelope.
    release:
        Release time in seconds for click-free envelope.
    """

    def __init__(
        self,
        waveform: str | Waveform = "sine",
        attack: float = 0.005,
        release: float = 0.005,
    ) -> None:
        if isinstance(waveform, str):
            key = waveform.lower()
            if key not in _BUILTIN:
                raise ValueError(
                    f"Unknown waveform {waveform!r}. "
                    f"Use one of {list(_BUILTIN)}"
                )
            self._fn: Waveform = _BUILTIN[key]
            self._name = key
        else:
            self._fn = waveform
            self._name = getattr(
                waveform, "__name__", "custom"
            )
        self.attack = attack
        self.release = release

    def generate(
        self,
        frequency: float,
        duration: float,
        amplitude: float = 1.0,
        sample_rate: int = Audio.DEFAULT_SAMPLE_RATE,
    ) -> Audio:
        """Render a tone as an Audio object.

        Parameters
        ----------
        frequency:
            Pitch in Hz. If 0, returns silence.
        duration:
            Length in seconds.
        amplitude:
            Peak amplitude, 0.0 to 1.0.
        sample_rate:
            Samples per second.
        """
        n = int(duration * sample_rate)
        if n == 0:
            return Audio.silence(0.0, sample_rate=sample_rate)
        if frequency == 0.0:
            return Audio.silence(
                duration, sample_rate=sample_rate
            )
        t = np.arange(n, dtype=np.float64) / sample_rate
        phase = 2.0 * np.pi * frequency * t
        samples = self._fn(phase) * amplitude
        samples = self._apply_envelope(samples, sample_rate)
        return Audio.from_array(samples, sample_rate)

    def _apply_envelope(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        n = len(samples)
        attack_n = min(
            int(self.attack * sample_rate), n // 2
        )
        release_n = min(
            int(self.release * sample_rate), n // 2
        )
        if attack_n > 0:
            ramp = np.linspace(0.0, 1.0, attack_n)
            samples[:attack_n] *= ramp
        if release_n > 0:
            ramp = np.linspace(1.0, 0.0, release_n)
            samples[-release_n:] *= ramp
        return samples

    def __repr__(self) -> str:
        return (
            f"Oscillator({self._name!r}, "
            f"attack={self.attack}, "
            f"release={self.release})"
        )
