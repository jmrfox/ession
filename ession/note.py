from __future__ import annotations

from dataclasses import dataclass

from ession.pitch import parse_pitch


@dataclass
class Note:
    """A single sound event in a sequence.

    Parameters
    ----------
    pitch:
        Frequency in Hz (float/int) or note name (str).
        Use 0 or ``None`` for a rest.
    duration:
        Length in beats (default) or seconds.
    amplitude:
        Volume from 0.0 to 1.0.
    """

    pitch: float | str | None = 440.0
    duration: float = 1.0
    amplitude: float = 1.0

    @property
    def frequency(self) -> float:
        """Resolved frequency in Hz (0.0 for rests)."""
        if self.pitch is None:
            return 0.0
        return parse_pitch(self.pitch)

    @property
    def is_rest(self) -> bool:
        return self.frequency == 0.0

    def __repr__(self) -> str:
        if self.is_rest:
            return f"Note(rest, dur={self.duration})"
        return (
            f"Note({self.pitch}, dur={self.duration}, "
            f"amp={self.amplitude})"
        )


def rest(duration: float = 1.0) -> Note:
    """Shorthand to create a rest of a given duration."""
    return Note(pitch=None, duration=duration)
