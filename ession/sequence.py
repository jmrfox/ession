from __future__ import annotations

from ession.audio import Audio
from ession.note import Note
from ession.oscillator import Oscillator


class Sequence:
    """An ordered list of notes rendered at a given tempo.

    Parameters
    ----------
    notes:
        List of :class:`Note` objects.
    tempo:
        Beats per minute. Used to convert beat durations
        to seconds.
    oscillator:
        Waveform generator for rendering notes.
    use_seconds:
        If ``True``, note durations are treated as
        absolute seconds instead of beats.
    """

    def __init__(
        self,
        notes: list[Note],
        tempo: float = 120.0,
        oscillator: Oscillator | None = None,
        use_seconds: bool = False,
    ) -> None:
        self.notes = list(notes)
        self.tempo = tempo
        self.oscillator = oscillator or Oscillator()
        self.use_seconds = use_seconds

    def _beat_to_seconds(self, beats: float) -> float:
        return beats * 60.0 / self.tempo

    def _note_duration_s(self, note: Note) -> float:
        if self.use_seconds:
            return note.duration
        return self._beat_to_seconds(note.duration)

    def render(
        self,
        repeat: int = 1,
        sample_rate: int = Audio.DEFAULT_SAMPLE_RATE,
    ) -> Audio:
        """Render the sequence to an Audio object.

        Parameters
        ----------
        repeat:
            Number of times to repeat the sequence.
        sample_rate:
            Samples per second.
        """
        clips: list[Audio] = []
        for note in self.notes:
            dur_s = self._note_duration_s(note)
            clip = self.oscillator.generate(
                frequency=note.frequency,
                duration=dur_s,
                amplitude=note.amplitude,
                sample_rate=sample_rate,
            )
            clips.append(clip)
        if not clips:
            return Audio.silence(
                0.0, sample_rate=sample_rate
            )
        one_pass = Audio.concatenate(*clips)
        if repeat <= 1:
            return one_pass
        return Audio.concatenate(
            *[one_pass] * repeat
        )

    @property
    def duration_beats(self) -> float:
        """Total duration in beats."""
        return sum(n.duration for n in self.notes)

    @property
    def duration_seconds(self) -> float:
        """Total duration in seconds."""
        return sum(
            self._note_duration_s(n) for n in self.notes
        )

    def __repr__(self) -> str:
        n = len(self.notes)
        return (
            f"Sequence({n} notes, "
            f"{self.tempo} BPM, "
            f"{self.duration_seconds:.2f}s)"
        )
