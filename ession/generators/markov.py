from __future__ import annotations

import numpy as np

from ession.audio import Audio
from ession.note import Note
from ession.oscillator import Oscillator


class MarkovChain:
    """Generate note sequences via Markov chain transitions.

    Parameters
    ----------
    states:
        List of pitches (Hz or note names) that form
        the state space. Use 0 or None for a rest state.
    transition_matrix:
        Square matrix of transition probabilities.
        Row *i* gives the probabilities of moving from
        ``states[i]`` to each other state. Rows are
        normalized automatically.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        states: list[float | str | None],
        transition_matrix: np.ndarray | list[list[float]],
        seed: int | None = None,
    ) -> None:
        self.states = list(states)
        self.matrix = np.asarray(
            transition_matrix, dtype=np.float64
        )
        n = len(self.states)
        if self.matrix.shape != (n, n):
            raise ValueError(
                f"Matrix shape {self.matrix.shape} "
                f"doesn't match {n} states"
            )
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.matrix = self.matrix / row_sums
        self.rng = np.random.default_rng(seed)

    @classmethod
    def uniform(
        cls,
        states: list[float | str | None],
        seed: int | None = None,
    ) -> MarkovChain:
        """All transitions equally likely."""
        n = len(states)
        matrix = np.ones((n, n), dtype=np.float64)
        return cls(states, matrix, seed)

    @classmethod
    def from_sequence(
        cls,
        states: list[float | str | None],
        observed: list[float | str | None],
        seed: int | None = None,
    ) -> MarkovChain:
        """Learn transitions from an observed sequence."""
        idx = {s: i for i, s in enumerate(states)}
        n = len(states)
        matrix = np.zeros((n, n), dtype=np.float64)
        for a, b in zip(observed[:-1], observed[1:]):
            if a in idx and b in idx:
                matrix[idx[a], idx[b]] += 1.0
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums
        return cls(states, matrix, seed)

    def walk(
        self,
        n_steps: int,
        start: int | None = None,
    ) -> list[float | str | None]:
        """Generate a sequence of states.

        Parameters
        ----------
        n_steps:
            Number of states to generate.
        start:
            Index of the starting state. Random if None.
        """
        if start is None:
            idx = self.rng.integers(len(self.states))
        else:
            idx = start
        result = []
        for _ in range(n_steps):
            result.append(self.states[idx])
            idx = self.rng.choice(
                len(self.states), p=self.matrix[idx]
            )
        return result

    def generate_notes(
        self,
        n_steps: int,
        duration: float = 1.0,
        amplitude: float = 1.0,
        start: int | None = None,
    ) -> list[Note]:
        """Generate a list of Notes from a random walk."""
        pitches = self.walk(n_steps, start)
        return [
            Note(
                pitch=p,
                duration=duration,
                amplitude=amplitude,
            )
            for p in pitches
        ]

    def generate_audio(
        self,
        n_steps: int,
        duration: float = 1.0,
        amplitude: float = 1.0,
        tempo: float = 120.0,
        oscillator: Oscillator | None = None,
        sample_rate: int = Audio.DEFAULT_SAMPLE_RATE,
        start: int | None = None,
    ) -> Audio:
        """Render a Markov walk directly to Audio."""
        from ession.sequence import Sequence

        notes = self.generate_notes(
            n_steps, duration, amplitude, start
        )
        seq = Sequence(
            notes=notes,
            tempo=tempo,
            oscillator=oscillator or Oscillator(),
        )
        return seq.render(sample_rate=sample_rate)

    def __repr__(self) -> str:
        n = len(self.states)
        return f"MarkovChain({n} states)"
