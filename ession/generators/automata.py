from __future__ import annotations

import numpy as np

from ession.audio import Audio


class CellularAutomaton:
    """1D elementary cellular automaton for audio generation.

    Parameters
    ----------
    rule:
        Wolfram rule number (0-255).
    width:
        Number of cells.
    seed:
        Random seed for initial state generation.
    """

    def __init__(
        self,
        rule: int = 30,
        width: int = 256,
        seed: int | None = None,
    ) -> None:
        if not 0 <= rule <= 255:
            raise ValueError(
                f"Rule must be 0-255, got {rule}"
            )
        self.rule = rule
        self.width = width
        self.rng = np.random.default_rng(seed)
        self._ruleset = np.array(
            [(rule >> i) & 1 for i in range(8)],
            dtype=np.uint8,
        )

    def _step(self, row: np.ndarray) -> np.ndarray:
        """Compute the next generation."""
        left = np.roll(row, 1)
        right = np.roll(row, -1)
        idx = (left << 2) | (row << 1) | right
        return self._ruleset[idx]

    def run(
        self,
        generations: int,
        initial: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run the CA and return the full grid.

        Returns shape ``(generations, width)`` of 0/1 values.
        """
        if initial is not None:
            row = np.asarray(initial, dtype=np.uint8)
        else:
            row = np.zeros(self.width, dtype=np.uint8)
            row[self.width // 2] = 1
        grid = np.zeros(
            (generations, self.width), dtype=np.uint8
        )
        grid[0] = row
        for g in range(1, generations):
            row = self._step(row)
            grid[g] = row
        return grid

    def random_initial(self, density: float = 0.5) -> np.ndarray:
        """Generate a random initial row."""
        return (
            self.rng.random(self.width) < density
        ).astype(np.uint8)

    # ── Audio generation ─────────────────────────────────────

    def to_waveform(
        self,
        generations: int,
        initial: np.ndarray | None = None,
        sample_rate: int = Audio.DEFAULT_SAMPLE_RATE,
    ) -> Audio:
        """Map CA grid directly to audio samples.

        Each row is flattened into the waveform
        sequentially. Cell values (0/1) are mapped
        to (-1, +1).
        """
        grid = self.run(generations, initial)
        samples = grid.flatten().astype(np.float64)
        samples = samples * 2.0 - 1.0
        return Audio.from_array(samples, sample_rate)

    def to_spectral(
        self,
        generations: int,
        initial: np.ndarray | None = None,
        sample_rate: int = Audio.DEFAULT_SAMPLE_RATE,
        hop_length: int = 512,
    ) -> Audio:
        """Treat the CA grid as a spectrogram and invert.

        Rows = time frames, columns = frequency bins.
        Uses ISTFT to convert back to audio.
        """
        from scipy.signal import istft

        grid = self.run(generations, initial)
        magnitude = grid.astype(np.float64)
        phase = self.rng.uniform(
            -np.pi, np.pi, size=magnitude.shape
        )
        S = magnitude * np.exp(1j * phase)
        _, samples = istft(
            S.T, fs=sample_rate, nperseg=self.width,
            noverlap=self.width - hop_length,
        )
        if np.max(np.abs(samples)) > 0:
            samples = samples / np.max(np.abs(samples))
        return Audio.from_array(samples, sample_rate)

    def to_note_triggers(
        self,
        generations: int,
        cell_index: int = 0,
        initial: np.ndarray | None = None,
    ) -> list[bool]:
        """Extract a trigger pattern from one cell.

        Returns a list of booleans: True when the
        cell is alive, False when dead. Useful as a
        rhythmic pattern.
        """
        grid = self.run(generations, initial)
        return [bool(v) for v in grid[:, cell_index]]

    def __repr__(self) -> str:
        return (
            f"CellularAutomaton(rule={self.rule}, "
            f"width={self.width})"
        )
