from __future__ import annotations

import numpy as np

from ession.audio import Audio


class LogisticMap:
    """Logistic map: x_{n+1} = r * x_n * (1 - x_n).

    Produces sequences ranging from periodic to chaotic
    depending on the parameter *r* (interesting range
    roughly 2.5 to 4.0).

    Parameters
    ----------
    r:
        Growth rate parameter.
    x0:
        Initial value in (0, 1).
    """

    def __init__(
        self, r: float = 3.9, x0: float = 0.5
    ) -> None:
        self.r = r
        self.x0 = x0

    def iterate(self, n: int) -> np.ndarray:
        """Return *n* values of the logistic map."""
        xs = np.empty(n, dtype=np.float64)
        x = self.x0
        for i in range(n):
            xs[i] = x
            x = self.r * x * (1.0 - x)
        return xs

    def to_audio(
        self,
        n_samples: int,
        sample_rate: int = Audio.DEFAULT_SAMPLE_RATE,
    ) -> Audio:
        """Map the logistic sequence directly to audio.

        Values are rescaled from (0,1) to (-1,+1).
        """
        xs = self.iterate(n_samples)
        samples = xs * 2.0 - 1.0
        return Audio.from_array(samples, sample_rate)

    def to_control(self, n: int) -> np.ndarray:
        """Return raw (0,1) values for use as a control signal."""
        return self.iterate(n)

    def __repr__(self) -> str:
        return f"LogisticMap(r={self.r}, x0={self.x0})"


class LorenzAttractor:
    """Lorenz system integrated via RK4.

    The three state variables (x, y, z) can be mapped
    to audio channels, synthesis parameters, or
    spatial coordinates.

    Parameters
    ----------
    sigma, rho, beta:
        Standard Lorenz parameters.
        Defaults are the classic chaotic values.
    dt:
        Integration time step.
    state:
        Initial (x, y, z). Defaults to (1, 1, 1).
    """

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        dt: float = 0.01,
        state: tuple[float, float, float] = (
            1.0, 1.0, 1.0
        ),
    ) -> None:
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.state = np.array(state, dtype=np.float64)

    def _deriv(self, s: np.ndarray) -> np.ndarray:
        x, y, z = s
        return np.array([
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z,
        ])

    def _rk4_step(self, s: np.ndarray) -> np.ndarray:
        dt = self.dt
        k1 = self._deriv(s)
        k2 = self._deriv(s + 0.5 * dt * k1)
        k3 = self._deriv(s + 0.5 * dt * k2)
        k4 = self._deriv(s + dt * k3)
        return s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def integrate(self, n_steps: int) -> np.ndarray:
        """Return shape ``(n_steps, 3)`` trajectory."""
        traj = np.empty(
            (n_steps, 3), dtype=np.float64
        )
        s = self.state.copy()
        for i in range(n_steps):
            traj[i] = s
            s = self._rk4_step(s)
        self.state = s
        return traj

    def to_audio(
        self,
        duration: float,
        sample_rate: int = Audio.DEFAULT_SAMPLE_RATE,
        axis: int = 0,
    ) -> Audio:
        """Map one axis of the trajectory to mono audio.

        Parameters
        ----------
        duration:
            Length in seconds.
        axis:
            Which Lorenz variable to use (0=x, 1=y, 2=z).
        """
        n = int(duration * sample_rate)
        traj = self.integrate(n)
        samples = traj[:, axis]
        mx = np.max(np.abs(samples))
        if mx > 0:
            samples = samples / mx
        return Audio.from_array(samples, sample_rate)

    def to_stereo(
        self,
        duration: float,
        sample_rate: int = Audio.DEFAULT_SAMPLE_RATE,
        axes: tuple[int, int] = (0, 1),
    ) -> Audio:
        """Map two axes to stereo audio."""
        n = int(duration * sample_rate)
        traj = self.integrate(n)
        left = traj[:, axes[0]]
        right = traj[:, axes[1]]
        mx = max(
            np.max(np.abs(left)),
            np.max(np.abs(right)),
        )
        if mx > 0:
            left = left / mx
            right = right / mx
        stereo = np.column_stack([left, right])
        return Audio.from_array(stereo, sample_rate)

    def to_control(
        self, n_steps: int
    ) -> np.ndarray:
        """Return normalized (0,1) trajectory for control.

        Returns shape ``(n_steps, 3)``.
        """
        traj = self.integrate(n_steps)
        for i in range(3):
            col = traj[:, i]
            mn, mx = col.min(), col.max()
            span = mx - mn
            if span > 0:
                traj[:, i] = (col - mn) / span
        return traj

    def __repr__(self) -> str:
        return (
            f"LorenzAttractor(sigma={self.sigma}, "
            f"rho={self.rho}, beta={self.beta:.4f})"
        )
