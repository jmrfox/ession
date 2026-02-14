from __future__ import annotations

import numpy as np

from ession.audio import Audio


class BrownianMotion:
    """Brownian motion (Wiener process) for audio generation.

    Produces drifting, organic signals. Can be used
    directly as audio or as a control signal for
    synthesis parameters.

    Parameters
    ----------
    drift:
        Constant drift term (mu).
    volatility:
        Scale of random increments (sigma).
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.drift = drift
        self.volatility = volatility
        self.rng = np.random.default_rng(seed)

    def sample(
        self, n: int, dt: float = 1.0
    ) -> np.ndarray:
        """Generate *n* samples of the process.

        Parameters
        ----------
        n:
            Number of samples.
        dt:
            Time step between samples.
        """
        increments = (
            self.drift * dt
            + self.volatility
            * np.sqrt(dt)
            * self.rng.standard_normal(n)
        )
        return np.cumsum(increments)

    def to_audio(
        self,
        duration: float,
        sample_rate: int = Audio.DEFAULT_SAMPLE_RATE,
    ) -> Audio:
        """Render Brownian motion as mono audio.

        The signal is normalized to [-1, 1].
        """
        n = int(duration * sample_rate)
        dt = 1.0 / sample_rate
        samples = self.sample(n, dt)
        mx = np.max(np.abs(samples))
        if mx > 0:
            samples = samples / mx
        return Audio.from_array(samples, sample_rate)

    def to_control(
        self,
        n: int,
        low: float = 0.0,
        high: float = 1.0,
    ) -> np.ndarray:
        """Generate a control signal mapped to [low, high].

        Uses a sigmoid to keep values bounded.
        """
        raw = self.sample(n)
        sigmoid = 1.0 / (1.0 + np.exp(-raw))
        return low + (high - low) * sigmoid

    def __repr__(self) -> str:
        return (
            f"BrownianMotion(drift={self.drift}, "
            f"volatility={self.volatility})"
        )


class OrnsteinUhlenbeck:
    """Ornstein-Uhlenbeck process (mean-reverting).

    Unlike pure Brownian motion, this process is
    pulled back toward a mean value, producing
    bounded, stationary fluctuations.

    Parameters
    ----------
    theta:
        Mean reversion speed.
    mu:
        Long-term mean.
    sigma:
        Volatility.
    x0:
        Initial value.
    seed:
        Random seed.
    """

    def __init__(
        self,
        theta: float = 1.0,
        mu: float = 0.0,
        sigma: float = 0.3,
        x0: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0
        self.rng = np.random.default_rng(seed)

    def sample(
        self, n: int, dt: float = 1.0
    ) -> np.ndarray:
        """Generate *n* samples."""
        xs = np.empty(n, dtype=np.float64)
        x = self.x0
        noise = self.rng.standard_normal(n)
        sqrt_dt = np.sqrt(dt)
        for i in range(n):
            xs[i] = x
            x += (
                self.theta * (self.mu - x) * dt
                + self.sigma * sqrt_dt * noise[i]
            )
        return xs

    def to_audio(
        self,
        duration: float,
        sample_rate: int = Audio.DEFAULT_SAMPLE_RATE,
    ) -> Audio:
        """Render as normalized mono audio."""
        n = int(duration * sample_rate)
        dt = 1.0 / sample_rate
        samples = self.sample(n, dt)
        mx = np.max(np.abs(samples))
        if mx > 0:
            samples = samples / mx
        return Audio.from_array(samples, sample_rate)

    def to_control(
        self,
        n: int,
        dt: float = 1.0,
        low: float = 0.0,
        high: float = 1.0,
    ) -> np.ndarray:
        """Control signal mapped to [low, high]."""
        raw = self.sample(n, dt)
        mn, mx = raw.min(), raw.max()
        span = mx - mn
        if span > 0:
            normed = (raw - mn) / span
        else:
            normed = np.full_like(raw, 0.5)
        return low + (high - low) * normed

    def __repr__(self) -> str:
        return (
            f"OrnsteinUhlenbeck(theta={self.theta}, "
            f"mu={self.mu}, sigma={self.sigma})"
        )


class PoissonProcess:
    """Poisson process for triggering sound events.

    Generates event times at a given average rate.
    Useful for scattering grains, clicks, or tones
    at random intervals.

    Parameters
    ----------
    rate:
        Average events per second (lambda).
    seed:
        Random seed.
    """

    def __init__(
        self,
        rate: float = 10.0,
        seed: int | None = None,
    ) -> None:
        self.rate = rate
        self.rng = np.random.default_rng(seed)

    def event_times(
        self, duration: float
    ) -> np.ndarray:
        """Generate event times within [0, duration]."""
        n_expected = int(self.rate * duration * 2) + 10
        intervals = self.rng.exponential(
            1.0 / self.rate, size=n_expected
        )
        times = np.cumsum(intervals)
        return times[times < duration]

    def to_impulse_audio(
        self,
        duration: float,
        sample_rate: int = Audio.DEFAULT_SAMPLE_RATE,
        amplitude: float = 1.0,
    ) -> Audio:
        """Render as a click/impulse train."""
        n = int(duration * sample_rate)
        samples = np.zeros(n, dtype=np.float64)
        times = self.event_times(duration)
        indices = (times * sample_rate).astype(int)
        indices = indices[indices < n]
        samples[indices] = amplitude
        return Audio.from_array(samples, sample_rate)

    def scatter_sound(
        self,
        sound: Audio,
        duration: float,
        amplitude_range: tuple[float, float] = (
            0.5, 1.0
        ),
    ) -> Audio:
        """Scatter copies of a sound at Poisson times.

        Parameters
        ----------
        sound:
            The Audio clip to scatter.
        duration:
            Total output duration in seconds.
        amplitude_range:
            (min, max) amplitude for each event.
        """
        sr = sound.sample_rate
        n = int(duration * sr)
        out = np.zeros(
            (n, sound.channels), dtype=np.float64
        )
        times = self.event_times(duration)
        lo, hi = amplitude_range
        for t in times:
            idx = int(t * sr)
            end = min(idx + len(sound), n)
            length = end - idx
            amp = self.rng.uniform(lo, hi)
            out[idx:end] += (
                sound.data[:length] * amp
            )
        return Audio(out, sr)

    def __repr__(self) -> str:
        return f"PoissonProcess(rate={self.rate})"
