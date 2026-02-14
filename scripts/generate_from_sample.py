"""Run generators using an audio file as source material.

Each generator produces a new audio file in an output
directory named after the input file.

Generators
----------
poisson     Scatter the input at random Poisson-distributed times.
brownian    Amplitude-modulate the input with Brownian motion.
lorenz      Stereo pan the input using a Lorenz attractor.
ca          Use a cellular automaton to gate the input on/off.
logistic    Modulate playback speed via the logistic map.
ou          Ornstein-Uhlenbeck filtered noise mixed with input.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from ession import (
    Audio,
    BrownianMotion,
    CellularAutomaton,
    LogisticMap,
    LorenzAttractor,
    OrnsteinUhlenbeck,
    PoissonProcess,
)

log = logging.getLogger(__name__)

ALL_GENERATORS = [
    "poisson",
    "brownian",
    "lorenz",
    "ca",
    "logistic",
    "ou",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the source audio file",
    )
    parser.add_argument(
        "--generators",
        "-g",
        nargs="+",
        choices=ALL_GENERATORS,
        default=ALL_GENERATORS,
        help="Which generators to run (default: all)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="Output duration in seconds " "(default: match input length)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        type=Path,
        default=None,
        help="Output directory " "(default: audio/<stem>_generated)",
    )
    return parser.parse_args()


def ensure_mono(audio: Audio) -> Audio:
    """Convert to mono if stereo."""
    if audio.is_mono:
        return audio
    mono_data = audio.data.mean(axis=1)
    return Audio(mono_data, audio.sample_rate, audio.bit_depth)


# ── Generator functions ───────────────────────────────────


def gen_poisson(
    src: Audio,
    duration: float,
    seed: int,
) -> Audio:
    """Scatter copies of the source at Poisson times."""
    pp = PoissonProcess(rate=4.0, seed=seed)
    out = pp.scatter_sound(
        src,
        duration,
        amplitude_range=(0.3, 1.0),
    )
    out.normalize(0.95)
    return out


def gen_brownian(
    src: Audio,
    duration: float,
    seed: int,
) -> Audio:
    """Amplitude-modulate the source with Brownian motion."""
    sr = src.sample_rate
    n = int(duration * sr)
    bm = BrownianMotion(volatility=0.5, seed=seed)
    env = bm.to_control(n, low=0.0, high=1.0)

    # Tile source to fill duration
    data = _tile_to_length(src.data, n)
    data = data * env[:, np.newaxis]
    out = Audio(data, sr)
    out.normalize(0.95)
    return out


def gen_lorenz(
    src: Audio,
    duration: float,
    seed: int,
) -> Audio:
    """Stereo-pan the source using a Lorenz attractor."""
    sr = src.sample_rate
    n = int(duration * sr)
    lorenz = LorenzAttractor(dt=1.0 / sr)
    ctrl = lorenz.to_control(n)
    pan = ctrl[:, 0]  # x-axis normalized to [0,1]

    mono = ensure_mono(src)
    data = _tile_to_length(mono.data, n).ravel()

    left = data * (1.0 - pan)
    right = data * pan
    stereo = np.column_stack([left, right])
    out = Audio(stereo, sr)
    out.normalize(0.95)
    return out


def gen_ca(
    src: Audio,
    duration: float,
    seed: int,
) -> Audio:
    """Gate the source on/off using a cellular automaton."""
    sr = src.sample_rate
    n = int(duration * sr)
    gate_res = 256
    ca = CellularAutomaton(rule=30, width=gate_res, seed=seed)
    gens = max(n // gate_res, 1)
    grid = ca.run(gens, ca.random_initial(0.5))
    pattern = grid.flatten().astype(np.float64)

    # Stretch pattern to match sample count
    indices = np.linspace(
        0,
        len(pattern) - 1,
        n,
    ).astype(int)
    gate = pattern[indices]

    # Smooth the gate to avoid clicks
    kernel_size = int(sr * 0.005)
    if kernel_size > 1:
        kernel = np.ones(kernel_size) / kernel_size
        gate = np.convolve(gate, kernel, mode="same")

    data = _tile_to_length(src.data, n)
    data = data * gate[:, np.newaxis]
    out = Audio(data, sr)
    out.normalize(0.95)
    return out


def gen_logistic(
    src: Audio,
    duration: float,
    seed: int,
) -> Audio:
    """Modulate amplitude with the logistic map."""
    sr = src.sample_rate
    n = int(duration * sr)
    lm = LogisticMap(r=3.99, x0=0.1 + seed * 1e-6)
    ctrl = lm.to_control(n)

    data = _tile_to_length(src.data, n)
    data = data * ctrl[:, np.newaxis]
    out = Audio(data, sr)
    out.normalize(0.95)
    return out


def gen_ou(
    src: Audio,
    duration: float,
    seed: int,
) -> Audio:
    """Mix source with Ornstein-Uhlenbeck noise."""
    sr = src.sample_rate
    n = int(duration * sr)
    ou = OrnsteinUhlenbeck(
        theta=3.0,
        sigma=0.4,
        seed=seed,
    )
    noise = ou.to_audio(duration, sr)

    data = _tile_to_length(src.data, n)
    noise_data = _tile_to_length(noise.data, n)
    mixed = 0.6 * data + 0.4 * noise_data
    out = Audio(mixed, sr)
    out.normalize(0.95)
    return out


# ── Helpers ───────────────────────────────────────────────


def _tile_to_length(
    data: np.ndarray,
    n: int,
) -> np.ndarray:
    """Tile (loop) audio data to exactly n samples."""
    src_len = data.shape[0]
    if src_len >= n:
        return data[:n].copy()
    reps = (n // src_len) + 1
    tiled = np.tile(data, (reps, 1))
    return tiled[:n].copy()


GENERATOR_MAP = {
    "poisson": ("poisson_scatter", gen_poisson),
    "brownian": ("brownian_amp", gen_brownian),
    "lorenz": ("lorenz_pan", gen_lorenz),
    "ca": ("ca_gate", gen_ca),
    "logistic": ("logistic_amp", gen_logistic),
    "ou": ("ou_mix", gen_ou),
}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    args = parse_args()
    src = Audio.read(args.input)
    duration = args.duration or src.duration
    stem = args.input.stem
    out_dir = args.outdir or (Path("audio") / f"{stem}_generated")
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Source: %s (%.3fs, %s)",
        args.input.name,
        src.duration,
        src,
    )
    log.info("Output duration: %.3fs", duration)
    log.info("Output directory: %s", out_dir)
    log.info("Generators: %s", ", ".join(args.generators))

    for name in args.generators:
        suffix, fn = GENERATOR_MAP[name]
        filename = f"{stem}_{suffix}.wav"
        out_path = out_dir / filename
        log.info("Running %s ...", name)
        result = fn(src, duration, args.seed)
        result.write(out_path)
        log.info("  -> %s (%s)", out_path, result)

    log.info("Done. %d files in %s/", len(args.generators), out_dir)


if __name__ == "__main__":
    main()
