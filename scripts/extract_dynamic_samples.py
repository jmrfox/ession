"""Extract the N most dynamic snippets from an audio file.

Snippets are ranked by RMS energy range (max - min) within
short analysis frames, so the "most dynamic" snippets are
those with the greatest variation in loudness.

Output is saved to audio/<stem>_samples/ (or _samples_2, etc.
if the directory already exists).
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from ession import Audio

log = logging.getLogger(__name__)


def rms_frames(
    data: np.ndarray,
    frame_size: int = 1024,
) -> np.ndarray:
    """Compute RMS energy per frame."""
    # Mix to mono for analysis
    if data.ndim == 2 and data.shape[1] > 1:
        mono = data.mean(axis=1)
    else:
        mono = data.ravel()
    n_frames = len(mono) // frame_size
    if n_frames == 0:
        return np.array([np.sqrt(np.mean(mono**2))])
    trimmed = mono[: n_frames * frame_size]
    frames = trimmed.reshape(n_frames, frame_size)
    return np.sqrt(np.mean(frames**2, axis=1))


def dynamics_score(
    data: np.ndarray,
    frame_size: int = 1024,
) -> float:
    """Score a snippet by its dynamic range.

    Returns the difference between the max and min
    RMS values across short frames within the snippet.
    """
    rms = rms_frames(data, frame_size)
    return float(rms.max() - rms.min())


def find_output_dir(stem: str) -> Path:
    """Find a non-existing output directory.

    Returns audio/<stem>_samples, or
    audio/<stem>_samples_2, _3, etc.
    """
    base = Path("audio") / f"{stem}_samples"
    if not base.exists():
        return base
    idx = 2
    while True:
        candidate = Path("audio") / f"{stem}_samples_{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the audio file",
    )
    parser.add_argument(
        "n",
        type=int,
        help="Number of snippets to extract",
    )
    parser.add_argument(
        "length",
        type=float,
        help="Snippet length in seconds",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    args = parse_args()
    input_path: Path = args.input
    n_snippets: int = args.n
    snippet_len: float = args.length

    audio = Audio.read(input_path)
    sr = audio.sample_rate
    snippet_samples = int(snippet_len * sr)
    total_samples = len(audio)

    if snippet_samples > total_samples:
        parser = argparse.ArgumentParser()
        parser.error(
            f"Snippet length {snippet_len}s exceeds "
            f"file duration {audio.duration:.3f}s"
        )

    # Slide through the file and score each position
    hop = max(snippet_samples // 4, 1)
    candidates: list[tuple[float, int]] = []
    pos = 0
    while pos + snippet_samples <= total_samples:
        chunk = audio.data[pos : pos + snippet_samples]
        score = dynamics_score(chunk)
        candidates.append((score, pos))
        pos += hop

    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Greedily pick top N non-overlapping snippets
    selected: list[tuple[float, int]] = []
    for score, start in candidates:
        end = start + snippet_samples
        overlap = False
        for _, sel_start in selected:
            sel_end = sel_start + snippet_samples
            if start < sel_end and end > sel_start:
                overlap = True
                break
        if not overlap:
            selected.append((score, start))
            if len(selected) == n_snippets:
                break

    if len(selected) < n_snippets:
        log.warning(
            "Only found %d non-overlapping snippets" " (requested %d)",
            len(selected),
            n_snippets,
        )

    # Sort selected by position for consistent ordering
    selected.sort(key=lambda x: x[1])

    stem = input_path.stem
    out_dir = find_output_dir(stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Extracting %d snippets of %ss from %s" " (%.3fs)",
        len(selected),
        snippet_len,
        input_path.name,
        audio.duration,
    )
    log.info("Output: %s/", out_dir)

    for i, (score, start) in enumerate(selected):
        t_start = start / sr
        t_end = t_start + snippet_len
        segment = audio.trim(t_start, t_end)
        filename = f"{stem}_{i + 1:03d}.wav"
        out_path = out_dir / filename
        segment.write(out_path)
        log.info(
            "  %s  t=%.3f-%.3fs  score=%.4f",
            filename,
            t_start,
            t_end,
            score,
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
