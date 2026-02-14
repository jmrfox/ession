"""Chop an audio file into equal-duration segments.

Segments are saved to audio/<stem>/ as <stem>_001.wav, etc.
"""

import argparse
import logging
from pathlib import Path

from ession import Audio

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the audio file",
    )
    parser.add_argument(
        "segments",
        type=int,
        help="Number of equal-duration segments",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only save the first N segments",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    args = parse_args()

    audio = Audio.read(args.input)
    stem = args.input.stem
    out_dir = Path("audio") / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    n_segments = args.segments
    seg_duration = audio.duration / n_segments
    n_output = min(n_segments, args.limit or n_segments)

    log.info(
        "Chopping %s (%.3fs) into %d segments" " of %.3fs each (saving %d)",
        args.input.name,
        audio.duration,
        n_segments,
        seg_duration,
        n_output,
    )

    for i in range(n_output):
        start = i * seg_duration
        end = (i + 1) * seg_duration
        segment = audio.trim(start, end)
        filename = f"{stem}_{i + 1:03d}.wav"
        out_path = out_dir / filename
        segment.write(out_path)
        log.info("  %s", out_path)

    log.info("Done. %d files in %s/", n_output, out_dir)


if __name__ == "__main__":
    main()
