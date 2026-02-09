from __future__ import annotations

_NOTE_NAMES = {
    "C": 0, "D": 2, "E": 4, "F": 5,
    "G": 7, "A": 9, "B": 11,
}

_A4_MIDI = 69
_A4_FREQ = 440.0


def note_to_midi(name: str) -> int:
    """Convert a note name like 'A4', 'C#3', 'Eb5' to MIDI number."""
    name = name.strip()
    if len(name) < 2:
        raise ValueError(f"Invalid note name: {name!r}")
    letter = name[0].upper()
    if letter not in _NOTE_NAMES:
        raise ValueError(f"Invalid note letter: {letter!r}")
    rest = name[1:]
    semitone = _NOTE_NAMES[letter]
    if rest.startswith("#") or rest.startswith("♯"):
        semitone += 1
        rest = rest[1:]
    elif rest.startswith("b") or rest.startswith("♭"):
        semitone -= 1
        rest = rest[1:]
    try:
        octave = int(rest)
    except ValueError:
        raise ValueError(
            f"Invalid octave in note name: {name!r}"
        ) from None
    return (octave + 1) * 12 + semitone


def midi_to_freq(midi: int) -> float:
    """Convert a MIDI note number to frequency in Hz."""
    return _A4_FREQ * 2.0 ** ((midi - _A4_MIDI) / 12.0)


def note_to_freq(name: str) -> float:
    """Convert a note name like 'A4' to frequency in Hz."""
    return midi_to_freq(note_to_midi(name))


def parse_pitch(pitch: float | str) -> float:
    """Accept a frequency (float/int) or note name (str).

    Returns frequency in Hz. Returns 0.0 for rest values.
    """
    if isinstance(pitch, str):
        return note_to_freq(pitch)
    return float(pitch)
