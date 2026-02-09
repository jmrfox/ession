import matplotlib
matplotlib.use("Agg")

from ession import (
    Audio,
    Note,
    Oscillator,
    Sequence,
    rest,
    note_to_freq,
)

print("All imports OK")
print(Note("A4", 1.0))
print(Note(440.0, 0.5, 0.8))
print(rest(2))
print(Oscillator())
print(f"A4 = {note_to_freq('A4'):.2f} Hz")
print(f"C4 = {note_to_freq('C4'):.2f} Hz")

osc = Oscillator("sine")
tone = osc.generate(440.0, 0.5)
print(tone)

seq = Sequence(
    notes=[
        Note("C4", 1),
        Note("E4", 1),
        Note("G4", 1),
        rest(1),
    ],
    tempo=120,
)
print(seq)
audio = seq.render()
print(f"Rendered: {audio}")
audio.write("audio/test_sequence.wav")
print("Wrote audio/test_sequence.wav")
