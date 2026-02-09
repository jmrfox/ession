from ession import (
    Note,
    Oscillator,
    Sequence,
    rest,
)

osc = Oscillator("triangle")

notes = []
for _ in range(9):
    notes.append(Note("C3", 1 / 4))
notes.append(rest(1 / 4))
for _ in range(6):
    notes.append(Note("A3", 1 / 4))
for _ in range(9):
    notes.append(Note("Bb2", 1 / 4))
notes.append(rest(1 / 4))
for _ in range(6):
    notes.append(Note("Eb3", 1 / 4))

seq = Sequence(
    notes=notes,
    tempo=120,
    oscillator=osc,
)
audio = seq.render(repeat=4)
audio.write("audio/example_1.wav")
