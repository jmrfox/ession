# ession — TODO

## Core (done)

- [x] `Audio` base class (numpy arrays, WAV write, normalize, trim, concat, mix, plot)
- [x] `Note` dataclass (pitch as Hz or name, duration, amplitude)
- [x] `Oscillator` (sine, square, sawtooth, triangle, attack/release envelope)
- [x] `Sequence` (render notes at tempo, repeat, beat or second durations)
- [x] `pitch` helpers (note name <-> frequency conversion)

## Generators — Probabilistic / Stochastic

- [ ] **Markov chains** — Transition matrices over pitches, chords, or timbral states. Drifting ambient sequences.
- [ ] **Brownian motion / Ornstein-Uhlenbeck** — Stochastic processes as audio signals or as control signals for synthesis params.
- [ ] **Poisson event triggers** — Sound events (grains, clicks, tones) at random intervals. Varying rate over time for density changes.
- [ ] **Bayesian generative models** — Priors over spectral shapes, sample posterior textures.

## Generators — Cellular Automata

- [ ] **1D automata -> waveforms** — Map each CA row to audio samples or short grains. Evolving textures from Rule 30, Rule 110, etc.
- [ ] **1D automata -> sequences** — Each cell = pitch/amplitude, each generation = one beat.
- [ ] **2D automata (Game of Life) -> spectrograms** — Time x frequency grid, run Life, invert via ISTFT.
- [ ] **Continuous automata** — Smooth-valued cells for organic ambient textures.

## Generators — Chaos / Dynamical Systems

- [ ] **Logistic map** — Different r values: silence -> periodic tones -> noise.
- [ ] **Lorenz attractor** — Map 3D trajectory to audio parameters or directly to waveform.
- [ ] **Henon map** — Chaotic 2D map as stereo audio or parameter control.

## Generators — Graph-Based

- [ ] **Tone graphs** — Directed weighted graph of pitches. Random walks, shortest paths, Eulerian paths as melodies.
- [ ] **Signal flow graphs** — DAG of DSP nodes (oscillators, filters, mixers, delays). Modular synth in code.
- [ ] **Spectral graphs** — Graph Laplacian eigenvalues as frequency components of drones.

## Generators — Pure Math

- [ ] **L-systems** — Fractal string rewriting interpreted as musical instructions (pitch shifts, duration changes).
- [ ] **Euclidean rhythms** — Bjorklund algorithm for beat patterns.
- [ ] **Number theory sequences** — Prime gaps, Fibonacci, Collatz trajectories as pitch sequences.
- [ ] **Exotic Fourier synthesis** — Timbres from Weierstrass functions, Dirichlet series, zeta function zeros as harmonics.
- [ ] **Iterated function systems** — Fractal point sets as time-frequency pairs.

## Generators — Neural / Learned

- [ ] **Tiny autoencoders** — Train on spectrograms, sample latent space for new textures.
- [ ] **RNN / LSTM sequence generation** — Train on note/feature sequences, sample for non-repeating ambient.
- [ ] **Neural ODEs** — Continuous dynamical system, integrate trajectory as audio parameter curves.

## Granular / Microsound

- [ ] **Grain cloud engine** — Chop audio into tiny grains (1-100ms), scatter with controlled params (pitch, position, density, pan).
- [ ] **Concatenative synthesis** — Corpus of micro-sounds, stitch by similarity metrics.

## Sample Support

- [ ] **Sample loader** — Read audio files into Audio objects for use as source material.
- [ ] **Sample manipulation** — Pitch shift, time stretch, reverse, chop.

## Infrastructure

- [ ] **`ession/generators/` subpackage** — Organize generator modules.
- [ ] **Effects pipeline** — Chain pedalboard effects on Audio objects.
- [ ] **Mixer / layering tools** — Multi-track composition helpers beyond Audio.mix.
