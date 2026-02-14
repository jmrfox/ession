import matplotlib
matplotlib.use("Agg")

from ession import (
    Audio,
    BrownianMotion,
    CellularAutomaton,
    LogisticMap,
    LorenzAttractor,
    MarkovChain,
    OrnsteinUhlenbeck,
    PoissonProcess,
    Oscillator,
)

print("=== All imports OK ===\n")

# --- Markov Chain ---
print("--- MarkovChain ---")
mc = MarkovChain(
    states=["C3", "E3", "G3", "A3", None],
    transition_matrix=[
        [0.1, 0.3, 0.3, 0.2, 0.1],
        [0.2, 0.1, 0.3, 0.3, 0.1],
        [0.3, 0.2, 0.1, 0.2, 0.2],
        [0.2, 0.3, 0.2, 0.1, 0.2],
        [0.3, 0.2, 0.2, 0.3, 0.0],
    ],
    seed=42,
)
print(mc)
audio = mc.generate_audio(
    n_steps=16, tempo=90, oscillator=Oscillator("sine")
)
print(f"  Rendered: {audio}")
audio.write("audio/gen_markov.wav")
print("  Wrote audio/gen_markov.wav\n")

# --- Cellular Automaton ---
print("--- CellularAutomaton ---")
ca = CellularAutomaton(rule=30, width=256, seed=42)
print(ca)

audio = ca.to_waveform(generations=200)
print(f"  Waveform: {audio}")
audio.write("audio/gen_ca_waveform.wav")

audio = ca.to_spectral(generations=200)
print(f"  Spectral: {audio}")
audio.write("audio/gen_ca_spectral.wav")
print("  Wrote CA audio files\n")

# --- Logistic Map ---
print("--- LogisticMap ---")
lm = LogisticMap(r=3.99, x0=0.1)
print(lm)
audio = lm.to_audio(n_samples=44100 * 3)
print(f"  Audio: {audio}")
audio.write("audio/gen_logistic.wav")
print("  Wrote audio/gen_logistic.wav\n")

# --- Lorenz Attractor ---
print("--- LorenzAttractor ---")
lorenz = LorenzAttractor(dt=1.0 / 44100)
print(lorenz)
audio = lorenz.to_stereo(duration=3.0)
print(f"  Stereo: {audio}")
audio.write("audio/gen_lorenz.wav")
print("  Wrote audio/gen_lorenz.wav\n")

# --- Brownian Motion ---
print("--- BrownianMotion ---")
bm = BrownianMotion(volatility=1.0, seed=42)
print(bm)
audio = bm.to_audio(duration=3.0)
print(f"  Audio: {audio}")
audio.write("audio/gen_brownian.wav")
print("  Wrote audio/gen_brownian.wav\n")

# --- Ornstein-Uhlenbeck ---
print("--- OrnsteinUhlenbeck ---")
ou = OrnsteinUhlenbeck(theta=5.0, sigma=0.5, seed=42)
print(ou)
audio = ou.to_audio(duration=3.0)
print(f"  Audio: {audio}")
audio.write("audio/gen_ou.wav")
print("  Wrote audio/gen_ou.wav\n")

# --- Poisson Process ---
print("--- PoissonProcess ---")
pp = PoissonProcess(rate=20.0, seed=42)
print(pp)
audio = pp.to_impulse_audio(duration=3.0)
print(f"  Impulse train: {audio}")
audio.write("audio/gen_poisson_impulse.wav")

# Scatter a short tone at Poisson times
osc = Oscillator("sine", attack=0.002, release=0.01)
grain = osc.generate(880.0, 0.05)
audio = pp.scatter_sound(grain, duration=5.0)
audio.normalize()
print(f"  Scattered: {audio}")
audio.write("audio/gen_poisson_scatter.wav")
print("  Wrote Poisson audio files\n")

print("=== All generators OK ===")
