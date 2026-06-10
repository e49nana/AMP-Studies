"""
dft_fft.py
==========

Transformée de Fourier discrète et FFT.

Couvre :
    - DFT from-scratch : X[k] = Σ x[n] e^{-2πikn/N}, O(N²)
    - FFT (Cooley-Tukey) : O(N log N)
    - TF inverse : x[n] = (1/N) Σ X[k] e^{2πikn/N}
    - Fréquences : f_k = k/(NΔt), Nyquist = 1/(2Δt)
    - Spectre d'un signal composé
    - Zéro-padding et résolution fréquentielle

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import time


# ======================================================================
#  1. DFT from-scratch
# ======================================================================

def dft(x: np.ndarray) -> np.ndarray:
    """
    DFT : X[k] = Σ_{n=0}^{N-1} x[n] · e^{-2πi kn/N}.
    Complexité O(N²).
    """
    N = len(x)
    n = np.arange(N)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        X[k] = np.sum(x * np.exp(-2j * np.pi * k * n / N))
    return X


def idft(X: np.ndarray) -> np.ndarray:
    """IDFT : x[n] = (1/N) Σ X[k] e^{2πikn/N}."""
    N = len(X)
    k = np.arange(N)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        x[n] = np.sum(X * np.exp(2j * np.pi * n * k / N)) / N
    return x


# ======================================================================
#  2. FFT (Cooley-Tukey)
# ======================================================================

def fft_cooley_tukey(x: np.ndarray) -> np.ndarray:
    """
    FFT récursive (Cooley-Tukey, radix-2).
    N doit être une puissance de 2. Complexité O(N log N).
    """
    N = len(x)
    if N <= 1:
        return x.astype(complex)
    if N % 2 != 0:
        return dft(x)  # fallback si N n'est pas puissance de 2

    # Diviser
    X_even = fft_cooley_tukey(x[::2])
    X_odd = fft_cooley_tukey(x[1::2])

    # Twiddle factors
    W = np.exp(-2j * np.pi * np.arange(N//2) / N)

    # Combiner (butterfly)
    X = np.zeros(N, dtype=complex)
    X[:N//2] = X_even + W * X_odd
    X[N//2:] = X_even - W * X_odd
    return X


# ======================================================================
#  3. Utilitaires spectraux
# ======================================================================

def frequences(N: int, dt: float) -> np.ndarray:
    """Fréquences correspondant aux bins DFT : f_k = k/(N·dt)."""
    return np.fft.fftfreq(N, dt)


def spectre_amplitude(X: np.ndarray) -> np.ndarray:
    """|X[k]| / N (spectre normalisé)."""
    return np.abs(X) / len(X)


def spectre_phase(X: np.ndarray) -> np.ndarray:
    """arg(X[k])."""
    return np.angle(X)


def frequence_nyquist(dt: float) -> float:
    """f_Nyquist = 1/(2Δt)."""
    return 0.5 / dt


# ======================================================================
#  4. Benchmark DFT vs FFT
# ======================================================================

def benchmark(n_max: int = 12) -> dict:
    """Compare les temps DFT O(N²) vs FFT O(N log N) vs numpy."""
    results = {"N": [], "DFT": [], "FFT_mine": [], "numpy": []}
    for p in range(3, n_max + 1):
        N = 2**p
        x = np.random.randn(N)
        results["N"].append(N)

        t0 = time.perf_counter()
        dft(x)
        results["DFT"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        fft_cooley_tukey(x)
        results["FFT_mine"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        np.fft.fft(x)
        results["numpy"].append(time.perf_counter() - t0)

    return results


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_spectre_signal(ax1=None, ax2=None) -> None:
    """Spectre d'un signal composé de 3 sinusoïdes."""
    if ax1 is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    dt = 0.001; N = 1024
    t = np.arange(N) * dt
    f1, f2, f3 = 50, 120, 300
    signal = 1.0*np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t) + 0.3*np.sin(2*np.pi*f3*t)

    ax1.plot(t[:200]*1000, signal[:200], "b-", linewidth=1)
    ax1.set_xlabel("$t$ (ms)"); ax1.set_ylabel("amplitude")
    ax1.set_title("Signal : $\\sin(2\\pi\\cdot50t) + 0.5\\sin(2\\pi\\cdot120t) + 0.3\\sin(2\\pi\\cdot300t)$")
    ax1.grid(True, alpha=0.3)

    X = np.fft.fft(signal)
    freqs = frequences(N, dt)
    mask = freqs >= 0

    ax2.plot(freqs[mask], 2*spectre_amplitude(X)[mask], "b-", linewidth=1.5)
    ax2.set_xlabel("fréquence (Hz)"); ax2.set_ylabel("amplitude")
    ax2.set_title(f"Spectre (FFT, $f_{{Nyquist}} = {frequence_nyquist(dt):.0f}$ Hz)")
    for f, A in [(f1, 1), (f2, 0.5), (f3, 0.3)]:
        ax2.annotate(f"{f} Hz", (f, A), textcoords="offset points",
                    xytext=(5, 5), fontsize=10, color="red")
    ax2.grid(True, alpha=0.3); ax2.set_xlim(0, 500)


def tracer_benchmark(ax=None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    r = benchmark(12)
    ax.loglog(r["N"], r["DFT"], "rs-", markersize=5, label="DFT $O(N^2)$")
    ax.loglog(r["N"], r["FFT_mine"], "go-", markersize=5, label="FFT (mine) $O(N\\log N)$")
    ax.loglog(r["N"], r["numpy"], "b^-", markersize=5, label="numpy.fft")
    ax.set_xlabel("$N$"); ax.set_ylabel("temps (s)")
    ax.set_title("DFT $O(N^2)$ vs FFT $O(N\\log N)$")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_verification(ax=None) -> plt.Axes:
    """Vérifie DFT from-scratch vs numpy."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    N = 64
    x = np.random.randn(N)
    X_mine = fft_cooley_tukey(x)
    X_numpy = np.fft.fft(x)

    ax.plot(np.abs(X_mine), "bo", markersize=5, label="FFT (mine)")
    ax.plot(np.abs(X_numpy), "r+", markersize=8, label="numpy.fft")
    ax.set_xlabel("bin $k$"); ax.set_ylabel("$|X[k]|$")
    ax.set_title(f"Vérification : ||mine - numpy||_∞ = {np.max(np.abs(X_mine - X_numpy)):.2e}")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== DFT from-scratch ===\n")
    x = np.array([1, 2, 3, 4], dtype=float)
    X_mine = dft(x)
    X_numpy = np.fft.fft(x)
    print(f"  x = {x}")
    print(f"  DFT (mine)  = {np.round(X_mine, 4)}")
    print(f"  DFT (numpy) = {np.round(X_numpy, 4)}")
    print(f"  ||erreur|| = {np.max(np.abs(X_mine - X_numpy)):.2e}")

    # IDFT
    x_back = idft(X_mine)
    print(f"  IDFT → {np.round(x_back.real, 4)} ✓")

    print(f"\n=== FFT Cooley-Tukey ===\n")
    N = 1024
    x = np.random.randn(N)
    X_ct = fft_cooley_tukey(x)
    X_np = np.fft.fft(x)
    print(f"  N = {N}")
    print(f"  ||FFT_mine - numpy|| = {np.max(np.abs(X_ct - X_np)):.2e}")

    print(f"\n=== Nyquist ===\n")
    dt = 0.01
    print(f"  Δt = {dt} s → f_Nyquist = {frequence_nyquist(dt)} Hz")
    print(f"  → Ne peut détecter que des fréquences < {frequence_nyquist(dt)} Hz")

    print(f"\n=== Benchmark ===\n")
    r = benchmark(10)
    for N, t_dft, t_fft in zip(r["N"], r["DFT"], r["FFT_mine"]):
        speedup = t_dft / t_fft if t_fft > 0 else 0
        print(f"  N={N:>5} : DFT {t_dft:.4f}s, FFT {t_fft:.6f}s, speedup = {speedup:.0f}×")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tracer_spectre_signal(axes[0, 0], axes[0, 1])
    tracer_benchmark(axes[1, 0])
    tracer_verification(axes[1, 1])
    plt.tight_layout()
    plt.savefig("dft_fft_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
