"""
signal_processing.py
====================

Traitement du signal : filtrage, fenêtrage, spectrogramme.

Couvre :
    - Filtrage fréquentiel : passe-bas, passe-haut, passe-bande
    - Fenêtrage : Hann, Hamming, Blackman (réduction du leakage)
    - Spectrogramme (STFT) : représentation temps-fréquence
    - Débruitage : signal + bruit → filtre → signal propre
    - Échantillonnage et aliasing (Shannon)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Filtrage fréquentiel
# ======================================================================

def filtre_passe_bas(signal: np.ndarray, dt: float, f_coupure: float) -> np.ndarray:
    """Filtre passe-bas idéal dans le domaine fréquentiel."""
    N = len(signal)
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, dt)
    X[np.abs(freqs) > f_coupure] = 0
    return np.real(np.fft.ifft(X))


def filtre_passe_haut(signal: np.ndarray, dt: float, f_coupure: float) -> np.ndarray:
    """Filtre passe-haut idéal."""
    N = len(signal)
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, dt)
    X[np.abs(freqs) < f_coupure] = 0
    return np.real(np.fft.ifft(X))


def filtre_passe_bande(signal: np.ndarray, dt: float,
                         f_low: float, f_high: float) -> np.ndarray:
    """Filtre passe-bande idéal."""
    N = len(signal)
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, dt)
    mask = (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high)
    X[~mask] = 0
    return np.real(np.fft.ifft(X))


# ======================================================================
#  2. Fenêtrage
# ======================================================================

def fenetre_hann(N: int) -> np.ndarray:
    """w[n] = 0.5(1 - cos(2πn/(N-1)))."""
    n = np.arange(N)
    return 0.5 * (1 - np.cos(2*np.pi*n / (N-1)))


def fenetre_hamming(N: int) -> np.ndarray:
    """w[n] = 0.54 - 0.46 cos(2πn/(N-1))."""
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2*np.pi*n / (N-1))


def fenetre_blackman(N: int) -> np.ndarray:
    """w[n] = 0.42 - 0.5 cos(2πn/(N-1)) + 0.08 cos(4πn/(N-1))."""
    n = np.arange(N)
    return 0.42 - 0.5*np.cos(2*np.pi*n/(N-1)) + 0.08*np.cos(4*np.pi*n/(N-1))


# ======================================================================
#  3. Spectrogramme (STFT)
# ======================================================================

def spectrogramme(
    signal: np.ndarray, dt: float, window_size: int = 256,
    hop: int = 64, window_fn: callable = fenetre_hann,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Short-Time Fourier Transform (STFT).
    Renvoie (times, freqs, magnitude).
    """
    N = len(signal)
    n_frames = (N - window_size) // hop + 1
    window = window_fn(window_size)
    freqs = np.fft.rfftfreq(window_size, dt)
    times = np.arange(n_frames) * hop * dt

    spec = np.zeros((len(freqs), n_frames))
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + window_size] * window
        spec[:, i] = np.abs(np.fft.rfft(frame))

    return times, freqs, spec


# ======================================================================
#  4. Aliasing
# ======================================================================

def demo_aliasing(f_signal: float, f_sample: float, duration: float = 0.1) -> dict:
    """
    Shannon : f_sample > 2·f_signal pour éviter l'aliasing.
    Si f_sample < 2·f_signal, on voit une fausse fréquence f_alias.
    """
    dt = 1 / f_sample
    t = np.arange(0, duration, dt)
    signal = np.sin(2*np.pi*f_signal*t)
    f_nyquist = f_sample / 2
    f_alias = abs(f_signal - round(f_signal / f_sample) * f_sample)
    aliased = f_signal > f_nyquist
    return {
        "t": t, "signal": signal,
        "f_nyquist": f_nyquist,
        "aliased": aliased,
        "f_alias": f_alias if aliased else f_signal,
    }


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_filtrage(ax=None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    dt = 0.001; N = 2048
    t = np.arange(N) * dt
    rng = np.random.default_rng(42)

    signal_pur = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*50*t)
    bruit = 0.8 * rng.standard_normal(N)
    signal_bruite = signal_pur + bruit
    signal_filtre = filtre_passe_bas(signal_bruite, dt, 60)

    ax.plot(t[:500], signal_bruite[:500], "grey", alpha=0.4, linewidth=0.5, label="bruité")
    ax.plot(t[:500], signal_filtre[:500], "b-", linewidth=2, label="filtré (< 60 Hz)")
    ax.plot(t[:500], signal_pur[:500], "r--", linewidth=1, alpha=0.7, label="original")
    ax.set_xlabel("$t$ (s)"); ax.set_ylabel("amplitude")
    ax.set_title("Débruitage par filtre passe-bas")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_fenetres(ax=None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    N = 64
    for nom, w in [("Rectangle", np.ones(N)), ("Hann", fenetre_hann(N)),
                    ("Hamming", fenetre_hamming(N)), ("Blackman", fenetre_blackman(N))]:
        # Spectre de la fenêtre
        W = np.fft.fftshift(np.abs(np.fft.fft(w, 512)))
        W_db = 20*np.log10(W / np.max(W) + 1e-15)
        f = np.linspace(-0.5, 0.5, 512)
        ax.plot(f, W_db, linewidth=1.5, label=nom)

    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-100, 5)
    ax.set_xlabel("fréquence normalisée"); ax.set_ylabel("dB")
    ax.set_title("Spectres des fenêtres (leakage)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_spectrogramme(ax=None) -> plt.Axes:
    """Spectrogramme d'un chirp (fréquence croissante)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    dt = 0.0005; duration = 2
    t = np.arange(0, duration, dt)
    # Chirp : fréquence de 50 Hz à 500 Hz
    f_inst = 50 + 225 * t
    signal = np.sin(2*np.pi * (50*t + 225/2*t**2))

    times, freqs, spec = spectrogramme(signal, dt, window_size=256, hop=32)
    ax.pcolormesh(times, freqs, 20*np.log10(spec + 1e-10), shading="gouraud",
                   cmap="inferno")
    ax.set_xlabel("$t$ (s)"); ax.set_ylabel("fréquence (Hz)")
    ax.set_title("Spectrogramme d'un chirp (50 → 500 Hz)")
    ax.set_ylim(0, 600)
    return ax


def tracer_aliasing(ax=None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    t_fine = np.linspace(0, 0.05, 5000)
    f_signal = 100

    for f_sample in [500, 200, 150]:
        r = demo_aliasing(f_signal, f_sample, 0.05)
        ax.plot(r["t"], r["signal"], "o-", markersize=3, linewidth=1,
                label=f"$f_s = {f_sample}$ Hz" +
                (" ✗ alias" if r["aliased"] else " ✓"))

    ax.plot(t_fine, np.sin(2*np.pi*f_signal*t_fine), "k-", alpha=0.2,
            linewidth=1, label=f"signal continu ({f_signal} Hz)")
    ax.set_xlabel("$t$ (s)"); ax.set_ylabel("amplitude")
    ax.set_title(f"Aliasing : $f_{{signal}} = {f_signal}$ Hz, Shannon $f_s > {2*f_signal}$ Hz")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Filtrage ===\n")
    dt = 0.001; N = 2048
    t = np.arange(N) * dt
    rng = np.random.default_rng(42)
    signal = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*50*t)
    bruit = 0.8*rng.standard_normal(N)
    filtre = filtre_passe_bas(signal + bruit, dt, 60)
    erreur = np.sqrt(np.mean((filtre - signal)**2))
    print(f"  Signal : 10 Hz + 50 Hz, bruit σ = 0.8")
    print(f"  Filtre passe-bas (< 60 Hz)")
    print(f"  RMSE après filtrage : {erreur:.4f}")

    print(f"\n=== Aliasing ===\n")
    for fs in [500, 200, 150, 100]:
        r = demo_aliasing(100, fs)
        print(f"  f_s = {fs:>3} Hz : f_Nyquist = {r['f_nyquist']:.0f} Hz, "
              f"aliased = {r['aliased']}, f_perçue = {r['f_alias']:.0f} Hz")

    print(f"\n=== Fenêtrage ===")
    print(f"  Rectangle : résolution max mais leakage fort")
    print(f"  Hann/Hamming : bon compromis résolution/leakage")
    print(f"  Blackman : leakage minimal mais résolution réduite")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tracer_filtrage(axes[0, 0])
    tracer_fenetres(axes[0, 1])
    tracer_spectrogramme(axes[1, 0])
    tracer_aliasing(axes[1, 1])
    plt.tight_layout()
    plt.savefig("signal_processing_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
