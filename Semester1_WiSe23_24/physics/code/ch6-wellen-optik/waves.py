"""
waves.py
========

Ondes mécaniques : transversales, longitudinales, superposition.

Couvre :
    - Onde harmonique : y(x,t) = A sin(kx - ωt + φ)
    - Relation k = 2π/λ, ω = 2πf, v = λf = ω/k
    - Superposition et interférence
    - Ondes stationnaires : y = 2A sin(kx) cos(ωt)
    - Battements : f_batt = |f₁ - f₂|
    - Effet Doppler

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def onde_harmonique(
    x: np.ndarray, t: float, A: float, k: float, omega: float, phi: float = 0,
) -> np.ndarray:
    """y(x,t) = A sin(kx - ωt + φ)."""
    return A * np.sin(k * x - omega * t + phi)


def vitesse_onde(lam: float = None, f: float = None, k: float = None, omega: float = None) -> float:
    """v = λf = ω/k."""
    if lam is not None and f is not None:
        return lam * f
    if omega is not None and k is not None:
        return omega / k
    raise ValueError("Fournir (λ, f) ou (ω, k).")


def onde_stationnaire(
    x: np.ndarray, t: float, A: float, k: float, omega: float,
) -> np.ndarray:
    """y = 2A sin(kx) cos(ωt). Résultat de deux ondes contra-propagatives."""
    return 2 * A * np.sin(k * x) * np.cos(omega * t)


def battements(t: np.ndarray, A: float, f1: float, f2: float) -> np.ndarray:
    """Superposition de deux fréquences proches → battements."""
    return A * np.sin(2*np.pi*f1*t) + A * np.sin(2*np.pi*f2*t)


def frequence_battement(f1: float, f2: float) -> float:
    """f_batt = |f₁ - f₂|."""
    return abs(f1 - f2)


def doppler(f_source: float, v_source: float, v_obs: float, v_son: float = 343) -> float:
    """
    Effet Doppler :
        f_obs = f_s · (v_son + v_obs) / (v_son + v_source).
    Convention : v > 0 quand on s'éloigne.
    """
    return f_source * (v_son + v_obs) / (v_son + v_source)


# ======================================================================
#  Tracés
# ======================================================================

def tracer_onde_propagation(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    A, lam, f = 1.0, 2.0, 1.0
    k = 2*np.pi/lam
    omega = 2*np.pi*f
    x = np.linspace(0, 6, 500)

    for t in [0, 0.25/f, 0.5/f, 0.75/f]:
        y = onde_harmonique(x, t, A, k, omega)
        ax.plot(x, y, linewidth=1.5, alpha=0.7, label=f"$t = {t:.2f}$ s")

    ax.set_xlabel("$x$ (m)"); ax.set_ylabel("$y$")
    ax.set_title(f"Onde progressive ($\\lambda = {lam}$ m, $f = {f}$ Hz, $v = {lam*f}$ m/s)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_stationnaire(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    L = 1.0  # corde de 1 m
    for n in [1, 2, 3, 4]:
        k = n * np.pi / L
        x = np.linspace(0, L, 300)
        # Enveloppe
        ax.plot(x, 2*np.sin(k*x), "--", alpha=0.3, color=f"C{n-1}")
        ax.plot(x, -2*np.sin(k*x), "--", alpha=0.3, color=f"C{n-1}")
        # Instantané
        ax.plot(x, onde_stationnaire(x, 0, 1, k, 1), linewidth=2,
                label=f"$n={n}$ ($\\lambda = {2*L/n:.2f}$ m)")

    ax.set_xlabel("$x$ (m)"); ax.set_ylabel("$y$")
    ax.set_title(f"Ondes stationnaires sur une corde ($L = {L}$ m)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_battements(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    f1, f2 = 10, 11
    t = np.linspace(0, 3, 3000)
    y = battements(t, 1, f1, f2)

    ax.plot(t, y, "b-", linewidth=0.5)
    # Enveloppe
    env = 2 * np.cos(np.pi*(f1-f2)*t)
    ax.plot(t, env, "r--", linewidth=2, label=f"enveloppe ($f_{{batt}} = {abs(f1-f2)}$ Hz)")
    ax.plot(t, -env, "r--", linewidth=2)

    ax.set_xlabel("$t$ (s)"); ax.set_ylabel("$y$")
    ax.set_title(f"Battements : $f_1 = {f1}$ Hz, $f_2 = {f2}$ Hz")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Onde harmonique ===\n")
    lam, f = 0.5, 440  # La 440 Hz
    v = vitesse_onde(lam=lam, f=f)
    k = 2*np.pi/lam
    omega = 2*np.pi*f
    print(f"  λ = {lam} m, f = {f} Hz")
    print(f"  v = λf = {v} m/s")
    print(f"  k = 2π/λ = {k:.2f} rad/m")
    print(f"  ω = 2πf = {omega:.2f} rad/s")

    print(f"\n=== Ondes stationnaires ===\n")
    L = 1.0
    for n in range(1, 6):
        lam_n = 2*L/n
        f_n = v / lam_n
        print(f"  n={n} : λ = {lam_n:.3f} m, f = {f_n:.0f} Hz")

    print(f"\n=== Battements ===\n")
    print(f"  f₁ = 440, f₂ = 442 : f_batt = {frequence_battement(440, 442)} Hz")

    print(f"\n=== Effet Doppler ===\n")
    f_s = 1000  # Hz
    for v_s in [-30, 0, 30]:
        f_obs = doppler(f_s, v_s, 0)
        direction = "s'éloigne" if v_s > 0 else "s'approche" if v_s < 0 else "immobile"
        print(f"  Source {direction} ({abs(v_s)} m/s) : f_obs = {f_obs:.0f} Hz")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_onde_propagation(ax=axes[0])
    tracer_stationnaire(ax=axes[1])
    tracer_battements(ax=axes[2])
    plt.tight_layout()
    plt.savefig("waves_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
