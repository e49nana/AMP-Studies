"""
oscillations.py
===============

Oscillations mécaniques : ressort, pendule, résonance.

Couvre :
    - Oscillateur harmonique : x'' + ω₀²x = 0
    - Oscillateur amorti : x'' + 2γx' + ω₀²x = 0
    - Oscillateur forcé : x'' + 2γx' + ω₀²x = F₀/m · cos(ωt)
    - Résonance : amplitude max quand ω ≈ ω₀
    - Diagramme de Bode (amplitude et phase vs fréquence)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def rk4_sys(f, t0, y0, t_end, h):
    y = np.array(y0, dtype=float)
    n = int((t_end-t0)/h)
    t = np.zeros(n+1); ys = np.zeros((n+1, len(y)))
    t[0] = t0; ys[0] = y
    for k in range(n):
        k1 = np.array(f(t[k], y))
        k2 = np.array(f(t[k]+h/2, y+h/2*k1))
        k3 = np.array(f(t[k]+h/2, y+h/2*k2))
        k4 = np.array(f(t[k]+h, y+h*k3))
        y = y + h/6*(k1+2*k2+2*k3+k4)
        t[k+1] = t[k]+h; ys[k+1] = y
    return t, ys


# ======================================================================
#  1. Solutions analytiques
# ======================================================================

def harmonique(x0: float, v0: float, omega0: float, t: np.ndarray) -> np.ndarray:
    """x(t) = x₀ cos(ω₀t) + (v₀/ω₀) sin(ω₀t)."""
    return x0 * np.cos(omega0 * t) + (v0 / omega0) * np.sin(omega0 * t)


def amorti(x0: float, v0: float, omega0: float, gamma: float, t: np.ndarray) -> np.ndarray:
    """Solution selon le régime (sous-amorti, critique, sur-amorti)."""
    disc = gamma**2 - omega0**2
    if disc < 0:  # sous-amorti
        omega_d = np.sqrt(-disc)
        A = x0
        B = (v0 + gamma * x0) / omega_d
        return np.exp(-gamma * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
    elif abs(disc) < 1e-10:  # critique
        return np.exp(-gamma * t) * (x0 + (v0 + gamma * x0) * t)
    else:  # sur-amorti
        r1 = -gamma + np.sqrt(disc)
        r2 = -gamma - np.sqrt(disc)
        A = (v0 - r2 * x0) / (r1 - r2)
        B = x0 - A
        return A * np.exp(r1 * t) + B * np.exp(r2 * t)


# ======================================================================
#  2. Oscillateur forcé : résonance
# ======================================================================

def amplitude_resonance(omega: np.ndarray, omega0: float, gamma: float, F0_m: float) -> np.ndarray:
    """Amplitude stationnaire : A(ω) = F₀/m / √((ω₀²-ω²)² + (2γω)²)."""
    return F0_m / np.sqrt((omega0**2 - omega**2)**2 + (2*gamma*omega)**2)


def phase_resonance(omega: np.ndarray, omega0: float, gamma: float) -> np.ndarray:
    """Phase : φ(ω) = -arctan(2γω / (ω₀² - ω²))."""
    return -np.arctan2(2*gamma*omega, omega0**2 - omega**2)


def frequence_resonance(omega0: float, gamma: float) -> float:
    """ω_res = √(ω₀² - 2γ²) (si > 0)."""
    disc = omega0**2 - 2*gamma**2
    return np.sqrt(disc) if disc > 0 else 0


def facteur_qualite(omega0: float, gamma: float) -> float:
    """Q = ω₀ / (2γ). Q grand → résonance aiguë."""
    return omega0 / (2 * gamma) if gamma > 0 else float("inf")


# ======================================================================
#  3. Tracés
# ======================================================================

def tracer_regimes_amortissement(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    omega0 = 5.0
    t = np.linspace(0, 5, 500)

    cas = [
        ("non amorti (γ=0)", 0),
        ("sous-amorti (γ=1)", 1),
        ("critique (γ=ω₀=5)", omega0),
        ("sur-amorti (γ=8)", 8),
    ]
    for nom, gamma in cas:
        x = amorti(1.0, 0, omega0, gamma, t)
        ax.plot(t, x, linewidth=2, label=nom)

    ax.set_xlabel("$t$ (s)"); ax.set_ylabel("$x(t)$")
    ax.set_title(f"Régimes d'amortissement ($\\omega_0 = {omega0}$)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_resonance(omega0: float = 5, ax_amp: plt.Axes | None = None,
                      ax_phase: plt.Axes | None = None) -> tuple:
    if ax_amp is None:
        fig, (ax_amp, ax_phase) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    omega = np.linspace(0.1, 3*omega0, 500)

    for gamma in [0.2, 0.5, 1.0, 2.0]:
        A = amplitude_resonance(omega, omega0, gamma, 1.0)
        phi = phase_resonance(omega, omega0, gamma)
        Q = facteur_qualite(omega0, gamma)
        ax_amp.plot(omega/omega0, A, linewidth=2, label=f"γ={gamma} (Q={Q:.1f})")
        ax_phase.plot(omega/omega0, np.degrees(phi), linewidth=2)

    ax_amp.axvline(1, color="red", linestyle=":", alpha=0.3)
    ax_amp.set_ylabel("Amplitude $A(\\omega)$")
    ax_amp.set_title(f"Résonance ($\\omega_0 = {omega0}$)")
    ax_amp.legend(fontsize=9); ax_amp.grid(True, alpha=0.3)

    ax_phase.axvline(1, color="red", linestyle=":", alpha=0.3)
    ax_phase.axhline(-90, color="grey", linestyle=":", alpha=0.3)
    ax_phase.set_xlabel("$\\omega / \\omega_0$")
    ax_phase.set_ylabel("Phase $\\varphi$ (°)")
    ax_phase.grid(True, alpha=0.3)

    return ax_amp, ax_phase


def tracer_force_simulation(ax: plt.Axes | None = None) -> plt.Axes:
    """Simule l'oscillateur forcé par RK4 et montre le transitoire."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    omega0, gamma, F0_m = 5.0, 0.3, 1.0

    for omega_f in [2, 5, 8]:
        f = lambda t, y, w=omega_f: np.array([
            y[1],
            -2*gamma*y[1] - omega0**2*y[0] + F0_m*np.cos(w*t)
        ])
        t, ys = rk4_sys(f, 0, [0, 0], 20, 0.001)
        ax.plot(t, ys[:, 0], linewidth=1.5, alpha=0.7,
                label=f"$\\omega_f = {omega_f}$" + (" (résonance)" if omega_f == omega0 else ""))

    ax.set_xlabel("$t$ (s)"); ax.set_ylabel("$x(t)$")
    ax.set_title("Oscillateur forcé : transitoire + régime permanent")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    omega0 = 5.0

    print("=== Oscillateur harmonique (ω₀ = 5) ===\n")
    print(f"  T = 2π/ω₀ = {2*np.pi/omega0:.4f} s")
    print(f"  f = ω₀/(2π) = {omega0/(2*np.pi):.4f} Hz")

    print(f"\n=== Amortissement ===\n")
    for gamma in [0, 1, omega0, 8]:
        disc = gamma**2 - omega0**2
        regime = "non amorti" if gamma == 0 else "sous-amorti" if disc < 0 else "critique" if abs(disc) < 0.1 else "sur-amorti"
        print(f"  γ = {gamma:>4} : {regime}")

    print(f"\n=== Résonance ===\n")
    for gamma in [0.2, 0.5, 1, 2]:
        Q = facteur_qualite(omega0, gamma)
        w_res = frequence_resonance(omega0, gamma)
        A_max = amplitude_resonance(np.array([w_res]), omega0, gamma, 1.0)[0] if w_res > 0 else 0
        print(f"  γ={gamma:<3} : Q={Q:>5.1f}, ω_res={w_res:.2f}, A_max={A_max:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tracer_regimes_amortissement(ax=axes[0, 0])
    tracer_force_simulation(ax=axes[0, 1])
    tracer_resonance(omega0, ax_amp=axes[1, 0], ax_phase=axes[1, 1])
    plt.tight_layout()
    plt.savefig("oscillations_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
