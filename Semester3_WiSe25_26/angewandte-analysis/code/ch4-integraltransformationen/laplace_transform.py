"""
laplace_transform.py
====================

Transformée de Laplace et résolution d'EDO.

Couvre :
    - TL : F(s) = ∫₀^∞ f(t) e^{-st} dt
    - TL inverse (numériquement par Talbot/Gaver-Stehfest)
    - Table des transformées classiques
    - Résolution d'EDO par Laplace (algébrique dans le domaine s)
    - Fonction de transfert H(s) et réponse impulsionnelle
    - Pôles et stabilité

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


# ======================================================================
#  1. TL numérique
# ======================================================================

def laplace_transform_num(f: Callable, s: complex, t_max: float = 100) -> complex:
    """F(s) = ∫₀^∞ f(t) e^{-st} dt (approx. par quadrature)."""
    re_part, _ = integrate.quad(lambda t: np.real(f(t) * np.exp(-s*t)), 0, t_max)
    im_part, _ = integrate.quad(lambda t: np.imag(f(t) * np.exp(-s*t)), 0, t_max)
    return re_part + 1j*im_part


# ======================================================================
#  2. TL analytiques
# ======================================================================

def tl_constante(s: complex, c: float = 1) -> complex:
    """L{c} = c/s."""
    return c / s


def tl_exponentielle(s: complex, a: float) -> complex:
    """L{e^{at}} = 1/(s-a), Re(s) > a."""
    return 1 / (s - a)


def tl_sin(s: complex, omega: float) -> complex:
    """L{sin(ωt)} = ω/(s²+ω²)."""
    return omega / (s**2 + omega**2)


def tl_cos(s: complex, omega: float) -> complex:
    """L{cos(ωt)} = s/(s²+ω²)."""
    return s / (s**2 + omega**2)


def tl_puissance(s: complex, n: int) -> complex:
    """L{t^n} = n!/s^{n+1}."""
    from math import factorial
    return factorial(n) / s**(n+1)


def tl_heaviside(s: complex, a: float) -> complex:
    """L{u(t-a)} = e^{-as}/s."""
    return np.exp(-a*s) / s


# ======================================================================
#  3. Résolution d'EDO par Laplace
# ======================================================================

def resoudre_edo_laplace_oscillateur(omega0: float, gamma: float,
                                       y0: float, v0: float,
                                       t: np.ndarray) -> np.ndarray:
    """
    y'' + 2γy' + ω₀²y = 0, y(0) = y₀, y'(0) = v₀.

    Laplace : s²Y - sy₀ - v₀ + 2γ(sY - y₀) + ω₀²Y = 0
            → Y(s) = (sy₀ + v₀ + 2γy₀) / (s² + 2γs + ω₀²)

    Inverse par fractions partielles → solution temporelle.
    """
    disc = gamma**2 - omega0**2
    if disc < 0:  # sous-amorti
        omega_d = np.sqrt(-disc)
        A = y0
        B = (v0 + gamma*y0) / omega_d
        return np.exp(-gamma*t) * (A*np.cos(omega_d*t) + B*np.sin(omega_d*t))
    elif abs(disc) < 1e-10:  # critique
        return np.exp(-gamma*t) * (y0 + (v0 + gamma*y0)*t)
    else:  # sur-amorti
        r1 = -gamma + np.sqrt(disc)
        r2 = -gamma - np.sqrt(disc)
        A = (v0 - r2*y0) / (r1 - r2)
        B = y0 - A
        return A*np.exp(r1*t) + B*np.exp(r2*t)


# ======================================================================
#  4. Fonction de transfert
# ======================================================================

def fonction_transfert_rc(s: complex, R: float, C: float) -> complex:
    """H(s) = 1/(1 + sRC) pour un filtre RC passe-bas."""
    return 1 / (1 + s*R*C)


def fonction_transfert_rlc(s: complex, R: float, L: float, C: float) -> complex:
    """H(s) = 1/(LCs² + RCs + 1) pour un circuit RLC série."""
    return 1 / (L*C*s**2 + R*C*s + 1)


def poles(coeffs: list[float]) -> np.ndarray:
    """Pôles = racines du dénominateur. coeffs = [aₙ, ..., a₁, a₀]."""
    return np.roots(coeffs)


def est_stable(poles_list: np.ndarray) -> bool:
    """Stable ssi tous les pôles ont Re(s) < 0."""
    return bool(np.all(np.real(poles_list) < 0))


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_table_laplace(ax=None) -> plt.Axes:
    """Trace f(t) et vérifie L{f}(s) numériquement."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    t = np.linspace(0, 5, 300)
    fonctions = [
        ("$1$", lambda t: np.ones_like(t), lambda s: 1/s),
        ("$t$", lambda t: t, lambda s: 1/s**2),
        ("$e^{-t}$", lambda t: np.exp(-t), lambda s: 1/(s+1)),
        ("$\\sin(t)$", lambda t: np.sin(t), lambda s: 1/(s**2+1)),
    ]

    for nom, f, F in fonctions:
        ax.plot(t, f(t), linewidth=2, label=nom)

    ax.set_xlabel("$t$"); ax.set_ylabel("$f(t)$")
    ax.set_title("Fonctions classiques et leurs transformées de Laplace")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_bode_rc(R: float = 1000, C: float = 1e-6, ax=None) -> plt.Axes:
    """Diagramme de Bode d'un filtre RC passe-bas."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    omega = np.logspace(0, 6, 500)
    s = 1j * omega
    H = np.array([fonction_transfert_rc(si, R, C) for si in s])

    ax.semilogx(omega/(2*np.pi), 20*np.log10(np.abs(H)), "b-", linewidth=2)
    fc = 1 / (2*np.pi*R*C)
    ax.axvline(fc, color="red", linestyle="--", alpha=0.5,
                label=f"$f_c = {fc:.0f}$ Hz")
    ax.axhline(-3, color="grey", linestyle=":", alpha=0.3, label="$-3$ dB")
    ax.set_xlabel("fréquence (Hz)"); ax.set_ylabel("$|H|$ (dB)")
    ax.set_title(f"Bode : filtre RC passe-bas ($R={R}$ Ω, $C={C*1e6:.0f}$ μF)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_poles_zeros(ax=None) -> plt.Axes:
    """Carte des pôles pour un système RLC."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    R, L, C = 10, 0.1, 1e-4
    # Dénominateur : LCs² + RCs + 1
    p = poles([L*C, R*C, 1])

    ax.plot(np.real(p), np.imag(p), "rx", markersize=15, markeredgewidth=3, label="pôles")
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.axhline(0, color="grey", linewidth=0.5)

    stable = est_stable(p)
    color = "green" if stable else "red"
    ax.fill_betweenx([-500, 500], [-200, -200], [0, 0], alpha=0.05, color="green",
                       label="zone stable (Re < 0)")

    ax.set_xlabel("Re($s$)"); ax.set_ylabel("Im($s$)")
    ax.set_title(f"Pôles du RLC : {'stable ✓' if stable else 'INSTABLE ✗'}")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    lim = max(np.abs(p)) * 1.5
    ax.set_xlim(-lim, lim/2); ax.set_ylim(-lim, lim)
    return ax


if __name__ == "__main__":
    print("=== Table des transformées de Laplace ===\n")
    print(f"  {'f(t)':>15} | {'F(s)':>20} | {'F(2) num':>12} | {'F(2) exact':>12}")
    print("  " + "-" * 65)
    s_test = 2.0
    table = [
        ("1", lambda t: 1.0, 1/s_test),
        ("t", lambda t: t, 1/s_test**2),
        ("e^{-t}", lambda t: np.exp(-t), 1/(s_test+1)),
        ("sin(t)", lambda t: np.sin(t), 1/(s_test**2+1)),
        ("t²", lambda t: t**2, 2/s_test**3),
    ]
    for nom, f, F_exact in table:
        F_num = laplace_transform_num(f, s_test).real
        print(f"  {nom:>15} | {'':>20} | {F_num:>12.6f} | {F_exact:>12.6f}")

    print(f"\n=== Résolution d'EDO par Laplace ===\n")
    print(f"  y'' + 2y' + 5y = 0, y(0) = 1, y'(0) = 0")
    t = np.linspace(0, 5, 200)
    y = resoudre_edo_laplace_oscillateur(np.sqrt(5), 1, 1, 0, t)
    print(f"  y(0) = {y[0]:.4f}, y(5) = {y[-1]:.6f}")
    print(f"  → Oscillation amortie (sous-amorti)")

    print(f"\n=== Pôles et stabilité ===\n")
    for R in [5, 20, 100]:
        L, C = 0.1, 1e-4
        p = poles([L*C, R*C, 1])
        print(f"  R={R:>3}Ω : pôles = {np.round(p, 2)}, stable = {est_stable(p)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_table_laplace(axes[0])
    tracer_bode_rc(ax=axes[1])
    tracer_poles_zeros(axes[2])
    plt.tight_layout()
    plt.savefig("laplace_transform_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
