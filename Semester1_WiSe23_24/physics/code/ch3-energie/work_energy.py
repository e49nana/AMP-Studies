"""
work_energy.py
==============

Travail et énergie cinétique.

Couvre :
    - Travail W = ∫ F·ds (force constante et variable)
    - Énergie cinétique E_cin = ½mv²
    - Théorème travail-énergie : W_net = ΔE_cin
    - Travail d'un ressort : W = ½kx²
    - Travail de la gravité : W = mgh
    - Travail le long d'un chemin (intégrale curviligne)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


G = 9.81


def travail_force_constante(F: float, d: float, theta: float = 0) -> float:
    """W = F·d·cos(θ). θ = angle entre F et le déplacement."""
    return F * d * np.cos(theta)


def travail_force_variable(F: Callable, a: float, b: float) -> float:
    """W = ∫_a^b F(x) dx."""
    val, _ = integrate.quad(F, a, b)
    return val


def energie_cinetique(m: float, v: float) -> float:
    """E_cin = ½mv²."""
    return 0.5 * m * v**2


def vitesse_from_Ecin(m: float, Ecin: float) -> float:
    """v = √(2E_cin/m)."""
    return np.sqrt(2 * Ecin / m)


def travail_ressort(k: float, x1: float, x2: float) -> float:
    """W = ½k(x₁² - x₂²). Attention au signe : le ressort résiste."""
    return 0.5 * k * (x1**2 - x2**2)


def travail_gravite(m: float, h1: float, h2: float, g: float = G) -> float:
    """W_grav = mg(h₁ - h₂). Positif quand on descend."""
    return m * g * (h1 - h2)


def theoreme_travail_energie(m: float, v1: float, v2: float) -> float:
    """W_net = ΔE_cin = ½m(v₂² - v₁²)."""
    return 0.5 * m * (v2**2 - v1**2)


# ======================================================================
#  Tracés
# ======================================================================

def tracer_travail_graphique(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre W = aire sous la courbe F(x)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Force variable : F(x) = 3x + 2
    F = lambda x: 3*x + 2
    x = np.linspace(0, 4, 200)

    ax.plot(x, F(x), "b-", linewidth=2, label="$F(x) = 3x + 2$")
    ax.fill_between(x[:150], F(x[:150]), alpha=0.3, color="orange",
                     label=f"$W = \\int_0^3 F\\,dx = {travail_force_variable(F, 0, 3):.1f}$ J")

    ax.set_xlabel("$x$ (m)"); ax.set_ylabel("$F$ (N)")
    ax.set_title("Travail = aire sous $F(x)$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_ressort(ax: plt.Axes | None = None) -> plt.Axes:
    """Énergie potentielle du ressort et travail."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    k = 50  # N/m
    x = np.linspace(-0.3, 0.3, 200)
    F = -k * x  # force de rappel
    Ep = 0.5 * k * x**2

    ax.plot(x*100, F, "b-", linewidth=2, label="$F = -kx$")
    ax.plot(x*100, Ep, "r-", linewidth=2, label="$E_p = \\frac{1}{2}kx^2$")

    # Travail pour comprimer de 0 à 20 cm
    x_comp = 0.2
    W = 0.5 * k * x_comp**2
    ax.fill_between(x[:100]*100, 0, Ep[:100], alpha=0.2, color="red")
    ax.annotate(f"W = {W:.1f} J", xy=(10, W/2), fontsize=11, color="red")

    ax.set_xlabel("$x$ (cm)"); ax.set_ylabel("$F$ (N) / $E_p$ (J)")
    ax.set_title(f"Ressort ($k = {k}$ N/m)")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_theoreme_WE(ax: plt.Axes | None = None) -> plt.Axes:
    """Vérifie W_net = ΔE_cin sur un exemple concret."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Bloc poussé sur surface rugueuse
    m = 5  # kg
    F_app = 30  # N
    mu = 0.3
    F_frott = mu * m * G
    F_net = F_app - F_frott

    t = np.linspace(0, 3, 200)
    a = F_net / m
    v = a * t
    x = 0.5 * a * t**2
    Ecin = 0.5 * m * v**2
    W_app = F_app * x
    W_frott = -F_frott * x
    W_net = W_app + W_frott

    ax.plot(t, Ecin, "b-", linewidth=2, label="$E_{cin} = \\frac{1}{2}mv^2$")
    ax.plot(t, W_net, "r--", linewidth=2, label="$W_{net} = W_{app} + W_{frott}$")
    ax.plot(t, W_app, "g:", linewidth=1.5, alpha=0.5, label="$W_{app}$")
    ax.plot(t, W_frott, "m:", linewidth=1.5, alpha=0.5, label="$W_{frott}$")

    ax.set_xlabel("$t$ (s)"); ax.set_ylabel("énergie (J)")
    ax.set_title(f"Théorème W-E : $W_{{net}} = \\Delta E_{{cin}}$ (m={m} kg, F={F_app} N, μ={mu})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Travail d'une force constante ===\n")
    for theta_deg in [0, 30, 60, 90]:
        W = travail_force_constante(100, 5, np.radians(theta_deg))
        print(f"  F=100N, d=5m, θ={theta_deg}° : W = {W:.1f} J")

    print(f"\n=== Travail d'une force variable ===\n")
    F = lambda x: 3*x + 2
    W = travail_force_variable(F, 0, 3)
    print(f"  F(x) = 3x+2, x ∈ [0,3] : W = {W:.1f} J")
    print(f"  Vérif : ∫(3x+2)dx = [3x²/2+2x] = {3*9/2+6:.1f} J ✓")

    print(f"\n=== Théorème travail-énergie ===\n")
    m, v1, v2 = 2, 3, 7
    W = theoreme_travail_energie(m, v1, v2)
    print(f"  m={m}kg, v₁={v1} → v₂={v2} m/s")
    print(f"  W_net = ΔE_cin = ½·{m}·({v2}²-{v1}²) = {W:.1f} J")

    print(f"\n=== Ressort ===\n")
    k = 200
    for x in [0.05, 0.1, 0.2]:
        Ep = 0.5*k*x**2
        v = vitesse_from_Ecin(0.5, Ep)
        print(f"  k={k}, x={x*100:.0f}cm : E_p = {Ep:.2f} J → v (m=0.5kg) = {v:.2f} m/s")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_travail_graphique(ax=axes[0])
    tracer_ressort(ax=axes[1])
    tracer_theoreme_WE(ax=axes[2])
    plt.tight_layout()
    plt.savefig("work_energy_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
