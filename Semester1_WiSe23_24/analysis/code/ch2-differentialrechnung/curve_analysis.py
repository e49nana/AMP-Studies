"""
curve_analysis.py
=================

Kurvendiskussion complète d'une fonction.

Couvre :
    - Domaine, zéros, signe
    - Monotonie et extrema (f' = 0, signe de f'')
    - Convexité et points d'inflexion (f'' = 0, signe change)
    - Asymptotes horizontales et verticales
    - Tracé annoté automatique

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def deriv(f: Callable, x: float, h: float = 1e-6) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)


def deriv2(f: Callable, x: float, h: float = 1e-4) -> float:
    return (f(x + h) - 2*f(x) + f(x - h)) / (h*h)


@dataclass
class Kurvendiskussion:
    """Résultat complet d'une analyse de courbe."""
    nullstellen: list[float] = field(default_factory=list)
    extrema: list[tuple[float, float, str]] = field(default_factory=list)  # (x, f(x), "min"/"max")
    wendepunkte: list[tuple[float, float]] = field(default_factory=list)  # (x, f(x))
    monotonie: list[tuple[float, float, str]] = field(default_factory=list)  # (a, b, "↑"/"↓")


def trouver_zeros(f: Callable, a: float, b: float, n_scan: int = 1000) -> list[float]:
    """Trouve les zéros de f sur [a, b] par balayage + bissection."""
    xs = np.linspace(a, b, n_scan)
    zeros = []
    for i in range(len(xs) - 1):
        fa, fb = f(xs[i]), f(xs[i+1])
        if fa * fb < 0:
            # Bissection
            lo, hi = xs[i], xs[i+1]
            for _ in range(60):
                m = (lo + hi) / 2
                if f(lo) * f(m) < 0:
                    hi = m
                else:
                    lo = m
            zeros.append((lo + hi) / 2)
        elif abs(fa) < 1e-10:
            zeros.append(xs[i])
    return zeros


def trouver_extrema(f: Callable, a: float, b: float) -> list[tuple[float, float, str]]:
    """Trouve les extrema locaux (f' = 0, classifie par f'')."""
    zeros_fprime = trouver_zeros(lambda x: deriv(f, x), a, b)
    extrema = []
    for x0 in zeros_fprime:
        fpp = deriv2(f, x0)
        if fpp > 1e-6:
            extrema.append((x0, f(x0), "minimum"))
        elif fpp < -1e-6:
            extrema.append((x0, f(x0), "maximum"))
        # fpp ≈ 0 → point de selle ou inflexion, on ignore
    return extrema


def trouver_wendepunkte(f: Callable, a: float, b: float) -> list[tuple[float, float]]:
    """Points d'inflexion : f'' = 0 avec changement de signe."""
    zeros_fpp = trouver_zeros(lambda x: deriv2(f, x), a, b)
    wendepunkte = []
    for x0 in zeros_fpp:
        # Vérifier changement de signe de f''
        eps = 0.01
        if deriv2(f, x0 - eps) * deriv2(f, x0 + eps) < 0:
            wendepunkte.append((x0, f(x0)))
    return wendepunkte


def analyser(f: Callable, a: float, b: float) -> Kurvendiskussion:
    """Kurvendiskussion complète sur [a, b]."""
    return Kurvendiskussion(
        nullstellen=trouver_zeros(f, a, b),
        extrema=trouver_extrema(f, a, b),
        wendepunkte=trouver_wendepunkte(f, a, b),
    )


def tracer_kurvendiskussion(
    f: Callable, a: float, b: float, nom: str = "f",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Tracé annoté avec extrema et points d'inflexion."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    kd = analyser(f, a, b)
    x = np.linspace(a, b, 500)
    y = np.array([f(xi) for xi in x])

    ax.plot(x, y, "b-", linewidth=2, label=f"${nom}(x)$")
    ax.axhline(0, color="grey", linewidth=0.5)

    # Zéros
    for z in kd.nullstellen:
        ax.plot(z, 0, "ko", markersize=8)
        ax.annotate(f"x₀={z:.2f}", (z, 0), textcoords="offset points",
                    xytext=(5, -15), fontsize=8)

    # Extrema
    for x0, y0, typ in kd.extrema:
        color = "red" if typ == "maximum" else "green"
        marker = "v" if typ == "maximum" else "^"
        ax.plot(x0, y0, marker, color=color, markersize=12,
                label=f"{typ} ({x0:.2f}, {y0:.2f})")

    # Wendepunkte
    for x0, y0 in kd.wendepunkte:
        ax.plot(x0, y0, "s", color="orange", markersize=10,
                label=f"inflexion ({x0:.2f}, {y0:.2f})")

    # f' et f''
    y_prime = [deriv(f, xi) for xi in x]
    y_pp = [deriv2(f, xi) for xi in x]
    ax.plot(x, y_prime, "r--", linewidth=1, alpha=0.5, label=f"${nom}'(x)$")
    ax.plot(x, y_pp, "g:", linewidth=1, alpha=0.5, label=f"${nom}''(x)$")

    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(f"Kurvendiskussion de ${nom}$")
    ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    # f(x) = x³ - 3x + 1
    f = lambda x: x**3 - 3*x + 1

    print("=== Kurvendiskussion : f(x) = x³ - 3x + 1 ===\n")
    kd = analyser(f, -3, 3)

    print(f"  Nullstellen : {[f'{z:.4f}' for z in kd.nullstellen]}")
    for x0, y0, typ in kd.extrema:
        print(f"  {typ:8s} en x = {x0:.4f}, f(x) = {y0:.4f}")
    for x0, y0 in kd.wendepunkte:
        print(f"  Inflexion en x = {x0:.4f}, f(x) = {y0:.4f}")

    print(f"\n=== Analyse : f(x) = sin(x)·e^(-x/3) sur [0, 15] ===\n")
    g = lambda x: np.sin(x) * np.exp(-x/3)
    kd2 = analyser(g, 0, 15)
    for x0, y0, typ in kd2.extrema:
        print(f"  {typ:8s} en x = {x0:.4f}, f(x) = {y0:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    tracer_kurvendiskussion(f, -3, 3, "x^3-3x+1", ax=axes[0])
    tracer_kurvendiskussion(g, 0, 15, "\\sin(x)e^{-x/3}", ax=axes[1])
    plt.tight_layout()
    plt.savefig("curve_analysis_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
