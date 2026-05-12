"""
optimization.py
===============

Optimisation 1D : trouver les extrema de fonctions.

Couvre :
    - Newton pour f'(x) = 0 (recherche de points critiques)
    - Gradient descent 1D : x_{k+1} = x_k - α f'(x_k)
    - Méthode de la section dorée (golden section search)
    - Comparaison des vitesses de convergence
    - Conditions suffisantes : f''(x*) > 0 → minimum

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def deriv(f, x, h=1e-6):
    return (f(x+h) - f(x-h)) / (2*h)


def deriv2(f, x, h=1e-4):
    return (f(x+h) - 2*f(x) + f(x-h)) / h**2


@dataclass
class OptResult:
    x: float
    fx: float
    iterations: int
    methode: str
    historique_x: list[float] = field(default_factory=list)


def newton_optimisation(
    f: Callable, x0: float, tol: float = 1e-12, n_max: int = 100,
) -> OptResult:
    """
    Newton pour optimisation : x_{k+1} = x_k - f'(x_k) / f''(x_k).
    Cherche un point critique (f' = 0). Convergence quadratique.
    """
    x = x0
    hist = [x]
    for k in range(1, n_max + 1):
        fp = deriv(f, x)
        fpp = deriv2(f, x)
        if abs(fpp) < 1e-15:
            break
        x = x - fp / fpp
        hist.append(x)
        if abs(fp) < tol:
            break
    return OptResult(x=x, fx=f(x), iterations=k, methode="Newton", historique_x=hist)


def gradient_descent(
    f: Callable, x0: float, alpha: float = 0.1,
    tol: float = 1e-10, n_max: int = 10000,
) -> OptResult:
    """
    Descente de gradient 1D : x_{k+1} = x_k - α f'(x_k).
    Convergence linéaire. α trop grand → diverge, α trop petit → lent.
    """
    x = x0
    hist = [x]
    for k in range(1, n_max + 1):
        fp = deriv(f, x)
        x = x - alpha * fp
        hist.append(x)
        if abs(fp) < tol:
            break
    return OptResult(x=x, fx=f(x), iterations=k, methode=f"GD (α={alpha})",
                     historique_x=hist)


def golden_section(
    f: Callable, a: float, b: float, tol: float = 1e-10, n_max: int = 200,
) -> OptResult:
    """
    Section dorée : recherche du minimum sur [a, b] sans dérivée.
    Réduit l'intervalle par le ratio doré φ = (√5-1)/2 à chaque pas.
    """
    phi = (np.sqrt(5) - 1) / 2
    hist = [(a + b) / 2]

    c = b - phi * (b - a)
    d = a + phi * (b - a)
    for k in range(1, n_max + 1):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        hist.append((a + b) / 2)
        if abs(b - a) < tol:
            break

    x_min = (a + b) / 2
    return OptResult(x=x_min, fx=f(x_min), iterations=k,
                     methode="Section dorée", historique_x=hist)


def tracer_optimisation(
    f: Callable, results: list[OptResult],
    intervalle: tuple, nom: str = "f",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(*intervalle, 300)
    ax.plot(x, [f(xi) for xi in x], "b-", linewidth=2, label=f"${nom}(x)$")

    colors = plt.cm.tab10(np.linspace(0, 0.5, len(results)))
    for r, c in zip(results, colors):
        xs = r.historique_x[:20]
        ax.plot(xs, [f(xi) for xi in xs], "o-", color=c, markersize=4,
                label=f"{r.methode} ({r.iterations} it.)")
        ax.plot(r.x, r.fx, "*", color=c, markersize=15)

    ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$")
    ax.set_title(f"Optimisation de ${nom}$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax


def tracer_effet_alpha(f: Callable, x0: float, ax: plt.Axes | None = None) -> plt.Axes:
    """Montre l'effet du pas α sur la convergence du gradient."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for alpha in [0.01, 0.05, 0.1, 0.3, 0.5]:
        r = gradient_descent(f, x0, alpha=alpha, n_max=200)
        vals = [f(xi) for xi in r.historique_x[:100]]
        ax.semilogy(range(len(vals)), [abs(v - r.fx) + 1e-16 for v in vals],
                    linewidth=1.5, label=f"α = {alpha}")

    ax.set_xlabel("itération"); ax.set_ylabel("$|f(x_k) - f(x^*)|$")
    ax.set_title("Effet du pas α sur la convergence")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    return ax


if __name__ == "__main__":
    f = lambda x: (x - 2)**2 + 1  # minimum en x = 2, f(2) = 1

    print("=== Minimum de f(x) = (x-2)² + 1 ===")
    r_n = newton_optimisation(f, 5.0)
    r_g = gradient_descent(f, 5.0, alpha=0.1)
    r_gs = golden_section(f, -1, 6)
    for r in [r_n, r_g, r_gs]:
        print(f"  {r.methode:20s} : x* = {r.x:.10f}, f(x*) = {r.fx:.10f}, {r.iterations} it.")

    print(f"\n=== Rosenbrock 1D : f(x) = (1-x)² + 100(x²-x)² ===")
    g = lambda x: (1-x)**2 + 100*(x**2 - x)**2
    for r in [newton_optimisation(g, -1.0), gradient_descent(g, -1.0, 0.001, n_max=50000)]:
        print(f"  {r.methode:20s} : x* = {r.x:.6f}, f(x*) = {r.fx:.2e}, {r.iterations} it.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_optimisation(f, [r_n, r_g, r_gs], (-1, 7), "(x-2)^2+1", ax=axes[0])
    tracer_effet_alpha(f, 5.0, ax=axes[1])
    plt.tight_layout()
    plt.savefig("optimization_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
