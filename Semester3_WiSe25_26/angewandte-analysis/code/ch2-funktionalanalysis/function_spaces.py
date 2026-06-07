"""
function_spaces.py
==================

Espaces de fonctions : L², C[a,b], convergence.

Couvre :
    - C[a,b] avec la norme sup : espace de Banach
    - L²[a,b] avec le produit scalaire intégral : espace de Hilbert
    - Convergence ponctuelle vs uniforme vs L²
    - Complétude de L² (les fonctions de Cauchy convergent)
    - Série de Fourier comme projection dans L²
    - Approximation et meilleure approximation

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Normes fonctionnelles
# ======================================================================

def norme_sup(f: Callable, a: float, b: float, n: int = 10000) -> float:
    """||f||_∞ = max_{x ∈ [a,b]} |f(x)| (norme de C[a,b])."""
    x = np.linspace(a, b, n)
    return float(np.max(np.abs(f(x))))


def norme_L2(f: Callable, a: float, b: float, n: int = 5000) -> float:
    """||f||₂ = √(∫_a^b |f(x)|² dx)."""
    x = np.linspace(a, b, n)
    return float(np.sqrt(np.trapezoid(f(x)**2, x)))


def norme_L1(f: Callable, a: float, b: float, n: int = 5000) -> float:
    """||f||₁ = ∫_a^b |f(x)| dx."""
    x = np.linspace(a, b, n)
    return float(np.trapezoid(np.abs(f(x)), x))


def distance_L2(f: Callable, g: Callable, a: float, b: float) -> float:
    """d₂(f, g) = ||f - g||₂."""
    return norme_L2(lambda x: f(x) - g(x), a, b)


def distance_sup(f: Callable, g: Callable, a: float, b: float) -> float:
    """d_∞(f, g) = ||f - g||_∞."""
    return norme_sup(lambda x: f(x) - g(x), a, b)


# ======================================================================
#  2. Types de convergence
# ======================================================================

def demo_convergence_types():
    """
    f_n(x) = x^n sur [0, 1].
    - Converge ponctuellement vers f(x) = 0 (x < 1), f(1) = 1
    - Ne converge PAS uniformément (||f_n||_∞ = 1 pour tout n)
    - Converge en L² : ||f_n||₂ = 1/√(2n+1) → 0
    """
    print("=== Convergence de f_n(x) = x^n sur [0,1] ===\n")
    for n in [1, 5, 10, 50, 100]:
        f_n = lambda x, n=n: x**n
        n_sup = norme_sup(f_n, 0, 1)
        n_L2 = norme_L2(f_n, 0, 1)
        n_L2_exact = 1 / np.sqrt(2*n + 1)
        print(f"  n={n:>3} : ||f_n||_∞ = {n_sup:.4f}, ||f_n||₂ = {n_L2:.6f} "
              f"(exact: {n_L2_exact:.6f})")

    print(f"\n  → ||f_n||_∞ = 1 toujours → PAS de convergence uniforme")
    print(f"  → ||f_n||₂ → 0 → convergence en L²")
    print(f"  → f_n(0.5) = 0.5^n → 0 → convergence ponctuelle (sauf en x=1)")


# ======================================================================
#  3. Fourier comme projection L²
# ======================================================================

def coefficients_fourier_L2(f: Callable, N: int, a: float = -np.pi,
                              b: float = np.pi) -> tuple[float, np.ndarray, np.ndarray]:
    """Coefficients de Fourier = projections sur la base ON {1, cos(nx), sin(nx)}."""
    L = (b - a) / 2
    x = np.linspace(a, b, 5000)
    fx = f(x)

    a0 = np.trapezoid(fx, x) / (b - a) * 2
    an = np.zeros(N)
    bn = np.zeros(N)
    for n in range(1, N + 1):
        an[n-1] = np.trapezoid(fx * np.cos(n * np.pi * x / L), x) / L
        bn[n-1] = np.trapezoid(fx * np.sin(n * np.pi * x / L), x) / L

    return a0, an, bn


def somme_partielle_fourier(x: np.ndarray, a0: float, an: np.ndarray,
                              bn: np.ndarray, L: float = np.pi) -> np.ndarray:
    """S_N(x) = a₀/2 + Σ(aₙcos + bₙsin)."""
    result = np.full_like(x, a0 / 2, dtype=float)
    for n in range(1, len(an) + 1):
        result += an[n-1] * np.cos(n * np.pi * x / L)
        result += bn[n-1] * np.sin(n * np.pi * x / L)
    return result


def erreur_fourier_L2(f: Callable, N_max: int = 50) -> list[tuple[int, float]]:
    """||f - S_N||₂ en fonction de N."""
    erreurs = []
    for N in range(1, N_max + 1):
        a0, an, bn = coefficients_fourier_L2(f, N)
        S_N = lambda x, a0=a0, an=an, bn=bn: somme_partielle_fourier(x, a0, an, bn)
        err = distance_L2(f, S_N, -np.pi, np.pi)
        erreurs.append((N, err))
    return erreurs


# ======================================================================
#  4. Complétude
# ======================================================================

def demo_completude_L2():
    """
    L² est complet : toute suite de Cauchy converge.
    C[a,b] avec ||·||₂ n'est PAS complet.
    Exemple : f_n → fonction escalier (pas continue) en L².
    """
    print("\n=== Complétude ===\n")
    print("  L²[a,b] est complet (Hilbert) ✓")
    print("  C[a,b] avec ||·||₂ n'est PAS complet ✗")
    print("  C[a,b] avec ||·||_∞ EST complet (Banach) ✓\n")

    # Suite qui converge en L² vers une discontinuité
    for n in [5, 20, 100]:
        f_n = lambda x, n=n: np.tanh(n * x)  # → signe(x) en L²
        err = distance_L2(f_n, lambda x: np.sign(x), -1, 1)
        print(f"  tanh({n:>3}·x) vs sign(x) : ||·||₂ = {err:.6f}")
    print(f"  → Converge vers sign(x) en L² (qui n'est pas dans C[a,b])")


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_convergences(ax: plt.Axes | None = None) -> plt.Axes:
    """f_n = x^n : convergence ponctuelle et L² mais pas uniforme."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(0, 1, 300)
    for n in [1, 2, 5, 10, 50]:
        ax.plot(x, x**n, linewidth=1.5, label=f"$x^{{{n}}}$")

    ax.plot([0, 1, 1], [0, 0, 1], "k--", linewidth=2, label="limite ponctuelle")
    ax.set_xlabel("$x$"); ax.set_ylabel("$f_n(x)$")
    ax.set_title("$f_n = x^n$ : conv. ponctuelle mais pas uniforme")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_fourier_L2(ax: plt.Axes | None = None) -> plt.Axes:
    """Erreur L² de la série de Fourier en fonction de N."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Signal carré
    f = lambda x: np.sign(np.sin(x))
    erreurs = erreur_fourier_L2(f, 40)
    ns, errs = zip(*erreurs)

    ax.semilogy(ns, errs, "bo-", markersize=4, linewidth=1.5)
    ax.set_xlabel("$N$ (termes de Fourier)")
    ax.set_ylabel("$\\|f - S_N\\|_{L^2}$")
    ax.set_title("Fourier = meilleure approximation dans $L^2$")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_normes_comparaison(ax: plt.Axes | None = None) -> plt.Axes:
    """Compare ||f_n||_∞ et ||f_n||₂ pour x^n."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ns = range(1, 60)
    n_sup = [1.0 for _ in ns]  # ||x^n||_∞ = 1 toujours
    n_L2 = [1/np.sqrt(2*n+1) for n in ns]

    ax.plot(list(ns), n_sup, "r-", linewidth=2, label="$\\|x^n\\|_\\infty = 1$")
    ax.plot(list(ns), n_L2, "b-", linewidth=2, label="$\\|x^n\\|_2 = 1/\\sqrt{2n+1}$")
    ax.set_xlabel("$n$"); ax.set_ylabel("norme")
    ax.set_title("$\\|\\cdot\\|_\\infty$ vs $\\|\\cdot\\|_2$ pour $f_n = x^n$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Normes fonctionnelles ===\n")
    f = np.sin
    print(f"  f = sin sur [0, π]")
    print(f"  ||sin||_∞ = {norme_sup(f, 0, np.pi):.6f} (exact: 1)")
    print(f"  ||sin||₂  = {norme_L2(f, 0, np.pi):.6f} (exact: √(π/2) = {np.sqrt(np.pi/2):.6f})")
    print(f"  ||sin||₁  = {norme_L1(f, 0, np.pi):.6f} (exact: 2)")

    demo_convergence_types()
    demo_completude_L2()

    print(f"\n=== Fourier dans L² ===\n")
    f = lambda x: np.sign(np.sin(x))
    for N in [1, 5, 10, 20, 40]:
        a0, an, bn = coefficients_fourier_L2(f, N)
        S_N = lambda x, a0=a0, an=an.copy(), bn=bn.copy(): somme_partielle_fourier(x, a0, an, bn)
        err = distance_L2(f, S_N, -np.pi, np.pi)
        print(f"  N={N:>2} : ||f - S_N||₂ = {err:.6f}")
    print(f"  → Fourier = projection orthogonale = meilleure approximation en L²")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_convergences(ax=axes[0])
    tracer_fourier_L2(ax=axes[1])
    tracer_normes_comparaison(ax=axes[2])
    plt.tight_layout()
    plt.savefig("function_spaces_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
