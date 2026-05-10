"""
taylor_series.py
================

Séries de Taylor et développements limités.

Couvre :
    - Polynôme de Taylor T_n(x) = Σ f^(k)(a)/k! · (x-a)^k
    - Développements classiques : e^x, sin, cos, ln(1+x), 1/(1-x)
    - Reste de Taylor (Lagrange) : R_n(x) = f^{(n+1)}(ξ)/(n+1)! · (x-a)^{n+1}
    - Convergence : rayon de convergence
    - Visualisation : approximation de plus en plus précise

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from math import factorial


# ======================================================================
#  1. Développements de Taylor classiques
# ======================================================================

def taylor_exp(x: np.ndarray, n: int) -> np.ndarray:
    """T_n(e^x) = Σ_{k=0}^n x^k / k!."""
    result = np.zeros_like(x, dtype=float)
    for k in range(n + 1):
        result += x**k / factorial(k)
    return result


def taylor_sin(x: np.ndarray, n: int) -> np.ndarray:
    """T_n(sin x) = Σ_{k=0}^n (-1)^k x^{2k+1} / (2k+1)!."""
    result = np.zeros_like(x, dtype=float)
    for k in range(n + 1):
        result += (-1)**k * x**(2*k + 1) / factorial(2*k + 1)
    return result


def taylor_cos(x: np.ndarray, n: int) -> np.ndarray:
    """T_n(cos x) = Σ_{k=0}^n (-1)^k x^{2k} / (2k)!."""
    result = np.zeros_like(x, dtype=float)
    for k in range(n + 1):
        result += (-1)**k * x**(2*k) / factorial(2*k)
    return result


def taylor_ln1px(x: np.ndarray, n: int) -> np.ndarray:
    """T_n(ln(1+x)) = Σ_{k=1}^n (-1)^{k+1} x^k / k. Rayon = 1."""
    result = np.zeros_like(x, dtype=float)
    for k in range(1, n + 1):
        result += (-1)**(k + 1) * x**k / k
    return result


def taylor_arctan(x: np.ndarray, n: int) -> np.ndarray:
    """T_n(arctan x) = Σ_{k=0}^n (-1)^k x^{2k+1} / (2k+1). Rayon = 1."""
    result = np.zeros_like(x, dtype=float)
    for k in range(n + 1):
        result += (-1)**k * x**(2*k + 1) / (2*k + 1)
    return result


def taylor_generique(f_derivees: list[float], a: float, x: np.ndarray, n: int) -> np.ndarray:
    """
    T_n f(x) autour de a, avec f_derivees = [f(a), f'(a), f''(a), ...].
    """
    result = np.zeros_like(x, dtype=float)
    for k in range(min(n + 1, len(f_derivees))):
        result += f_derivees[k] / factorial(k) * (x - a)**k
    return result


# ======================================================================
#  2. Erreur de Taylor
# ======================================================================

def erreur_taylor(f_exact: np.ndarray, taylor_approx: np.ndarray) -> np.ndarray:
    """Erreur absolue |f(x) - T_n(x)|."""
    return np.abs(f_exact - taylor_approx)


# ======================================================================
#  3. Formules remarquables de π
# ======================================================================

def pi_leibniz(n: int) -> float:
    """π/4 = 1 - 1/3 + 1/5 - 1/7 + ... (arctan(1))."""
    return 4 * sum((-1)**k / (2*k + 1) for k in range(n))


def pi_machin(n: int) -> float:
    """π/4 = 4·arctan(1/5) - arctan(1/239) (formule de Machin)."""
    at5 = sum((-1)**k / ((2*k+1) * 5**(2*k+1)) for k in range(n))
    at239 = sum((-1)**k / ((2*k+1) * 239**(2*k+1)) for k in range(n))
    return 4 * (4*at5 - at239)


# ======================================================================
#  4. Tracés
# ======================================================================

def tracer_taylor_convergence(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre T_1, T_3, T_5, ... convergeant vers sin(x)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    x = np.linspace(-2*np.pi, 2*np.pi, 300)
    ax.plot(x, np.sin(x), "k-", linewidth=2.5, label="$\\sin(x)$")

    for n in [1, 3, 5, 9, 15]:
        ax.plot(x, taylor_sin(x, n), "--", linewidth=1.5,
                label=f"$T_{{{2*n+1}}}$")

    ax.set_ylim(-3, 3)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("Taylor de $\\sin(x)$ : convergence avec l'ordre")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_erreur_taylor(ax: plt.Axes | None = None) -> plt.Axes:
    """Erreur max de Taylor pour e^x sur [-1, 1] en fonction de n."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(-1, 1, 500)
    ns = range(1, 20)
    erreurs = [np.max(erreur_taylor(np.exp(x), taylor_exp(x, n))) for n in ns]

    ax.semilogy(list(ns), erreurs, "bo-", markersize=5)
    ax.set_xlabel("ordre $n$"); ax.set_ylabel("$\\max|e^x - T_n(x)|$ sur $[-1,1]$")
    ax.set_title("Erreur de Taylor pour $e^x$ : convergence factorielle")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_rayon_convergence(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre que ln(1+x) diverge pour |x| > 1 (rayon = 1)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(-0.95, 1.5, 300)
    ax.plot(x, np.log(1 + x), "k-", linewidth=2.5, label="$\\ln(1+x)$")
    for n in [3, 5, 10, 20]:
        y = taylor_ln1px(x, n)
        y = np.clip(y, -5, 5)
        ax.plot(x, y, "--", linewidth=1.5, label=f"$T_{{{n}}}$")

    ax.axvline(1, color="red", linestyle=":", alpha=0.5, label="rayon $R = 1$")
    ax.axvline(-1, color="red", linestyle=":", alpha=0.5)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("$\\ln(1+x)$ : rayon de convergence $R = 1$")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Développements de Taylor en x = 0 ===")
    x_test = 1.0
    for nom, f_exact, f_taylor in [
        ("e^1", np.e, [taylor_exp(np.array([x_test]), n)[0] for n in range(10)]),
        ("sin(1)", np.sin(1), [taylor_sin(np.array([x_test]), n)[0] for n in range(8)]),
        ("ln(2)", np.log(2), [taylor_ln1px(np.array([x_test]), n)[0] for n in range(1,15)]),
    ]:
        print(f"\n  {nom} = {f_exact:.15f}")
        for k, val in enumerate(f_taylor[:6]):
            print(f"    T_{k:>2} = {val:>18.15f}  (err = {abs(val - f_exact):.2e})")

    print(f"\n=== Approximation de π ===")
    for n in [10, 100, 1000]:
        print(f"  Leibniz({n:>4}) = {pi_leibniz(n):.10f}")
    print(f"  Machin(20)   = {pi_machin(20):.15f}")
    print(f"  π exact      = {np.pi:.15f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_taylor_convergence(ax=axes[0])
    tracer_erreur_taylor(ax=axes[1])
    tracer_rayon_convergence(ax=axes[2])
    plt.tight_layout()
    plt.savefig("taylor_series_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
