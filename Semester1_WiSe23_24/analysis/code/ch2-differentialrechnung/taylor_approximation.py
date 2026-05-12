"""
taylor_approximation.py
=======================

Approximation de Taylor et analyse d'erreur.

Couvre :
    - Polynôme de Taylor autour d'un point a quelconque
    - Reste de Lagrange : R_n(x) = f^{(n+1)}(ξ) / (n+1)! · (x-a)^{n+1}
    - Approximation de fonctions usuelles avec estimation d'erreur
    - Applications : calcul de sin/cos/exp sans bibliothèque
    - Comparaison erreur théorique vs erreur observée

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from math import factorial
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def taylor_polynomial(
    derivees: list[float], a: float, x: np.ndarray, n: int,
) -> np.ndarray:
    """
    T_n f(x; a) = Σ_{k=0}^n f^{(k)}(a)/k! · (x-a)^k.
    derivees = [f(a), f'(a), f''(a), ...].
    """
    result = np.zeros_like(x, dtype=float)
    for k in range(min(n + 1, len(derivees))):
        result += derivees[k] / factorial(k) * (x - a)**k
    return result


def reste_lagrange_borne(M: float, n: int, x: np.ndarray, a: float) -> np.ndarray:
    """
    Borne du reste : |R_n(x)| ≤ M / (n+1)! · |x-a|^{n+1}
    avec M = max |f^{(n+1)}| sur l'intervalle.
    """
    return M / factorial(n + 1) * np.abs(x - a)**(n + 1)


def approximer_sin(x: float, n_termes: int = 5) -> tuple[float, float]:
    """
    sin(x) ≈ Σ_{k=0}^{n-1} (-1)^k x^{2k+1}/(2k+1)!
    Renvoie (approximation, borne d'erreur).
    """
    result = 0.0
    for k in range(n_termes):
        result += (-1)**k * x**(2*k+1) / factorial(2*k+1)
    # Borne : le premier terme omis
    erreur = abs(x**(2*n_termes+1)) / factorial(2*n_termes+1)
    return result, erreur


def approximer_exp(x: float, n_termes: int = 10) -> tuple[float, float]:
    """
    e^x ≈ Σ_{k=0}^{n-1} x^k / k!
    Borne : |R| ≤ e^|x| · |x|^n / n!
    """
    result = sum(x**k / factorial(k) for k in range(n_termes))
    erreur = np.exp(abs(x)) * abs(x)**n_termes / factorial(n_termes)
    return result, erreur


def tracer_approximation_taylor(
    f: Callable, derivees_fn: Callable, a: float,
    intervalle: tuple, nom: str, ordres: tuple[int, ...] = (1, 2, 4, 8),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Trace f et ses approximations de Taylor T_1, T_2, ..., T_n."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    x = np.linspace(*intervalle, 300)
    ax.plot(x, f(x), "k-", linewidth=2.5, label=f"${nom}(x)$")

    for n in ordres:
        derivees = [derivees_fn(a, k) for k in range(n + 1)]
        Tn = taylor_polynomial(derivees, a, x, n)
        ax.plot(x, Tn, "--", linewidth=1.5, label=f"$T_{{{n}}}(x; a={a})$")

    ax.plot(a, f(a), "ro", markersize=10, label=f"$a = {a}$")
    ymin, ymax = f(np.array(intervalle))
    margin = max(abs(ymin), abs(ymax)) * 0.5
    ax.set_ylim(min(ymin, -1) - margin, max(ymax, 1) + margin)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(f"Approximation de Taylor de ${nom}$ autour de $a = {a}$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax


def tracer_erreur_vs_theorique(ax: plt.Axes | None = None) -> plt.Axes:
    """Compare erreur observée vs borne de Lagrange pour e^x."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x_test = 2.0  # point d'évaluation
    ns = range(1, 20)
    err_obs = []
    err_borne = []

    for n in ns:
        approx = sum(x_test**k / factorial(k) for k in range(n + 1))
        err_obs.append(abs(approx - np.exp(x_test)))
        # Borne : e^2 · 2^{n+1} / (n+1)!
        err_borne.append(np.exp(x_test) * x_test**(n+1) / factorial(n+1))

    ax.semilogy(list(ns), err_obs, "bo-", markersize=5, label="erreur observée")
    ax.semilogy(list(ns), err_borne, "r--", linewidth=2, label="borne de Lagrange")
    ax.set_xlabel("ordre $n$")
    ax.set_ylabel("erreur $|e^2 - T_n(2)|$")
    ax.set_title("Reste de Lagrange : la borne est pessimiste mais correcte")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Approximation de sin(0.5) ===")
    for n in range(1, 8):
        approx, borne = approximer_sin(0.5, n)
        exact = np.sin(0.5)
        print(f"  {n} termes : {approx:.15f}  err = {abs(approx-exact):.2e}  borne = {borne:.2e}")

    print(f"\n=== Approximation de e^2 ===")
    for n in [3, 5, 10, 15]:
        approx, borne = approximer_exp(2.0, n)
        exact = np.exp(2.0)
        print(f"  {n:>2} termes : {approx:.10f}  err = {abs(approx-exact):.2e}  borne = {borne:.2e}")

    print(f"\n=== Taylor autour de a ≠ 0 ===")
    # e^x autour de a = 1 : toutes les dérivées = e^1
    a = 1.0
    for n in [1, 3, 5]:
        derivees = [np.e] * (n + 1)
        T = taylor_polynomial(derivees, a, np.array([2.0]), n)[0]
        print(f"  T_{n}(2; a=1) = {T:.10f}  (exact = {np.exp(2):.10f})")

    # Tracés
    def sin_derivees(a, k):
        """k-ème dérivée de sin en a."""
        return np.sin(a + k * np.pi / 2)

    def exp_derivees(a, k):
        return np.exp(a)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_approximation_taylor(np.sin, sin_derivees, 0,
                                 (-2*np.pi, 2*np.pi), "\\sin", (1, 3, 5, 9), ax=axes[0])
    tracer_approximation_taylor(np.exp, exp_derivees, 0,
                                 (-3, 3), "e^x", (1, 2, 4, 8), ax=axes[1])
    tracer_erreur_vs_theorique(ax=axes[2])
    plt.tight_layout()
    plt.savefig("taylor_approximation_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
