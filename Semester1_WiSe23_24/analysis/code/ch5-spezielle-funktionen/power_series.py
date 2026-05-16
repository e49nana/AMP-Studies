"""
power_series.py
===============

Séries entières et rayon de convergence.

Couvre :
    - Série entière Σ aₙ (x-a)^n
    - Rayon de convergence R par Hadamard : 1/R = lim sup |aₙ|^{1/n}
    - Rayon par d'Alembert : R = lim |aₙ/a_{n+1}|
    - Exemples : e^x, sin, cos, ln(1+x), arctan, (1+x)^α
    - Opérations : dérivation et intégration terme à terme
    - Convergence aux bords du disque

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from math import factorial

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Calcul du rayon de convergence
# ======================================================================

def rayon_dalembert(coeffs: np.ndarray) -> float:
    """
    R = lim |aₙ / a_{n+1}| (si la limite existe).
    """
    abs_c = np.abs(coeffs)
    ratios = abs_c[:-1] / np.maximum(abs_c[1:], 1e-300)
    # Prendre les derniers ratios stables
    tail = ratios[-5:]
    if np.std(tail) < 0.1 * np.mean(tail):
        return float(np.mean(tail))
    return float(ratios[-1])


def rayon_hadamard(coeffs: np.ndarray) -> float:
    """
    1/R = lim sup |aₙ|^{1/n} (formule de Hadamard, toujours applicable).
    """
    n = np.arange(1, len(coeffs) + 1)
    roots = np.abs(coeffs) ** (1.0 / n)
    # lim sup ≈ max des dernières valeurs
    return 1.0 / max(roots[-10:]) if max(roots[-10:]) > 0 else float("inf")


# ======================================================================
#  2. Séries entières classiques
# ======================================================================

def coeffs_exp(N: int) -> np.ndarray:
    """e^x = Σ x^n/n!, R = ∞."""
    return np.array([1.0/factorial(n) for n in range(N)])


def coeffs_sin(N: int) -> np.ndarray:
    """sin(x) = Σ (-1)^k x^{2k+1}/(2k+1)!, R = ∞."""
    c = np.zeros(N)
    for k in range(N):
        if k % 2 == 1 and (k//2) % 2 == 0:
            c[k] = 1.0 / factorial(k)
        elif k % 2 == 1:
            c[k] = -1.0 / factorial(k)
    return c


def coeffs_ln1px(N: int) -> np.ndarray:
    """ln(1+x) = Σ (-1)^{n+1} x^n/n, R = 1."""
    c = np.zeros(N)
    for n in range(1, N):
        c[n] = (-1)**(n+1) / n
    return c


def coeffs_arctan(N: int) -> np.ndarray:
    """arctan(x) = Σ (-1)^k x^{2k+1}/(2k+1), R = 1."""
    c = np.zeros(N)
    for k in range(N//2):
        n = 2*k + 1
        if n < N:
            c[n] = (-1)**k / n
    return c


def coeffs_geometrique(N: int) -> np.ndarray:
    """1/(1-x) = Σ x^n, R = 1."""
    return np.ones(N)


def coeffs_binomial(alpha: float, N: int) -> np.ndarray:
    """(1+x)^α = Σ C(α,n) x^n, R = 1 (si α ∉ N)."""
    c = np.zeros(N)
    c[0] = 1.0
    for n in range(1, N):
        c[n] = c[n-1] * (alpha - n + 1) / n
    return c


def evaluer_serie(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Évalue la série entière par Horner."""
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    for n in range(len(coeffs) - 1, -1, -1):
        result = result * x + coeffs[n]
    return result


# ======================================================================
#  3. Tracés
# ======================================================================

def tracer_convergence_serie(
    coeffs_fn, f_exact: callable, nom: str, R: float,
    Ns: tuple[int, ...] = (3, 5, 10, 20),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x_range = min(R * 1.3, 5) if np.isfinite(R) else 5
    x = np.linspace(-x_range, x_range, 400)
    ax.plot(x, f_exact(x), "k-", linewidth=2.5, label="exacte")

    for N in Ns:
        c = coeffs_fn(N)
        y = evaluer_serie(c, x)
        y = np.clip(y, -10, 10)
        ax.plot(x, y, "--", linewidth=1.5, label=f"$N={N}$")

    if np.isfinite(R):
        ax.axvline(R, color="red", linestyle=":", alpha=0.5)
        ax.axvline(-R, color="red", linestyle=":", alpha=0.5, label=f"$R = {R}$")

    ax.set_ylim(-3, 3)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(f"Série de ${nom}$ ($R = {R}$)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax


def tracer_rayons(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    series = [
        ("$e^x$", coeffs_exp, "∞"),
        ("$\\sin(x)$", coeffs_sin, "∞"),
        ("$\\ln(1+x)$", coeffs_ln1px, "1"),
        ("$\\arctan(x)$", coeffs_arctan, "1"),
        ("$1/(1-x)$", coeffs_geometrique, "1"),
        ("$(1+x)^{0.5}$", lambda N: coeffs_binomial(0.5, N), "1"),
    ]

    for nom, fn, R_str in series:
        c = fn(30)
        R_d = rayon_dalembert(c[c != 0]) if np.any(c != 0) else float("inf")
        R_h = rayon_hadamard(c)
        ax.barh(nom, R_d if np.isfinite(R_d) and R_d < 100 else 10,
                alpha=0.6, label=f"d'Alembert" if nom == "$e^x$" else "")

    ax.set_xlabel("Rayon de convergence $R$")
    ax.set_title("Rayons de convergence")
    ax.axvline(1, color="red", linestyle="--", alpha=0.3, label="$R = 1$")
    ax.grid(True, alpha=0.3, axis="x")
    return ax


if __name__ == "__main__":
    print("=== Rayons de convergence ===\n")
    print(f"  {'Série':20s} | {'R (Alembert)':>14} | {'R (Hadamard)':>14} | {'R exact':>10}")
    print("  " + "-" * 65)
    series = [
        ("e^x (R=∞)", coeffs_exp),
        ("sin(x) (R=∞)", coeffs_sin),
        ("ln(1+x) (R=1)", coeffs_ln1px),
        ("arctan(x) (R=1)", coeffs_arctan),
        ("1/(1-x) (R=1)", coeffs_geometrique),
        ("(1+x)^0.5 (R=1)", lambda N: coeffs_binomial(0.5, N)),
    ]
    for nom, fn in series:
        c = fn(50)
        c_nz = c[c != 0]
        Rd = rayon_dalembert(c_nz) if len(c_nz) > 2 else float("inf")
        Rh = rayon_hadamard(c)
        print(f"  {nom:20s} | {Rd:>14.4f} | {Rh:>14.4f} |")

    print(f"\n=== Dérivation terme à terme ===")
    print(f"  d/dx (Σ aₙxⁿ) = Σ n·aₙ·x^{{n-1}} (même rayon R)")
    c_exp = coeffs_exp(10)
    c_exp_deriv = np.array([n * c_exp[n] for n in range(1, len(c_exp))])
    x_test = 1.0
    print(f"  e^x en x=1 : série = {evaluer_serie(c_exp, np.array([x_test]))[0]:.10f}")
    print(f"  (e^x)' = e^x : série dérivée = {evaluer_serie(c_exp_deriv, np.array([x_test]))[0]:.10f}")

    print(f"\n=== Convergence aux bords ===")
    print(f"  ln(1+x) : R = 1")
    c = coeffs_ln1px(100)
    print(f"    x = +1 : série = {evaluer_serie(c, np.array([1.0]))[0]:.6f} = ln(2) = {np.log(2):.6f} ✓")
    print(f"    x = -1 : série = {evaluer_serie(c, np.array([-1.0]))[0]:.6f} → -∞ (diverge)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_convergence_serie(coeffs_ln1px, lambda x: np.log(1+x), "\\ln(1+x)", 1,
                              Ns=(3, 5, 10, 30), ax=axes[0])
    tracer_convergence_serie(coeffs_exp, np.exp, "e^x", float("inf"),
                              Ns=(3, 5, 10), ax=axes[1])
    plt.tight_layout()
    plt.savefig("power_series_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
