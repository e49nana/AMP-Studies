"""
lhopital.py
===========

Règle de L'Hôpital et formes indéterminées.

Couvre :
    - Forme 0/0 : lim f/g = lim f'/g' (si la limite existe)
    - Forme ∞/∞ : même règle
    - Transformation des autres formes : 0·∞, 1^∞, 0^0, ∞^0, ∞-∞
    - Application itérée de L'Hôpital
    - Vérification numérique vs calcul analytique
    - Cas où L'Hôpital ne marche pas

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def deriv(f, x, h=1e-7):
    return (f(x+h) - f(x-h)) / (2*h)


def limite_numerique(f, x0, direction="right", n_steps=15):
    """Estime la limite par valeurs successives."""
    vals = []
    for k in range(1, n_steps + 1):
        h = 10**(-k)
        x = x0 + h if direction == "right" else x0 - h
        try:
            vals.append(f(x))
        except (ZeroDivisionError, ValueError, OverflowError):
            pass
    return vals[-1] if vals else None


@dataclass
class LHopitalStep:
    """Un pas d'application de L'Hôpital."""
    numerateur: str
    denominateur: str
    limite: float | None


def lhopital_numerique(
    f: Callable, g: Callable, x0: float,
    n_applications: int = 5,
) -> list[LHopitalStep]:
    """
    Applique L'Hôpital itérativement :
        lim f/g = lim f'/g' = lim f''/g'' = ...

    Renvoie l'historique des applications.
    """
    steps = []

    # Étape 0 : f/g direct
    val = limite_numerique(lambda x: f(x)/g(x), x0)
    steps.append(LHopitalStep("f", "g", val))

    # Dérivées successives
    fn, gn = f, g
    for k in range(1, n_applications + 1):
        fn_prev, gn_prev = fn, gn
        fn = lambda x, fp=fn_prev: deriv(fp, x)
        gn = lambda x, gp=gn_prev: deriv(gp, x)

        val = limite_numerique(lambda x: fn(x)/gn(x), x0)
        steps.append(LHopitalStep(f"f{'′'*k}", f"g{'′'*k}", val))

        if val is not None and np.isfinite(val):
            break  # converge

    return steps


# ======================================================================
#  Exemples classiques
# ======================================================================

def exemples_0_sur_0() -> list[tuple[str, float, float]]:
    """Exemples de forme 0/0."""
    return [
        ("sin(x)/x → 0", limite_numerique(lambda x: np.sin(x)/x, 0), 1.0),
        ("(e^x-1)/x → 0", limite_numerique(lambda x: (np.exp(x)-1)/x, 0), 1.0),
        ("(1-cos x)/x² → 0", limite_numerique(lambda x: (1-np.cos(x))/x**2, 0), 0.5),
        ("tan(x)/x → 0", limite_numerique(lambda x: np.tan(x)/x, 0), 1.0),
        ("(x²-1)/(x-1) → 1", limite_numerique(lambda x: (x**2-1)/(x-1), 1), 2.0),
        ("x·ln(x) / (x-1) → 1", limite_numerique(lambda x: x*np.log(x)/(x-1), 1), 1.0),
    ]


def exemples_transformations() -> None:
    """Montre comment transformer les formes indéterminées."""
    print("=== Transformations des formes indéterminées ===\n")

    # 0 · ∞ → 0/0 ou ∞/∞
    print("  0·∞ : x·ln(x) → 0⁺")
    print("    = ln(x) / (1/x) [forme ∞/∞]")
    val = limite_numerique(lambda x: x*np.log(x), 0, "right")
    print(f"    L'Hôpital : (1/x) / (-1/x²) = -x → 0 ✓  (numérique: {val:.6f})\n")

    # 1^∞ → exp(...)
    print("  1^∞ : (1+1/x)^x → +∞")
    print("    = exp(x·ln(1+1/x))")
    val = limite_numerique(lambda x: (1+1/x)**x, 0, direction="right")
    vals_inf = [(1+1/n)**n for n in [10, 100, 1000, 10000]]
    print(f"    Valeurs : {[f'{v:.6f}' for v in vals_inf]} → e = {np.e:.6f}\n")

    # 0^0
    print("  0^0 : x^x → 0⁺")
    print("    = exp(x·ln x) et x·ln x → 0")
    val = limite_numerique(lambda x: x**x, 0, "right")
    print(f"    → {val:.6f} = 1 ✓\n")


def cas_lhopital_echoue() -> None:
    """Cas où L'Hôpital ne conclut pas."""
    print("=== Quand L'Hôpital échoue ===\n")
    print("  f(x)/g(x) = x·sin(1/x) / sin(x)  en x → 0")
    print("  f'(x)/g'(x) oscille sans converger")
    print("  → La limite existe (= 1) mais L'Hôpital boucle.\n")

    print("  Moralité : L'Hôpital dit 'si lim f'/g' existe alors...'")
    print("  Si la limite de f'/g' n'existe PAS, on ne peut rien conclure.")


# ======================================================================
#  Tracé
# ======================================================================

def tracer_lhopital_visuel(ax: plt.Axes | None = None) -> plt.Axes:
    """Visualise L'Hôpital : f/g ≈ f'/g' près du point."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(-2, 2, 300)
    x_nz = x[np.abs(x) > 0.01]

    f = lambda t: np.sin(t)
    g = lambda t: t

    ax.plot(x, f(x), "b-", linewidth=2, label="$f(x) = \\sin(x)$")
    ax.plot(x, g(x), "r-", linewidth=2, label="$g(x) = x$")
    ax.plot(x_nz, f(x_nz)/g(x_nz), "g--", linewidth=2,
            label="$f(x)/g(x)$", alpha=0.7)

    # Tangentes en 0
    ax.plot(x, np.cos(0)*x, "b:", alpha=0.5, label="$f'(0)·x = x$ (tangente)")
    ax.plot(x, 1*x, "r:", alpha=0.5, label="$g'(0)·x = x$ (tangente)")

    ax.plot(0, 1, "ko", markersize=10, label="$\\lim = f'(0)/g'(0) = 1$")

    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("L'Hôpital : $\\sin(x)/x \\to \\cos(0)/1 = 1$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Formes 0/0 ===\n")
    for nom, val, exact in exemples_0_sur_0():
        print(f"  {nom:30s} ≈ {val:.10f}  (exact: {exact})")

    print()
    exemples_transformations()
    cas_lhopital_echoue()

    print("=== L'Hôpital itératif sur (e^x - 1 - x) / x² → 0 ===")
    steps = lhopital_numerique(
        lambda x: np.exp(x) - 1 - x,
        lambda x: x**2,
        0,
    )
    for i, s in enumerate(steps):
        print(f"  Étape {i} : {s.numerateur}/{s.denominateur} → {s.limite}")

    tracer_lhopital_visuel()
    plt.tight_layout()
    plt.savefig("lhopital_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
