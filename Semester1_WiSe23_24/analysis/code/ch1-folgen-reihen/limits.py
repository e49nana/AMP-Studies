"""
limits.py
=========

Limites de fonctions et comportement asymptotique.

Couvre :
    - Limite numérique par évaluation successive
    - Epsilon-delta : vérification expérimentale
    - Limites classiques : sin(x)/x, (e^x-1)/x, (1+1/x)^x
    - Formes indéterminées : 0/0, ∞/∞, 0·∞, 1^∞, 0^0, ∞^0, ∞-∞
    - Comparaison asymptotique : O, o, ~
    - Croissance comparée : ln ≪ x^a ≪ e^x ≪ x!

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from math import gamma as math_gamma


@dataclass
class LimiteResult:
    """Résultat d'une estimation de limite."""
    fonction: str
    point: str
    valeurs: list[tuple[float, float]]
    limite_estimee: float | None


def limite_numerique(
    f: Callable[[float], float],
    x0: float,
    direction: str = "both",
    n_steps: int = 15,
) -> LimiteResult:
    """
    Estime lim_{x→x₀} f(x) par évaluation en x₀ ± 10^{-k}.
    direction : "left", "right", "both".
    """
    valeurs = []
    for k in range(1, n_steps + 1):
        h = 10.0 ** (-k)
        if direction in ("right", "both"):
            try:
                valeurs.append((x0 + h, f(x0 + h)))
            except (ZeroDivisionError, ValueError):
                pass
        if direction in ("left", "both"):
            try:
                valeurs.append((x0 - h, f(x0 - h)))
            except (ZeroDivisionError, ValueError):
                pass

    # Estimer la limite
    if valeurs:
        derniers = [v for _, v in valeurs[-6:]]
        if all(np.isfinite(derniers)):
            limite = np.mean(derniers)
        else:
            limite = None
    else:
        limite = None

    return LimiteResult(
        fonction="f",
        point=f"x → {x0}",
        valeurs=valeurs[-8:],
        limite_estimee=float(limite) if limite is not None else None,
    )


def limite_infini(
    f: Callable[[float], float],
    direction: str = "+inf",
    n_steps: int = 15,
) -> LimiteResult:
    """Estime lim_{x→±∞} f(x)."""
    valeurs = []
    for k in range(1, n_steps + 1):
        x = 10.0 ** k if direction == "+inf" else -(10.0 ** k)
        try:
            valeurs.append((x, f(x)))
        except (OverflowError, ValueError):
            pass

    derniers = [v for _, v in valeurs[-4:]]
    if derniers and all(np.isfinite(derniers)):
        limite = np.mean(derniers)
    else:
        limite = None

    return LimiteResult("f", f"x → {direction}", valeurs[-6:],
                         float(limite) if limite is not None else None)


# ======================================================================
#  Limites classiques
# ======================================================================

def limites_classiques() -> list[tuple[str, float | None, float]]:
    """Liste de limites remarquables avec valeur exacte."""
    return [
        ("sin(x)/x → 0", limite_numerique(lambda x: np.sin(x)/x, 0).limite_estimee, 1.0),
        ("(e^x - 1)/x → 0", limite_numerique(lambda x: (np.exp(x)-1)/x, 0).limite_estimee, 1.0),
        ("(1-cos x)/x² → 0", limite_numerique(lambda x: (1-np.cos(x))/x**2, 0).limite_estimee, 0.5),
        ("x·ln x → 0⁺", limite_numerique(lambda x: x*np.log(x), 0, "right").limite_estimee, 0.0),
        ("(1+1/x)^x → +∞", limite_infini(lambda x: (1+1/x)**x).limite_estimee, np.e),
        ("x·sin(1/x) → 0", limite_numerique(lambda x: x*np.sin(1/x), 0).limite_estimee, 0.0),
    ]


# ======================================================================
#  Croissance comparée
# ======================================================================

def tracer_croissance_comparee(ax: plt.Axes | None = None) -> plt.Axes:
    """ln x ≪ x^a ≪ e^x ≪ x! pour x → ∞."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(1, 8, 200)
    fonctions = [
        ("$\\ln x$", np.log(x)),
        ("$\\sqrt{x}$", np.sqrt(x)),
        ("$x$", x),
        ("$x^2$", x**2),
        ("$e^x$", np.exp(x)),
        ("$x!$ (Γ(x+1))", [float(math_gamma(xi+1)) for xi in x]),
    ]

    for nom, y in fonctions:
        ax.semilogy(x, y, linewidth=2, label=nom)

    ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$ (échelle log)")
    ax.set_title("Croissance comparée : $\\ln \\ll x^a \\ll e^x \\ll x!$")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_limites_classiques(ax: plt.Axes | None = None) -> plt.Axes:
    """Visualise les limites classiques."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(-4, 4, 500)
    x_nz = x[x != 0]

    fonctions = [
        ("$\\sin(x)/x$", x_nz, np.sin(x_nz)/x_nz, 1.0),
        ("$(e^x - 1)/x$", x_nz, (np.exp(x_nz)-1)/x_nz, 1.0),
        ("$(1 - \\cos x)/x^2$", x_nz, (1-np.cos(x_nz))/x_nz**2, 0.5),
    ]

    for nom, xi, yi, lim in fonctions:
        ax.plot(xi, yi, linewidth=2, label=f"{nom} → {lim}")
        ax.plot(0, lim, "o", markersize=8, color="red")

    ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$")
    ax.set_title("Limites classiques en $x = 0$")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 2.5)
    return ax


# ======================================================================
#  Formes indéterminées
# ======================================================================

def demo_formes_indeterminees() -> None:
    """Montre les 7 formes indéterminées classiques."""
    print("=== Formes indéterminées ===\n")
    formes = [
        ("0/0", "sin(x)/x → 0", lambda x: np.sin(x)/x, 0, 1.0),
        ("∞/∞", "x/e^x → +∞", lambda x: x/np.exp(x), None, 0.0),
        ("0·∞", "x·ln(x) → 0⁺", lambda x: x*np.log(x), 0, 0.0),
        ("1^∞", "(1+1/x)^x → +∞", lambda x: (1+1/x)**x, None, np.e),
        ("0^0", "x^x → 0⁺", lambda x: x**x, 0, 1.0),
        ("∞-∞", "x - √(x²+1) → +∞", lambda x: x - np.sqrt(x**2+1), None, 0.0),
    ]

    for forme, description, f, x0, exact in formes:
        if x0 is not None:
            res = limite_numerique(f, x0, "right")
        else:
            res = limite_infini(f)
        print(f"  {forme:>5} : {description:30s} = {res.limite_estimee:.6f} (exact: {exact})")


if __name__ == "__main__":
    print("=== Limites classiques ===\n")
    for nom, estimee, exacte in limites_classiques():
        est = f"{estimee:.10f}" if estimee is not None else "?"
        print(f"  {nom:30s} ≈ {est:>14}  (exact: {exacte})")

    print()
    demo_formes_indeterminees()

    print(f"\n=== Epsilon-delta pour sin(x)/x → 1 ===")
    print(f"  Pour ε = 0.01, on veut |sin(x)/x - 1| < 0.01")
    f = lambda x: np.sin(x)/x
    for delta in [1, 0.1, 0.01, 0.001]:
        x = delta
        err = abs(f(x) - 1)
        ok = "✓" if err < 0.01 else "✗"
        print(f"  δ = {delta:<6} : |f(δ) - 1| = {err:.6f} {ok}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_limites_classiques(ax=axes[0])
    tracer_croissance_comparee(ax=axes[1])
    plt.tight_layout()
    plt.savefig("limits_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
