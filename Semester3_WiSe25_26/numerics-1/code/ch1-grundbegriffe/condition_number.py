"""
condition_number.py
===================

Étude du conditionnement de problèmes mathématiques.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", sections 1.2.3–1.2.5.

Couvre :
    - Condition absolue et relative (Def. 1.7, 1.8)
    - Conditionnement de l'évaluation de fonctions (Satz 1.9)
    - Exemples bien et mal conditionnés (Beispiel 1.10)
    - Visualisation : petite perturbation → grande erreur

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Condition d'une fonction scalaire (Satz 1.9)
# ======================================================================

def condition_absolue(df_x: float) -> float:
    """cond_abs(f, x) = |f'(x)| (Satz 1.9)."""
    return abs(df_x)


def condition_relative(f_x: float, df_x: float, x: float) -> float:
    """
    cond_rel(f, x) = |f'(x)| · |x| / |f(x)| (Satz 1.9).

    Facteur d'amplification des erreurs relatives.
    """
    if f_x == 0:
        return float("inf")
    return abs(df_x) * abs(x) / abs(f_x)


# ======================================================================
#  2. Exemples canoniques (Beispiel 1.10)
# ======================================================================

def condition_soustraction(x1: float, x2: float) -> float:
    """
    cond_rel de x₁ - x₂ : (|x₁| + |x₂|) / |x₁ - x₂|.
    Catastrophique quand x₁ ≈ x₂.
    """
    d = x1 - x2
    if d == 0:
        return float("inf")
    return (abs(x1) + abs(x2)) / abs(d)


def condition_sqrt(x: float) -> float:
    """cond_rel(√x) = 1/2 — toujours bien conditionné."""
    return 0.5


def condition_exp(x: float) -> float:
    """cond_rel(eˣ) = |x| — mal conditionné pour |x| grand."""
    return abs(x)


def condition_log(x: float) -> float:
    """cond_rel(ln x) = 1/|ln x| — mal conditionné pour x ≈ 1."""
    if x <= 0:
        return float("inf")
    lnx = np.log(x)
    if lnx == 0:
        return float("inf")
    return 1.0 / abs(lnx)


def condition_sin(x: float) -> float:
    """cond_rel(sin x) = |x cos x / sin x| = |x / tan x|."""
    s = np.sin(x)
    if s == 0:
        return float("inf")
    return abs(x * np.cos(x) / s)


# ======================================================================
#  3. Expérience : perturbation et amplification
# ======================================================================

def experience_perturbation(
    f, df, x: float, perturbations: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Perturbe x par Δx et mesure l'erreur relative résultante sur f(x).

    Renvoie (perturbations_rel, erreurs_rel, cond_rel_theorique).
    """
    if perturbations is None:
        perturbations = np.logspace(-15, -1, 30)

    f_exact = f(x)
    cond = condition_relative(f_exact, df(x), x)
    pert_rel = perturbations / abs(x) if x != 0 else perturbations
    err_rel = np.array([abs(f(x + dx) - f_exact) / abs(f_exact)
                        for dx in perturbations])
    return pert_rel, err_rel, cond


# ======================================================================
#  4. Tracé
# ======================================================================

def tracer_condition(ax: plt.Axes | None = None) -> plt.Axes:
    """Trace cond_rel pour plusieurs fonctions en fonction de x."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    xs = np.linspace(0.01, 10, 500)
    fonctions = [
        ("$\\sqrt{x}$", [condition_sqrt(x) for x in xs]),
        ("$e^x$", [condition_exp(x) for x in xs]),
        ("$\\ln x$", [condition_log(x) for x in xs]),
        ("$\\sin x$", [condition_sin(x) for x in xs]),
    ]
    for name, conds in fonctions:
        conds = np.clip(conds, 0, 50)
        ax.plot(xs, conds, linewidth=2, label=name)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$\\text{cond}_{rel}(f, x)$")
    ax.set_title("Satz 1.9 — conditionnement relatif")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 20)
    return ax


def tracer_amplification(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre l'amplification de la soustraction x₁ - x₂ quand x₁ ≈ x₂."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x1 = 1.0
    x2s = np.linspace(0.0, 0.999, 500)
    conds = [condition_soustraction(x1, x2) for x2 in x2s]

    ax.semilogy(x2s, conds, "r-", linewidth=2)
    ax.set_xlabel("$x_2$ (avec $x_1 = 1$)")
    ax.set_ylabel("cond$_{rel}(x_1 - x_2)$")
    ax.set_title("Soustraction : explosion quand $x_2 \\to x_1$")
    ax.grid(True, which="both", alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Conditionnement relatif (Satz 1.9) ===")
    cas = [
        ("√x,  x=4", condition_sqrt(4)),
        ("eˣ,  x=1", condition_exp(1)),
        ("eˣ,  x=50", condition_exp(50)),
        ("ln x, x=2", condition_log(2)),
        ("ln x, x=1.001", condition_log(1.001)),
        ("sin x, x=π/4", condition_sin(np.pi / 4)),
        ("sin x, x=π", condition_sin(np.pi - 0.001)),
    ]
    for name, c in cas:
        print(f"  {name:20s} cond_rel = {c:.4f}")

    print("\n=== Soustraction ===")
    for x2 in [0.0, 0.9, 0.999, 0.999999]:
        print(f"  1 - {x2} : cond = {condition_soustraction(1.0, x2):.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_condition(ax=axes[0])
    tracer_amplification(ax=axes[1])
    plt.tight_layout()
    plt.savefig("condition_number_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
