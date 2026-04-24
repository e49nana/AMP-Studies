"""
newton_systems.py
=================

Méthode de Newton pour les systèmes d'équations non-linéaires f(x) = 0
avec f : R^n → R^n.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", section 3.2.

Couvre :
    - Newton multivarié avec Jacobienne analytique (section 3.2.3)
    - Newton avec Jacobienne par différences finies (section 3.1.7 étendue)
    - Reproduction de l'Übung 3.10
    - Ordre de convergence expérimental (extension section 3.1.4)
    - Condition du problème cond = ||J(x*)⁻¹|| (section 3.2.2)

Algorithme (section 3.2.3) :
    1. Calculer y = f(x) et J = f'(x)
    2. Résoudre J s = -y
    3. x⁺ = x + s

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Résultat
# ======================================================================

@dataclass
class ResultatNewtonSys:
    """Résultat du Newton multivarié."""
    x: np.ndarray
    iterations: int
    converge: bool
    methode: str
    historique_x: list[np.ndarray] = field(default_factory=list)
    historique_norm_f: list[float] = field(default_factory=list)
    ordres_experimentaux: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        statut = "convergé" if self.converge else "non convergé"
        return (
            f"ResultatNewtonSys({self.methode}, {statut} "
            f"en {self.iterations} it., ||f|| = {self.historique_norm_f[-1]:.2e})"
        )


# ======================================================================
#  2. Jacobienne par différences finies
# ======================================================================

def jacobienne_df(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    h: float | None = None,
) -> np.ndarray:
    """
    Approximation de la Jacobienne par différences finies progressives.

    J_ij ≈ (f_i(x + h e_j) - f_i(x)) / h.

    Le choix de h suit la recommandation de la section 3.1.7 :
        h ≈ |x_j| · √ε_mach  (ou √ε_mach si x_j = 0).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    fx = f(x)
    m = len(fx)
    J = np.empty((m, n), dtype=float)
    eps = np.sqrt(np.finfo(float).eps)

    for j in range(n):
        # Choix de h pour la j-ème composante (section 3.1.7)
        if h is not None:
            hj = h
        else:
            hj = eps * max(abs(x[j]), 1.0)
            # Astuce du script : s'assurer que x + hj - x == hj exactement
            x_delta = x[j] + hj
            hj = x_delta - x[j]

        e_j = np.zeros(n)
        e_j[j] = hj
        J[:, j] = (f(x + e_j) - fx) / hj

    return J


# ======================================================================
#  3. Newton avec Jacobienne analytique (section 3.2.3)
# ======================================================================

def newton_systeme(
    f: Callable[[np.ndarray], np.ndarray],
    jac: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-12,
    n_max: int = 50,
) -> ResultatNewtonSys:
    """
    Newton multivarié avec Jacobienne analytique.

    À chaque pas :
        1. y = f(x),  J = jac(x)
        2. Résoudre J s = -y  (système linéaire, formule 3.13)
        3. x⁺ = x + s

    Convergence quadratique (extension du Satz 3.5) si J(x*) régulière
    et x₀ assez proche.
    """
    x = np.asarray(x0, dtype=float).copy()
    hist_x = [x.copy()]
    hist_nf = [float(np.linalg.norm(f(x), np.inf))]

    converge = False
    for k in range(1, n_max + 1):
        y = f(x)
        J = jac(x)
        try:
            s = np.linalg.solve(J, -y)  # formule 3.13
        except np.linalg.LinAlgError:
            break

        x = x + s
        hist_x.append(x.copy())
        nf = float(np.linalg.norm(f(x), np.inf))
        hist_nf.append(nf)

        if nf < tol:
            converge = True
            break

    return ResultatNewtonSys(
        x=x, iterations=k, converge=converge, methode="Newton (Jac. analytique)",
        historique_x=hist_x, historique_norm_f=hist_nf,
    )


# ======================================================================
#  4. Newton avec Jacobienne par différences finies
# ======================================================================

def newton_systeme_df(
    f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-12,
    n_max: int = 50,
) -> ResultatNewtonSys:
    """
    Newton multivarié sans fournir la Jacobienne :
    elle est approchée par différences finies à chaque pas.

    Plus d'évaluations de f par itération (n+1 au lieu de 1),
    mais pas besoin de coder J.
    """
    x = np.asarray(x0, dtype=float).copy()
    hist_x = [x.copy()]
    hist_nf = [float(np.linalg.norm(f(x), np.inf))]

    converge = False
    for k in range(1, n_max + 1):
        y = f(x)
        J = jacobienne_df(f, x)
        try:
            s = np.linalg.solve(J, -y)
        except np.linalg.LinAlgError:
            break

        x = x + s
        hist_x.append(x.copy())
        nf = float(np.linalg.norm(f(x), np.inf))
        hist_nf.append(nf)

        if nf < tol:
            converge = True
            break

    return ResultatNewtonSys(
        x=x, iterations=k, converge=converge, methode="Newton (Jac. diff. finies)",
        historique_x=hist_x, historique_norm_f=hist_nf,
    )


# ======================================================================
#  5. Ordre de convergence expérimental (extension section 3.1.4)
# ======================================================================

def ordre_experimental(
    historique_x: list[np.ndarray], x_star: np.ndarray,
) -> list[float]:
    """Ordre de convergence avec solution connue, en norme ∞."""
    e = [float(np.linalg.norm(xi - x_star, np.inf)) for xi in historique_x]
    ordres = []
    for k in range(1, len(e) - 1):
        if e[k] > 0 and e[k - 1] > 0 and e[k + 1] > 0:
            num = np.log(e[k] / e[k + 1])
            den = np.log(e[k - 1] / e[k])
            if abs(den) > 1e-30:
                ordres.append(num / den)
    return ordres


# ======================================================================
#  6. Übung 3.10 du script
# ======================================================================

def f_uebung_3_10(x: np.ndarray) -> np.ndarray:
    """
    Übung 3.10 :
        2(x₁ + x₂)² + (x₁ - x₂)² - 8 = 0
        5 x₁²      + (x₂ - 3)²    - 9 = 0
    """
    return np.array([
        2 * (x[0] + x[1])**2 + (x[0] - x[1])**2 - 8,
        5 * x[0]**2 + (x[1] - 3)**2 - 9,
    ])


def jac_uebung_3_10(x: np.ndarray) -> np.ndarray:
    """Jacobienne analytique pour l'Übung 3.10."""
    return np.array([
        [4*(x[0]+x[1]) + 2*(x[0]-x[1]),  4*(x[0]+x[1]) - 2*(x[0]-x[1])],
        [10*x[0],                          2*(x[1]-3)],
    ])


# ======================================================================
#  7. Exemples classiques
# ======================================================================

def f_rosenbrock(x: np.ndarray) -> np.ndarray:
    """
    Zéro du gradient de Rosenbrock (minimum en (1,1)) :
        f₁ = -2(1-x₁) - 400 x₁(x₂ - x₁²) = 0
        f₂ = 200(x₂ - x₁²)                 = 0
    """
    return np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2),
    ])


def jac_rosenbrock(x: np.ndarray) -> np.ndarray:
    return np.array([
        [2 - 400*(x[1] - 3*x[0]**2), -400*x[0]],
        [-400*x[0],                     200],
    ])


# ======================================================================
#  8. Tracé
# ======================================================================

def tracer_convergence_systeme(
    resultats: list[ResultatNewtonSys],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for r in resultats:
        ax.semilogy(r.historique_norm_f, "o-", label=r.methode, markersize=5)
    ax.set_xlabel("itération $k$")
    ax.set_ylabel("$\\|f(x_k)\\|_\\infty$")
    ax.set_title("Newton multivarié — convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_trajectoire_2d(
    f: Callable[[np.ndarray], np.ndarray],
    resultats: list[ResultatNewtonSys],
    xlim: tuple[float, float] = (-2, 3),
    ylim: tuple[float, float] = (-2, 4),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Trace les courbes de niveau f₁=0, f₂=0 et la trajectoire de Newton."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    X, Y = np.meshgrid(np.linspace(*xlim, 200), np.linspace(*ylim, 200))
    F1 = np.zeros_like(X)
    F2 = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            fval = f(np.array([X[i, j], Y[i, j]]))
            F1[i, j] = fval[0]
            F2[i, j] = fval[1]

    ax.contour(X, Y, F1, levels=[0], colors="blue", linewidths=1.5)
    ax.contour(X, Y, F2, levels=[0], colors="red", linewidths=1.5)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")

    for r in resultats:
        pts = np.array(r.historique_x)
        ax.plot(pts[:, 0], pts[:, 1], "o-", markersize=6, label=r.methode)

    ax.set_title("Trajectoire de Newton (bleu: $f_1=0$, rouge: $f_2=0$)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


# ======================================================================
#  Démo
# ======================================================================

if __name__ == "__main__":
    print("=== Übung 3.10 : système 2×2 ===")
    x0 = np.array([2.0, 0.0])
    res_a = newton_systeme(f_uebung_3_10, jac_uebung_3_10, x0)
    res_df = newton_systeme_df(f_uebung_3_10, x0)
    print(f"Jac. analytique : {res_a}")
    print(f"  x* = {res_a.x}")
    print(f"Jac. diff. finies : {res_df}")
    print(f"  x* = {res_df.x}")

    x_star = res_a.x
    ordres = ordre_experimental(res_a.historique_x, x_star)
    print(f"  ordres exp. : {[f'{o:.3f}' for o in ordres]}")

    print("\n=== Rosenbrock (minimum en (1,1)) ===")
    res_ros = newton_systeme(f_rosenbrock, jac_rosenbrock, np.array([-1.0, 1.0]))
    print(f"  {res_ros}")
    print(f"  x* = {res_ros.x}")

    print("\n=== Tracés ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tracer_trajectoire_2d(f_uebung_3_10, [res_a], xlim=(-1, 3), ylim=(-1, 4), ax=axes[0])
    axes[0].set_title("Übung 3.10")
    tracer_convergence_systeme([res_a, res_df], ax=axes[1])
    plt.tight_layout()
    plt.savefig("newton_systems_demo.png", dpi=120)
    print("Figure sauvegardée : newton_systems_demo.png")
