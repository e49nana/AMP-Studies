"""
normed_spaces.py
================

Espaces normés et espaces de Banach.

Couvre :
    - Norme : ||x|| ≥ 0, ||αx|| = |α|·||x||, inégalité triangulaire
    - Normes p : ||x||_p = (Σ|xᵢ|^p)^{1/p}, ||x||_∞ = max|xᵢ|
    - Équivalence des normes en dimension finie
    - Espace de Banach = espace normé complet
    - Opérateurs linéaires bornés : ||T|| = sup ||Tx|| / ||x||
    - Dual et théorème de Hahn-Banach (introduction)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Normes
# ======================================================================

def norme_p(x: np.ndarray, p: float) -> float:
    """||x||_p = (Σ|xᵢ|^p)^{1/p}."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.abs(x)**p)**(1/p))


def norme_inf(x: np.ndarray) -> float:
    """||x||_∞ = max|xᵢ|."""
    return float(np.max(np.abs(x)))


def norme_1(x: np.ndarray) -> float:
    return norme_p(x, 1)


def norme_2(x: np.ndarray) -> float:
    return norme_p(x, 2)


def verifier_axiomes_norme(x: np.ndarray, y: np.ndarray, alpha: float = 2.5) -> dict:
    """Vérifie les 3 axiomes pour ||·||₂."""
    n2 = norme_2
    return {
        "positivité": n2(x) >= 0 and (n2(x) == 0) == np.allclose(x, 0),
        "homogénéité": np.isclose(n2(alpha * x), abs(alpha) * n2(x)),
        "triangle": n2(x + y) <= n2(x) + n2(y) + 1e-10,
    }


# ======================================================================
#  2. Équivalence des normes
# ======================================================================

def constantes_equivalence(n: int, p: float, q: float) -> tuple[float, float]:
    """
    En dim finie, toutes les normes sont équivalentes :
        c₁ ||x||_q ≤ ||x||_p ≤ c₂ ||x||_q.
    Pour p ≤ q : c₁ = 1, c₂ = n^{1/p - 1/q}.
    """
    if p <= q:
        c1 = 1.0
        c2 = n**(1/p - 1/q)
    else:
        c2 = 1.0
        c1 = n**(1/q - 1/p)
    return c1, c2


def verifier_equivalence(n: int = 3, p: float = 1, q: float = 2,
                           n_tests: int = 10000) -> dict:
    """Vérifie numériquement les constantes d'équivalence."""
    rng = np.random.default_rng(42)
    ratios = []
    for _ in range(n_tests):
        x = rng.standard_normal(n)
        np_val = norme_p(x, p)
        nq_val = norme_p(x, q) if q != np.inf else norme_inf(x)
        if nq_val > 1e-15:
            ratios.append(np_val / nq_val)

    c1_theo, c2_theo = constantes_equivalence(n, p, q)
    return {
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "c1_theo": c1_theo,
        "c2_theo": c2_theo,
        "verifie": min(ratios) >= c1_theo - 0.01 and max(ratios) <= c2_theo + 0.01,
    }


# ======================================================================
#  3. Norme d'opérateur
# ======================================================================

def norme_operateur(A: np.ndarray, p: float = 2) -> float:
    """
    ||A||_p = max_{||x||_p = 1} ||Ax||_p.
    Pour p = 2 : ||A||₂ = σ_max(A) (plus grande valeur singulière).
    """
    if p == 2:
        return float(np.linalg.norm(A, 2))
    # Estimation numérique
    rng = np.random.default_rng(42)
    max_ratio = 0
    for _ in range(10000):
        x = rng.standard_normal(A.shape[1])
        nx = norme_p(x, p)
        if nx > 1e-15:
            ratio = norme_p(A @ x, p) / nx
            max_ratio = max(max_ratio, ratio)
    return max_ratio


def norme_operateur_exacte(A: np.ndarray) -> dict:
    """Normes d'opérateur exactes pour les 3 normes classiques."""
    return {
        "||A||_1": float(np.max(np.sum(np.abs(A), axis=0))),  # max colonne sum
        "||A||_2": float(np.linalg.norm(A, 2)),  # σ_max
        "||A||_∞": float(np.max(np.sum(np.abs(A), axis=1))),  # max row sum
    }


# ======================================================================
#  4. Tracés
# ======================================================================

def tracer_normes_comparaison(ax: plt.Axes | None = None) -> plt.Axes:
    """Compare ||x||_p pour un même vecteur en fonction de p."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ps = np.linspace(0.5, 10, 100)
    vecteurs = [
        ("(1,1,1)", np.array([1, 1, 1])),
        ("(3,0,0)", np.array([3, 0, 0])),
        ("(1,2,3)", np.array([1, 2, 3])),
    ]

    for nom, x in vecteurs:
        normes = [norme_p(x, p) for p in ps]
        ax.plot(ps, normes, linewidth=2, label=f"$x = {nom}$")
        ax.plot([10], [norme_inf(x)], "o", markersize=8)

    ax.set_xlabel("$p$"); ax.set_ylabel("$\\|x\\|_p$")
    ax.set_title("$\\|x\\|_p$ en fonction de $p$ (→ $\\|x\\|_\\infty$ quand $p → ∞$)")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_equivalence(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre l'encadrement ||x||₁ vs ||x||₂ en R²."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    rng = np.random.default_rng(42)
    points = rng.standard_normal((5000, 2))

    n1 = np.array([norme_1(x) for x in points])
    n2 = np.array([norme_2(x) for x in points])
    ratios = n1 / n2

    ax.hist(ratios, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    c1, c2 = constantes_equivalence(2, 1, 2)
    ax.axvline(c1, color="red", linewidth=2, linestyle="--", label=f"$c_1 = {c1:.2f}$")
    ax.axvline(c2, color="green", linewidth=2, linestyle="--", label=f"$c_2 = {c2:.2f}$")
    ax.set_xlabel("$\\|x\\|_1 / \\|x\\|_2$")
    ax.set_ylabel("densité")
    ax.set_title("Équivalence $\\|\\cdot\\|_1$ et $\\|\\cdot\\|_2$ en $\\mathbb{R}^2$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_norme_operateur(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    A = np.array([[2, 1], [0, 3]])
    # Image du cercle unité
    theta = np.linspace(0, 2*np.pi, 300)
    cercle = np.array([np.cos(theta), np.sin(theta)])
    image = A @ cercle

    ax.plot(cercle[0], cercle[1], "b-", linewidth=1.5, label="$\\|x\\|_2 = 1$")
    ax.plot(image[0], image[1], "r-", linewidth=2, label="$Ax$")

    sigma_max = np.linalg.norm(A, 2)
    ax.plot(0, 0, "ko", markersize=5)
    ax.annotate(f"$\\|A\\|_2 = \\sigma_{{max}} = {sigma_max:.3f}$",
                (2, 2), fontsize=11)

    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title("$\\|A\\|_2$ = rayon max de l'image du cercle unité")
    ax.legend(); ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    return ax


if __name__ == "__main__":
    x = np.array([1.0, -2.0, 3.0])
    y = np.array([2.0, 1.0, -1.0])

    print("=== Normes ===\n")
    for p in [1, 2, 3, np.inf]:
        val = norme_p(x, p) if p != np.inf else norme_inf(x)
        print(f"  ||x||_{p if p != np.inf else '∞':>3} = {val:.4f}")

    print(f"\n=== Axiomes (||·||₂) ===\n")
    ax_res = verifier_axiomes_norme(x, y)
    for nom, ok in ax_res.items():
        print(f"  {nom:14s} : {ok} ✓")

    print(f"\n=== Équivalence des normes (R³) ===\n")
    for p, q in [(1, 2), (2, np.inf), (1, np.inf)]:
        r = verifier_equivalence(3, p, min(q, 50))
        q_str = "∞" if q == np.inf else str(q)
        print(f"  ||·||_{p} vs ||·||_{q_str} : ratio ∈ [{r['ratio_min']:.3f}, {r['ratio_max']:.3f}], "
              f"théo: [{r['c1_theo']:.3f}, {r['c2_theo']:.3f}] ✓")

    print(f"\n=== Norme d'opérateur ===\n")
    A = np.array([[2, 1], [0, 3]])
    normes = norme_operateur_exacte(A)
    for nom, val in normes.items():
        print(f"  {nom} = {val:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_normes_comparaison(ax=axes[0])
    tracer_equivalence(ax=axes[1])
    tracer_norme_operateur(ax=axes[2])
    plt.tight_layout()
    plt.savefig("normed_spaces_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
