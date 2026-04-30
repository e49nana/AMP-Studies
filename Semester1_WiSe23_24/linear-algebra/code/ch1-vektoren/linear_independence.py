"""
linear_independence.py
======================

Indépendance linéaire, rang, base et dimension.

Couvre :
    - Test d'indépendance linéaire (par échelonnement)
    - Rang d'une famille de vecteurs
    - Extraction d'une base d'un ensemble de vecteurs
    - Coordonnées dans une base donnée
    - Visualisation en 2D : vecteurs indépendants vs dépendants

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def echelonner(A: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, list[int]]:
    """
    Forme échelonnée par lignes avec pivot partiel.

    Renvoie (R, pivots) où pivots[i] = indice de la colonne du pivot
    de la i-ème ligne non nulle.
    """
    A = np.asarray(A, dtype=float).copy()
    m, n = A.shape
    pivots = []
    row = 0

    for col in range(n):
        # Chercher le pivot dans la colonne
        i_max = row + np.argmax(np.abs(A[row:, col]))
        if abs(A[i_max, col]) < tol:
            continue
        # Échange de lignes
        A[[row, i_max]] = A[[i_max, row]]
        # Normalisation (optionnel pour le rang)
        A[row] /= A[row, col]
        # Élimination vers le bas
        for i in range(row + 1, m):
            A[i] -= A[i, col] * A[row]
        pivots.append(col)
        row += 1

    return A, pivots


def rang(vecteurs: list[np.ndarray]) -> int:
    """Rang d'une famille de vecteurs = nombre de pivots."""
    A = np.row_stack(vecteurs)
    _, pivots = echelonner(A)
    return len(pivots)


def sont_lineairement_independants(vecteurs: list[np.ndarray]) -> bool:
    """Les vecteurs sont indépendants ssi rang = nombre de vecteurs."""
    return rang(vecteurs) == len(vecteurs)


def extraire_base(vecteurs: list[np.ndarray]) -> list[np.ndarray]:
    """
    Extrait une base (sous-famille maximale indépendante)
    à partir des colonnes pivots de la forme échelonnée.
    """
    A = np.column_stack(vecteurs)
    _, pivots = echelonner(A.T)
    return [vecteurs[i] for i in pivots]


def coordonnees_dans_base(v: np.ndarray, base: list[np.ndarray]) -> np.ndarray:
    """
    Trouve les coordonnées de v dans la base donnée.
    Résout B·c = v avec B = [b₁ | b₂ | ... | bₖ].
    """
    B = np.column_stack(base)
    return np.linalg.solve(B, v)


def tracer_independance_2d(ax: plt.Axes | None = None) -> plt.Axes:
    """Compare vecteurs indépendants et dépendants en R²."""
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        axes = [ax, ax]

    # Indépendants
    v1 = np.array([2, 1])
    v2 = np.array([-1, 2])
    axes[0].quiver(0, 0, v1[0], v1[1], angles="xy", scale_units="xy", scale=1,
                   color="blue", width=0.015, label=f"v₁ = {v1}")
    axes[0].quiver(0, 0, v2[0], v2[1], angles="xy", scale_units="xy", scale=1,
                   color="red", width=0.015, label=f"v₂ = {v2}")
    # Montrer qu'ils couvrent R²
    for s in np.linspace(-1, 1, 5):
        for t in np.linspace(-1, 1, 5):
            pt = s*v1 + t*v2
            axes[0].plot(pt[0], pt[1], "k.", markersize=2, alpha=0.3)
    axes[0].set_title("Indépendants → engendrent R²")
    axes[0].set_aspect("equal"); axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=9)
    axes[0].set_xlim(-4, 4); axes[0].set_ylim(-4, 4)

    # Dépendants
    if ax is None:
        w1 = np.array([2, 1])
        w2 = np.array([4, 2])  # = 2·w1
        axes[1].quiver(0, 0, w1[0], w1[1], angles="xy", scale_units="xy", scale=1,
                       color="blue", width=0.015, label=f"w₁ = {w1}")
        axes[1].quiver(0, 0, w2[0], w2[1], angles="xy", scale_units="xy", scale=1,
                       color="red", width=0.015, label=f"w₂ = {w2} = 2w₁")
        for t in np.linspace(-1.5, 1.5, 20):
            pt = t * w1
            axes[1].plot(pt[0], pt[1], "k.", markersize=3, alpha=0.5)
        axes[1].set_title("Dépendants → n'engendrent qu'une droite")
        axes[1].set_aspect("equal"); axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=9)
        axes[1].set_xlim(-5, 5); axes[1].set_ylim(-3, 3)

    return axes[0]


if __name__ == "__main__":
    print("=== Indépendance linéaire ===")
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    v3 = np.array([7, 8, 9])
    print(f"v₁ = {v1}, v₂ = {v2}, v₃ = {v3}")
    print(f"Rang = {rang([v1, v2, v3])} (3 vecteurs en R³)")
    print(f"Indépendants ? {sont_lineairement_independants([v1, v2, v3])}")
    print(f"→ v₃ = 2v₂ - v₁ : dépendants car v₃ est combinaison des deux premiers.\n")

    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])
    print(f"Base canonique : rang = {rang([v1, v2, v3])}")
    print(f"Indépendants ? {sont_lineairement_independants([v1, v2, v3])}\n")

    print("=== Extraction de base ===")
    w1 = np.array([1, 2])
    w2 = np.array([2, 4])
    w3 = np.array([0, 1])
    base = extraire_base([w1, w2, w3])
    print(f"Vecteurs : {[w.tolist() for w in [w1, w2, w3]]}")
    print(f"Base extraite : {[b.tolist() for b in base]}")

    print(f"\n=== Coordonnées ===")
    base = [np.array([1, 1]), np.array([1, -1])]
    v = np.array([3, 1])
    c = coordonnees_dans_base(v, base)
    print(f"v = {v} dans la base {[b.tolist() for b in base]}")
    print(f"Coordonnées : {c}")
    print(f"Vérification : {c[0]}·b₁ + {c[1]}·b₂ = {c[0]*base[0] + c[1]*base[1]}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tracer_independance_2d.__wrapped__ = None  # skip wrapper
    # Manual plot
    v1 = np.array([2, 1]); v2 = np.array([-1, 2])
    axes[0].quiver(0, 0, v1[0], v1[1], angles="xy", scale_units="xy", scale=1,
                   color="blue", width=0.015, label="v₁ (indép.)")
    axes[0].quiver(0, 0, v2[0], v2[1], angles="xy", scale_units="xy", scale=1,
                   color="red", width=0.015, label="v₂ (indép.)")
    axes[0].set_aspect("equal"); axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[0].set_title("Indépendants"); axes[0].set_xlim(-3,3); axes[0].set_ylim(-2,3)

    w1 = np.array([2, 1]); w2 = np.array([4, 2])
    axes[1].quiver(0, 0, w1[0], w1[1], angles="xy", scale_units="xy", scale=1,
                   color="blue", width=0.015, label="w₁")
    axes[1].quiver(0, 0, w2[0], w2[1], angles="xy", scale_units="xy", scale=1,
                   color="red", width=0.015, label="w₂ = 2w₁ (dép.)")
    axes[1].set_aspect("equal"); axes[1].grid(True, alpha=0.3); axes[1].legend()
    axes[1].set_title("Dépendants"); axes[1].set_xlim(-1,5); axes[1].set_ylim(-1,3)

    plt.tight_layout()
    plt.savefig("linear_independence_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
