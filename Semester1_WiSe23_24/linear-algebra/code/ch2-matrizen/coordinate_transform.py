"""
coordinate_transform.py
=======================

Changement de base et matrice de passage.

Couvre :
    - Coordonnées d'un vecteur dans une base B
    - Matrice de passage P de B₁ à B₂
    - Changement de base pour les vecteurs : [v]_B₂ = P⁻¹ [v]_B₁
    - Changement de base pour les matrices : [A]_B₂ = P⁻¹ A P
    - Visualisation : même vecteur, différentes coordonnées

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def coordonnees(v: np.ndarray, base: list[np.ndarray]) -> np.ndarray:
    """
    Coordonnées de v dans la base B = {b₁, ..., bₙ}.
    Résout B·c = v avec B = [b₁ | ... | bₙ].
    """
    B = np.column_stack(base)
    return np.linalg.solve(B, v)


def matrice_passage(base_depart: list[np.ndarray], base_arrivee: list[np.ndarray]) -> np.ndarray:
    """
    Matrice de passage P de B₁ (départ) à B₂ (arrivée).

    Les colonnes de P sont les coordonnées des vecteurs de B₁ exprimés dans B₂ :
        P = B₂⁻¹ · B₁

    Alors [v]_{B₂} = P⁻¹ · [v]_{B₁}.
    """
    B1 = np.column_stack(base_depart)
    B2 = np.column_stack(base_arrivee)
    return np.linalg.solve(B2, B1)


def changer_base_vecteur(
    v_coords: np.ndarray,
    base_depart: list[np.ndarray],
    base_arrivee: list[np.ndarray],
) -> np.ndarray:
    """
    Convertit les coordonnées d'un vecteur de B₁ vers B₂.
    [v]_{B₂} = P⁻¹ · [v]_{B₁}.
    """
    P = matrice_passage(base_depart, base_arrivee)
    return np.linalg.solve(P, v_coords)


def changer_base_matrice(
    A: np.ndarray,
    base_depart: list[np.ndarray],
    base_arrivee: list[np.ndarray],
) -> np.ndarray:
    """
    Matrice d'une application linéaire dans une nouvelle base.
    [A]_{B₂} = P⁻¹ · A · P.
    """
    P = matrice_passage(base_depart, base_arrivee)
    P_inv = np.linalg.inv(P)
    return P_inv @ A @ P


def tracer_bases_2d(
    v: np.ndarray,
    base1: list[np.ndarray],
    base2: list[np.ndarray],
    nom1: str = "B₁",
    nom2: str = "B₂",
) -> plt.Figure:
    """Visualise un même vecteur dans deux bases différentes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, base, nom, color in [(axes[0], base1, nom1, "blue"),
                                  (axes[1], base2, nom2, "red")]:
        # Vecteurs de base
        for i, b in enumerate(base):
            ax.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale=1,
                      color=color, alpha=0.6, width=0.012,
                      label=f"{nom}[{i+1}] = ({b[0]:.1f}, {b[1]:.1f})")

        # Le vecteur v
        ax.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1,
                  color="green", width=0.015, label=f"v = ({v[0]:.1f}, {v[1]:.1f})")

        # Coordonnées
        c = coordonnees(v, base)
        ax.set_title(f"Base {nom} : [v]_{nom} = ({c[0]:.3f}, {c[1]:.3f})")

        # Grille de la base
        for s in np.arange(-3, 4):
            p1 = s * base[0] - 3 * base[1]
            p2 = s * base[0] + 3 * base[1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=0.1, linewidth=0.5)
            p1 = -3 * base[0] + s * base[1]
            p2 = 3 * base[0] + s * base[1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=0.1, linewidth=0.5)

        ax.set_aspect("equal"); ax.grid(False)
        ax.legend(fontsize=8); ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5)

    plt.suptitle("Même vecteur, bases différentes", fontsize=14)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("=== Coordonnées dans une base ===")
    # Base canonique
    e1, e2 = np.array([1, 0], dtype=float), np.array([0, 1], dtype=float)
    # Base B
    b1 = np.array([1, 1], dtype=float)
    b2 = np.array([1, -1], dtype=float)

    v = np.array([3, 1], dtype=float)
    c_canon = coordonnees(v, [e1, e2])
    c_B = coordonnees(v, [b1, b2])
    print(f"v = {v}")
    print(f"[v]_canon = {c_canon}")
    print(f"[v]_B     = {c_B}")
    print(f"Vérif : {c_B[0]}·b₁ + {c_B[1]}·b₂ = {c_B[0]*b1 + c_B[1]*b2}")

    print(f"\n=== Matrice de passage ===")
    P = matrice_passage([e1, e2], [b1, b2])
    print(f"P (canon → B) =\n{P}")
    print(f"P⁻¹ · [v]_canon = {np.linalg.solve(P, c_canon)}")
    print(f"= [v]_B = {c_B} ✓")

    print(f"\n=== Changement de base pour une matrice ===")
    A = np.array([[2, 1], [0, 3]], dtype=float)
    A_B = changer_base_matrice(A, [e1, e2], [b1, b2])
    print(f"[A]_canon =\n{A}")
    print(f"[A]_B =\n{A_B}")
    print(f"Vérif : mêmes valeurs propres ?")
    print(f"  λ(A)     = {np.sort(np.linalg.eigvals(A))}")
    print(f"  λ([A]_B) = {np.sort(np.linalg.eigvals(A_B))}")

    fig = tracer_bases_2d(v, [e1, e2], [b1, b2], "canon", "B")
    plt.savefig("coordinate_transform_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
