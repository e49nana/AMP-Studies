"""
determinant_properties.py
=========================

Propriétés du déterminant — vérification numérique.

Couvre :
    - det(Aᵀ) = det(A)
    - det(AB) = det(A)·det(B) (Produktsatz)
    - det(A⁻¹) = 1/det(A)
    - det(λA) = λⁿ det(A)
    - Effet des opérations élémentaires sur det
    - det = 0 ssi A singulière (lignes dépendantes)
    - Interprétation géométrique : volume du parallélépipède

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def verifier_proprietes(A: np.ndarray, B: np.ndarray | None = None) -> None:
    """Vérifie toutes les propriétés principales du déterminant."""
    n = A.shape[0]
    dA = np.linalg.det(A)

    print(f"  A ({n}×{n}), det(A) = {dA:.6f}\n")

    # 1. det(Aᵀ) = det(A)
    dAT = np.linalg.det(A.T)
    print(f"  1. det(Aᵀ) = {dAT:.6f} = det(A) ? {np.isclose(dA, dAT)} ✓")

    # 2. det(AB) = det(A)·det(B)
    if B is not None:
        dB = np.linalg.det(B)
        dAB = np.linalg.det(A @ B)
        print(f"  2. det(AB) = {dAB:.6f}, det(A)·det(B) = {dA*dB:.6f}, "
              f"égaux ? {np.isclose(dAB, dA*dB)} ✓")

    # 3. det(A⁻¹) = 1/det(A)
    if abs(dA) > 1e-10:
        dAinv = np.linalg.det(np.linalg.inv(A))
        print(f"  3. det(A⁻¹) = {dAinv:.6f}, 1/det(A) = {1/dA:.6f}, "
              f"égaux ? {np.isclose(dAinv, 1/dA)} ✓")

    # 4. det(λA) = λⁿ det(A)
    lam = 3.0
    d_lam_A = np.linalg.det(lam * A)
    print(f"  4. det({lam}A) = {d_lam_A:.6f}, {lam}^{n}·det(A) = {lam**n * dA:.6f}, "
          f"égaux ? {np.isclose(d_lam_A, lam**n * dA)} ✓")

    # 5. det(I) = 1
    print(f"  5. det(I_{n}) = {np.linalg.det(np.eye(n)):.0f} ✓")


def effet_operations_elementaires() -> None:
    """Montre l'effet des opérations élémentaires sur le déterminant."""
    print("\n=== Effet des opérations élémentaires ===\n")
    A = np.array([[2, 1, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    dA = np.linalg.det(A)
    print(f"  det(A) = {dA:.4f}\n")

    # Échange de lignes → signe change
    B = A.copy()
    B[[0, 1]] = B[[1, 0]]
    print(f"  Échange L₁ ↔ L₂ : det = {np.linalg.det(B):.4f} = -det(A) ✓")

    # Multiplication d'une ligne par λ → det multiplié par λ
    C = A.copy()
    C[0] *= 5
    print(f"  L₁ ← 5·L₁ : det = {np.linalg.det(C):.4f} = 5·det(A) = {5*dA:.4f} ✓")

    # Ajouter un multiple d'une ligne → det inchangé
    D = A.copy()
    D[2] += 3 * D[0]
    print(f"  L₃ ← L₃ + 3L₁ : det = {np.linalg.det(D):.4f} = det(A) = {dA:.4f} ✓")

    # Lignes identiques → det = 0
    E = A.copy()
    E[2] = E[0]
    print(f"  L₃ = L₁ : det = {np.linalg.det(E):.4f} = 0 ✓")


def interpretation_geometrique() -> None:
    """Le déterminant mesure le volume signé du parallélépipède."""
    print("\n=== Interprétation géométrique (R²) ===")
    print("  |det(A)| = aire du parallélogramme engendré par les colonnes de A.\n")

    cas = [
        ("Carré unité", np.eye(2)),
        ("Étiré ×2", np.array([[2, 0], [0, 1]], dtype=float)),
        ("Rotation 45°", np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                     [np.sin(np.pi/4), np.cos(np.pi/4)]])),
        ("Cisaillement", np.array([[1, 2], [0, 1]], dtype=float)),
        ("Réflexion", np.array([[1, 0], [0, -1]], dtype=float)),
        ("Singulière", np.array([[1, 2], [2, 4]], dtype=float)),
    ]

    for nom, A in cas:
        d = np.linalg.det(A)
        print(f"  {nom:20s} : det = {d:>8.4f}, aire = {abs(d):.4f}")


def tracer_parallelogrammes(ax: plt.Axes | None = None) -> plt.Axes:
    """Visualise le parallélogramme et son aire pour différentes matrices."""
    if ax is None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flat
    else:
        axes = [ax]

    cas = [
        ("I₂ (det=1)", np.eye(2)),
        ("Étirement (det=2)", np.array([[2, 0], [0, 1]])),
        ("Rotation (det=1)", np.array([[np.cos(0.5), -np.sin(0.5)],
                                        [np.sin(0.5), np.cos(0.5)]])),
        ("Cisaillement (det=1)", np.array([[1, 1.5], [0, 1]])),
        ("Réflexion (det=-1)", np.array([[-1, 0], [0, 1]])),
        ("Singulière (det=0)", np.array([[1, 2], [0.5, 1]])),
    ]

    for ax_i, (nom, A) in zip(axes, cas):
        # Parallélogramme
        v1, v2 = A[:, 0], A[:, 1]
        vertices = np.array([[0,0], v1, v1+v2, v2, [0,0]])
        poly = Polygon(vertices[:4], alpha=0.3, facecolor="cyan", edgecolor="blue", linewidth=2)
        ax_i.add_patch(poly)
        ax_i.quiver(0, 0, v1[0], v1[1], angles="xy", scale_units="xy", scale=1,
                     color="red", width=0.02)
        ax_i.quiver(0, 0, v2[0], v2[1], angles="xy", scale_units="xy", scale=1,
                     color="green", width=0.02)
        d = np.linalg.det(A)
        ax_i.set_title(f"{nom}\naire = |det| = {abs(d):.2f}")
        ax_i.set_aspect("equal"); ax_i.grid(True, alpha=0.3)
        lim = max(2, max(abs(v1).max(), abs(v2).max(), abs(v1+v2).max())) * 1.3
        ax_i.set_xlim(-lim, lim); ax_i.set_ylim(-lim, lim)
        ax_i.axhline(0, color="grey", linewidth=0.5)
        ax_i.axvline(0, color="grey", linewidth=0.5)

    return axes[0]


if __name__ == "__main__":
    print("=== Vérification des propriétés ===\n")
    rng = np.random.default_rng(42)
    A = rng.standard_normal((4, 4))
    B = rng.standard_normal((4, 4))
    verifier_proprietes(A, B)

    effet_operations_elementaires()
    interpretation_geometrique()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    tracer_parallelogrammes(axes.flat[0])  # will only fill first
    # Redo properly
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    cas = [
        ("I₂", np.eye(2)),
        ("Étirement", np.array([[2,0],[0,1]])),
        ("Rotation", np.array([[np.cos(0.5),-np.sin(0.5)],[np.sin(0.5),np.cos(0.5)]])),
        ("Cisaillement", np.array([[1,1.5],[0,1]])),
        ("Réflexion", np.array([[-1,0],[0,1]])),
        ("Singulière", np.array([[1,2],[0.5,1]])),
    ]
    for ax, (nom, M) in zip(axes.flat, cas):
        v1, v2 = M[:,0], M[:,1]
        verts = np.array([[0,0], v1, v1+v2, v2])
        poly = Polygon(verts, alpha=0.3, facecolor="cyan", edgecolor="blue", linewidth=2)
        ax.add_patch(poly)
        ax.quiver(0,0,v1[0],v1[1], angles="xy", scale_units="xy", scale=1, color="red", width=0.02)
        ax.quiver(0,0,v2[0],v2[1], angles="xy", scale_units="xy", scale=1, color="green", width=0.02)
        ax.set_title(f"{nom} (det={np.linalg.det(M):.2f})")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        lim = 3; ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    plt.tight_layout()
    plt.savefig("determinant_properties_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
