"""
dot_product.py
==============

Produit scalaire, angles, et projections orthogonales.

Couvre :
    - Produit scalaire (Skalarprodukt) : ⟨a, b⟩ = Σ aᵢbᵢ
    - Angle entre vecteurs : cos θ = ⟨a, b⟩ / (||a||·||b||)
    - Projection orthogonale : proj_b(a) = (⟨a, b⟩/⟨b, b⟩)·b
    - Orthogonalité : ⟨a, b⟩ = 0
    - Inégalité de Cauchy-Schwarz : |⟨a, b⟩| ≤ ||a||·||b||

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Produit scalaire from-scratch : ⟨a, b⟩ = Σ aᵢbᵢ."""
    return float(np.sum(np.asarray(a) * np.asarray(b)))


def angle_entre(a: np.ndarray, b: np.ndarray) -> float:
    """Angle en radians entre a et b."""
    cos_theta = dot_product(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = np.clip(cos_theta, -1, 1)
    return float(np.arccos(cos_theta))


def projection_orthogonale(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Projection de a sur b : proj_b(a) = (⟨a,b⟩/⟨b,b⟩)·b.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (dot_product(a, b) / dot_product(b, b)) * b


def composante_orthogonale(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Composante de a orthogonale à b : a - proj_b(a)."""
    return np.asarray(a, float) - projection_orthogonale(a, b)


def sont_orthogonaux(a: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> bool:
    """Teste si ⟨a, b⟩ = 0."""
    return abs(dot_product(a, b)) < tol


def verifier_cauchy_schwarz(a: np.ndarray, b: np.ndarray) -> dict:
    """Vérifie |⟨a,b⟩| ≤ ||a||·||b|| (Cauchy-Schwarz)."""
    lhs = abs(dot_product(a, b))
    rhs = np.linalg.norm(a) * np.linalg.norm(b)
    return {"lhs": lhs, "rhs": rhs, "vérifié": lhs <= rhs + 1e-12}


def tracer_projection(
    a: np.ndarray, b: np.ndarray, ax: plt.Axes | None = None,
) -> plt.Axes:
    """Visualise la projection orthogonale de a sur b."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    proj = projection_orthogonale(a, b)
    orth = composante_orthogonale(a, b)

    ax.quiver(0, 0, a[0], a[1], angles="xy", scale_units="xy", scale=1,
              color="blue", width=0.015, label=f"a = ({a[0]}, {a[1]})")
    ax.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale=1,
              color="red", width=0.015, label=f"b = ({b[0]}, {b[1]})")
    ax.quiver(0, 0, proj[0], proj[1], angles="xy", scale_units="xy", scale=1,
              color="green", width=0.012, label=f"proj_b(a) = ({proj[0]:.2f}, {proj[1]:.2f})")
    ax.quiver(proj[0], proj[1], orth[0], orth[1], angles="xy", scale_units="xy", scale=1,
              color="orange", width=0.008, label="composante ⊥")
    # Angle droit
    ax.plot([proj[0], proj[0]+orth[0]*0.15], [proj[1], proj[1]+orth[1]*0.15],
            "k-", linewidth=0.8)

    theta = angle_entre(a, b)
    ax.set_title(f"Projection orthogonale (θ = {np.degrees(theta):.1f}°)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    lim = max(max(abs(a)), max(abs(b))) * 1.3
    ax.set_xlim(-1, lim); ax.set_ylim(-1, lim)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    return ax


def tracer_angles(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre l'angle entre vecteurs pour différentes configurations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    b = np.array([3, 0])
    angles_deg = [0, 30, 60, 90, 120, 150, 180]
    colors = plt.cm.viridis(np.linspace(0, 1, len(angles_deg)))

    ax.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale=1,
              color="red", width=0.015, label="b = (3, 0)")

    for deg, c in zip(angles_deg, colors):
        rad = np.radians(deg)
        a = 2.5 * np.array([np.cos(rad), np.sin(rad)])
        ax.quiver(0, 0, a[0], a[1], angles="xy", scale_units="xy", scale=1,
                  color=c, width=0.008, label=f"θ = {deg}°, ⟨a,b⟩ = {dot_product(a,b):.2f}")

    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Produit scalaire et angle")
    ax.set_xlim(-4, 4); ax.set_ylim(-1, 4)
    return ax


if __name__ == "__main__":
    a = np.array([3.0, 4.0])
    b = np.array([1.0, 2.0])

    print("=== Produit scalaire ===")
    print(f"⟨a, b⟩ = {dot_product(a, b)}  (numpy: {np.dot(a, b)})")
    print(f"θ = {np.degrees(angle_entre(a, b)):.2f}°")
    print(f"Cauchy-Schwarz : {verifier_cauchy_schwarz(a, b)}")

    print(f"\n=== Projection ===")
    proj = projection_orthogonale(a, b)
    orth = composante_orthogonale(a, b)
    print(f"proj_b(a) = {proj}")
    print(f"a⊥ = {orth}")
    print(f"⟨proj, a⊥⟩ = {dot_product(proj, orth):.2e} (≈ 0 ✓)")
    print(f"proj + a⊥ = {proj + orth} = a ✓")

    print(f"\n=== Orthogonalité ===")
    u = np.array([1, 1])
    v = np.array([1, -1])
    print(f"⟨{u}, {v}⟩ = {dot_product(u, v)} → orthogonaux : {sont_orthogonaux(u, v)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tracer_projection(a, b, ax=axes[0])
    tracer_angles(ax=axes[1])
    plt.tight_layout()
    plt.savefig("dot_product_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
