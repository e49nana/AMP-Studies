"""
cross_product.py
================

Produit vectoriel en R³, aires et volumes.

Couvre :
    - Produit vectoriel (Kreuzprodukt) : a × b
    - Aire du parallélogramme : ||a × b||
    - Volume du parallélépipède : |⟨a × b, c⟩| (Spatprodukt)
    - Orientation et règle de la main droite
    - Propriétés : anticommutativité, distributivité

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Produit vectoriel from-scratch :
        a × b = (a₂b₃ - a₃b₂, a₃b₁ - a₁b₃, a₁b₂ - a₂b₁).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ])


def aire_parallelogramme(a: np.ndarray, b: np.ndarray) -> float:
    """Aire = ||a × b||."""
    return float(np.linalg.norm(cross_product(a, b)))


def aire_triangle(a: np.ndarray, b: np.ndarray) -> float:
    """Aire du triangle = ||a × b|| / 2."""
    return aire_parallelogramme(a, b) / 2


def spatprodukt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Produit mixte (Spatprodukt) : ⟨a × b, c⟩ = det(a, b, c)."""
    return float(np.dot(cross_product(a, b), c))


def volume_parallelipede(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Volume = |⟨a × b, c⟩|."""
    return abs(spatprodukt(a, b, c))


def verifier_proprietes(a: np.ndarray, b: np.ndarray) -> None:
    """Vérifie les propriétés du produit vectoriel."""
    ab = cross_product(a, b)
    ba = cross_product(b, a)

    print("  Anticommutativité : a × b = -(b × a)")
    print(f"    a × b = {ab}")
    print(f"    b × a = {ba}")
    print(f"    Somme = {ab + ba} (≈ 0 ✓)")

    print("  Orthogonalité : (a × b) ⊥ a et (a × b) ⊥ b")
    print(f"    ⟨a×b, a⟩ = {np.dot(ab, a):.2e}")
    print(f"    ⟨a×b, b⟩ = {np.dot(ab, b):.2e}")

    print("  ||a × b|| = ||a||·||b||·sin θ")
    theta = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))
    expected = np.linalg.norm(a) * np.linalg.norm(b) * np.sin(theta)
    print(f"    ||a × b|| = {np.linalg.norm(ab):.6f}")
    print(f"    ||a||·||b||·sin θ = {expected:.6f}")


def tracer_cross_product(
    a: np.ndarray, b: np.ndarray, ax: plt.Axes | None = None,
) -> plt.Axes:
    """Visualise a, b et a × b en 3D."""
    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

    ab = cross_product(a, b)

    for v, nom, c in [(a, "a", "blue"), (b, "b", "red"), (ab, "a×b", "green")]:
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=c,
                  arrow_length_ratio=0.1, linewidth=2, label=f"{nom} = {v}")

    # Parallélogramme
    vertices = [[0,0,0], a.tolist(), (a+b).tolist(), b.tolist()]
    poly = Poly3DCollection([vertices], alpha=0.2, facecolor="yellow", edgecolor="grey")
    ax.add_collection3d(poly)

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"Produit vectoriel (aire = {np.linalg.norm(ab):.2f})")
    ax.legend(fontsize=9)
    return ax


if __name__ == "__main__":
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    print("=== Produit vectoriel ===")
    ab = cross_product(a, b)
    print(f"a = {a}, b = {b}")
    print(f"a × b = {ab}  (numpy: {np.cross(a, b)})")

    print(f"\n=== Aires et volumes ===")
    print(f"Aire parallélogramme = {aire_parallelogramme(a, b):.4f}")
    print(f"Aire triangle = {aire_triangle(a, b):.4f}")
    c = np.array([1.0, 0.0, 1.0])
    print(f"Spatprodukt ⟨a×b, c⟩ = {spatprodukt(a, b, c):.4f}")
    print(f"Volume parallélépipède = {volume_parallelipede(a, b, c):.4f}")

    print(f"\n=== Propriétés ===")
    verifier_proprietes(a, b)

    # Cas colinéaires : a × b = 0
    print(f"\n=== Cas colinéaire ===")
    print(f"a × (2a) = {cross_product(a, 2*a)} (= 0 ✓)")

    tracer_cross_product(np.array([2,0,0]), np.array([0,3,0]))
    plt.savefig("cross_product_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
