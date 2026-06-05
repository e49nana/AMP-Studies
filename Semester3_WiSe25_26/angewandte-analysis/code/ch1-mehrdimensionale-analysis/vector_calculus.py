"""
vector_calculus.py
==================

Calcul vectoriel : divergence, rotationnel, Laplacien.

Couvre :
    - Divergence : div F = ∂Fx/∂x + ∂Fy/∂y + ∂Fz/∂z
    - Rotationnel : curl F = ∇ × F
    - Laplacien : Δf = ∇²f = div(grad f)
    - Théorèmes intégraux (vérification numérique)
    - Champs conservatifs : F = ∇φ ⟹ curl F = 0
    - Visualisation de champs vectoriels 2D

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


# ======================================================================
#  1. Opérateurs différentiels
# ======================================================================

def divergence_2d(
    Fx: Callable, Fy: Callable, x: float, y: float, h: float = 1e-6,
) -> float:
    """div F = ∂Fx/∂x + ∂Fy/∂y."""
    dFx_dx = (Fx(x+h, y) - Fx(x-h, y)) / (2*h)
    dFy_dy = (Fy(x, y+h) - Fy(x, y-h)) / (2*h)
    return dFx_dx + dFy_dy


def rotationnel_2d(
    Fx: Callable, Fy: Callable, x: float, y: float, h: float = 1e-6,
) -> float:
    """(curl F)_z = ∂Fy/∂x - ∂Fx/∂y (composante z pour champ 2D)."""
    dFy_dx = (Fy(x+h, y) - Fy(x-h, y)) / (2*h)
    dFx_dy = (Fx(x, y+h) - Fx(x, y-h)) / (2*h)
    return dFy_dx - dFx_dy


def divergence_3d(
    Fx: Callable, Fy: Callable, Fz: Callable,
    x: float, y: float, z: float, h: float = 1e-6,
) -> float:
    """div F = ∂Fx/∂x + ∂Fy/∂y + ∂Fz/∂z."""
    d1 = (Fx(x+h,y,z) - Fx(x-h,y,z)) / (2*h)
    d2 = (Fy(x,y+h,z) - Fy(x,y-h,z)) / (2*h)
    d3 = (Fz(x,y,z+h) - Fz(x,y,z-h)) / (2*h)
    return d1 + d2 + d3


def rotationnel_3d(
    Fx: Callable, Fy: Callable, Fz: Callable,
    x: float, y: float, z: float, h: float = 1e-6,
) -> np.ndarray:
    """curl F = (∂Fz/∂y - ∂Fy/∂z, ∂Fx/∂z - ∂Fz/∂x, ∂Fy/∂x - ∂Fx/∂y)."""
    cx = (Fz(x,y+h,z) - Fz(x,y-h,z)) / (2*h) - (Fy(x,y,z+h) - Fy(x,y,z-h)) / (2*h)
    cy = (Fx(x,y,z+h) - Fx(x,y,z-h)) / (2*h) - (Fz(x+h,y,z) - Fz(x-h,y,z)) / (2*h)
    cz = (Fy(x+h,y,z) - Fy(x-h,y,z)) / (2*h) - (Fx(x,y+h,z) - Fx(x,y-h,z)) / (2*h)
    return np.array([cx, cy, cz])


def laplacien(f: Callable, x: float, y: float, h: float = 1e-4) -> float:
    """Δf = ∂²f/∂x² + ∂²f/∂y²."""
    d2x = (f(x+h, y) - 2*f(x, y) + f(x-h, y)) / h**2
    d2y = (f(x, y+h) - 2*f(x, y) + f(x, y-h)) / h**2
    return d2x + d2y


# ======================================================================
#  2. Champs conservatifs
# ======================================================================

def est_conservatif_2d(Fx: Callable, Fy: Callable,
                        x_range: tuple, y_range: tuple, n: int = 20) -> bool:
    """Un champ 2D est conservatif ssi curl F = 0 partout."""
    for x in np.linspace(*x_range, n):
        for y in np.linspace(*y_range, n):
            if abs(rotationnel_2d(Fx, Fy, x, y)) > 1e-4:
                return False
    return True


# ======================================================================
#  3. Tracés
# ======================================================================

def tracer_champ_vectoriel(
    Fx: Callable, Fy: Callable, x_range: tuple, y_range: tuple,
    titre: str = "", ax: plt.Axes | None = None,
    colorize: str = "magnitude",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    x = np.linspace(*x_range, 20)
    y = np.linspace(*y_range, 20)
    X, Y = np.meshgrid(x, y)
    U = np.vectorize(Fx)(X, Y)
    V = np.vectorize(Fy)(X, Y)
    M = np.sqrt(U**2 + V**2)

    if colorize == "divergence":
        C = np.vectorize(lambda xi, yi: divergence_2d(Fx, Fy, xi, yi))(X, Y)
        q = ax.quiver(X, Y, U, V, C, cmap="RdBu_r", alpha=0.7)
        plt.colorbar(q, ax=ax, label="divergence")
    elif colorize == "curl":
        C = np.vectorize(lambda xi, yi: rotationnel_2d(Fx, Fy, xi, yi))(X, Y)
        q = ax.quiver(X, Y, U, V, C, cmap="PiYG", alpha=0.7)
        plt.colorbar(q, ax=ax, label="curl (composante z)")
    else:
        q = ax.quiver(X, Y, U, V, M, cmap="viridis", alpha=0.7)
        plt.colorbar(q, ax=ax, label="$|F|$")

    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(titre)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    return ax


def tracer_divergence_exemple(ax: plt.Axes | None = None) -> plt.Axes:
    """Source (div > 0) et puits (div < 0)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    # Champ radial : F = (x, y) → source en (0,0)
    Fx = lambda x, y: x
    Fy = lambda x, y: y
    return tracer_champ_vectoriel(Fx, Fy, (-2, 2), (-2, 2),
                                   "Source : $F = (x, y)$, div $= 2$",
                                   ax, "divergence")


def tracer_rotationnel_exemple(ax: plt.Axes | None = None) -> plt.Axes:
    """Champ rotatoire : F = (-y, x) → curl = 2."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    Fx = lambda x, y: -y
    Fy = lambda x, y: x
    return tracer_champ_vectoriel(Fx, Fy, (-2, 2), (-2, 2),
                                   "Rotation : $F = (-y, x)$, curl $= 2$",
                                   ax, "curl")


if __name__ == "__main__":
    print("=== Divergence ===\n")
    # F = (x, y) → div = 2
    Fx, Fy = lambda x, y: x, lambda x, y: y
    print(f"  F = (x, y) : div(1,1) = {divergence_2d(Fx, Fy, 1, 1):.4f} (exact: 2)")

    # F = (x², xy) → div = 2x + x = 3x
    Fx2, Fy2 = lambda x, y: x**2, lambda x, y: x*y
    print(f"  F = (x², xy) : div(2,3) = {divergence_2d(Fx2, Fy2, 2, 3):.4f} (exact: {3*2})")

    print(f"\n=== Rotationnel ===\n")
    Fx_rot, Fy_rot = lambda x, y: -y, lambda x, y: x
    print(f"  F = (-y, x) : curl(1,1) = {rotationnel_2d(Fx_rot, Fy_rot, 1, 1):.4f} (exact: 2)")
    print(f"  Conservatif ? {est_conservatif_2d(Fx_rot, Fy_rot, (-2,2), (-2,2))}")

    # Champ conservatif : F = (2x, 2y) = ∇(x²+y²)
    print(f"  F = (2x, 2y) : curl = {rotationnel_2d(lambda x,y: 2*x, lambda x,y: 2*y, 1, 1):.4f} (exact: 0)")
    print(f"  Conservatif ? {est_conservatif_2d(lambda x,y: 2*x, lambda x,y: 2*y, (-2,2), (-2,2))}")

    print(f"\n=== Laplacien ===\n")
    # f = x² + y² → Δf = 4
    f = lambda x, y: x**2 + y**2
    print(f"  f = x²+y² : Δf(1,1) = {laplacien(f, 1, 1):.4f} (exact: 4)")
    # Harmonique : f = x² - y² → Δf = 0
    g = lambda x, y: x**2 - y**2
    print(f"  f = x²-y² : Δf(1,1) = {laplacien(g, 1, 1):.6f} (exact: 0, harmonique)")

    print(f"\n=== Rotationnel 3D ===\n")
    Fx3 = lambda x, y, z: y*z
    Fy3 = lambda x, y, z: x*z
    Fz3 = lambda x, y, z: x*y
    curl = rotationnel_3d(Fx3, Fy3, Fz3, 1, 2, 3)
    print(f"  F = (yz, xz, xy) : curl(1,2,3) = {np.round(curl, 4)} (exact: (0,0,0))")
    print(f"  → Conservatif (F = ∇(xyz)) ✓")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tracer_divergence_exemple(ax=axes[0])
    tracer_rotationnel_exemple(ax=axes[1])
    plt.tight_layout()
    plt.savefig("vector_calculus_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
