"""
linear_transformations_2d.py
============================

Galerie de transformations linéaires en R² avec visualisations.

Couvre :
    - Rotation, réflexion, projection, cisaillement, homothétie
    - Composition de transformations
    - Effet sur le cercle unité, le carré, et une image (lettre F)
    - Animation de la déformation continue (interpolation avec I)
    - Déterminant = facteur de changement d'aire

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def reflexion(theta: float) -> np.ndarray:
    """Réflexion par rapport à la droite d'angle θ."""
    c, s = np.cos(2*theta), np.sin(2*theta)
    return np.array([[c, s], [s, -c]])


def projection(theta: float) -> np.ndarray:
    """Projection orthogonale sur la droite d'angle θ."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c**2, c*s], [c*s, s**2]])


def cisaillement(k: float, direction: str = "x") -> np.ndarray:
    if direction == "x":
        return np.array([[1, k], [0, 1]])
    return np.array([[1, 0], [k, 1]])


def homothetie(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0], [0, sy]])


# ======================================================================
#  Objets à transformer
# ======================================================================

def carre_unite() -> np.ndarray:
    """Sommets du carré unité [0,1]²."""
    return np.array([[0,0],[1,0],[1,1],[0,1],[0,0]], dtype=float).T


def lettre_F() -> np.ndarray:
    """Forme de la lettre F pour montrer l'orientation."""
    pts = np.array([
        [0, 0], [0.3, 0], [0.3, 0.4], [0.8, 0.4], [0.8, 0.6],
        [0.3, 0.6], [0.3, 0.8], [1, 0.8], [1, 1], [0, 1], [0, 0],
    ], dtype=float).T
    # Centrer
    pts[0] -= 0.5
    pts[1] -= 0.5
    return pts


def cercle_unite(n: int = 100) -> np.ndarray:
    theta = np.linspace(0, 2*np.pi, n)
    return np.array([np.cos(theta), np.sin(theta)])


# ======================================================================
#  Visualisation
# ======================================================================

def tracer_transformation(
    M: np.ndarray, nom: str, ax: plt.Axes,
) -> None:
    """Trace l'objet original et transformé."""
    F = lettre_F()
    F_trans = M @ F

    ax.fill(F[0], F[1], alpha=0.2, color="blue")
    ax.plot(F[0], F[1], "b-", linewidth=1, alpha=0.5)
    ax.fill(F_trans[0], F_trans[1], alpha=0.3, color="red")
    ax.plot(F_trans[0], F_trans[1], "r-", linewidth=2)

    # Cercle → ellipse
    C = cercle_unite()
    C_trans = M @ C
    ax.plot(C[0], C[1], "b:", alpha=0.3)
    ax.plot(C_trans[0], C_trans[1], "r:", alpha=0.5)

    d = np.linalg.det(M)
    ax.set_title(f"{nom}\ndet = {d:.2f}, aire × {abs(d):.2f}")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)


def tracer_galerie() -> plt.Figure:
    """Galerie de 8 transformations."""
    transformations = [
        ("Rotation 30°", rotation(np.pi/6)),
        ("Rotation 90°", rotation(np.pi/2)),
        ("Réflexion (axe x)", reflexion(0)),
        ("Réflexion (y = x)", reflexion(np.pi/4)),
        ("Projection (axe x)", projection(0)),
        ("Cisaillement", cisaillement(0.8)),
        ("Homothétie (2, 0.5)", homothetie(2, 0.5)),
        ("Composition : rot + cis", rotation(np.pi/6) @ cisaillement(0.5)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, (nom, M) in zip(axes.flat, transformations):
        tracer_transformation(M, nom, ax)
    plt.suptitle("Galerie de transformations linéaires en R²", fontsize=16)
    plt.tight_layout()
    return fig


def tracer_deformation_continue(M: np.ndarray, nom: str, n_frames: int = 6,
                                  ax: plt.Axes | None = None) -> plt.Axes:
    """Interpolation I → M : M(t) = (1-t)I + tM."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, n_frames))
    F = lettre_F()

    for i, t in enumerate(np.linspace(0, 1, n_frames)):
        Mt = (1 - t) * np.eye(2) + t * M
        Ft = Mt @ F
        ax.fill(Ft[0], Ft[1], alpha=0.15, color=colors[i])
        ax.plot(Ft[0], Ft[1], color=colors[i], linewidth=1.5,
                label=f"t = {t:.1f}")

    ax.set_title(f"Déformation continue : I → {nom}")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    return ax


if __name__ == "__main__":
    print("=== Galerie de transformations ===")
    noms = ["Rotation 30°", "Réflexion (x)", "Projection", "Cisaillement",
            "Homothétie (2,0.5)", "Composition"]
    matrices = [rotation(np.pi/6), reflexion(0), projection(0),
                cisaillement(0.8), homothetie(2, 0.5),
                rotation(np.pi/6) @ cisaillement(0.5)]

    for nom, M in zip(noms, matrices):
        d = np.linalg.det(M)
        print(f"  {nom:25s} : det = {d:>6.3f}, conserve l'aire ? {'oui' if abs(abs(d)-1) < 0.01 else 'non'}")

    print(f"\n=== Propriétés ===")
    print(f"  Rotation     : det = 1 (conserve aire + orientation)")
    print(f"  Réflexion    : det = -1 (conserve aire, inverse orientation)")
    print(f"  Projection   : det = 0 (réduit la dimension)")
    print(f"  Cisaillement : det = 1 (conserve aire, déforme)")
    print(f"  Homothétie   : det = sx·sy (change l'aire)")

    fig = tracer_galerie()
    fig.savefig("transformations_2d_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
