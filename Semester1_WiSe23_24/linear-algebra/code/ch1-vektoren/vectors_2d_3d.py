"""
vectors_2d_3d.py
================

Opérations vectorielles fondamentales en R² et R³.

Couvre :
    - Addition, soustraction, multiplication scalaire
    - Norme et normalisation
    - Combinaisons linéaires
    - Visualisation 2D et 3D

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Vecteur:
    """Vecteur de R^n avec opérations from-scratch."""

    def __init__(self, *composantes: float) -> None:
        self.v = np.array(composantes, dtype=float)
        self.dim = len(self.v)

    def __repr__(self) -> str:
        return f"Vecteur({', '.join(f'{x:.4g}' for x in self.v)})"

    def __add__(self, other: Vecteur) -> Vecteur:
        return Vecteur(*(self.v + other.v))

    def __sub__(self, other: Vecteur) -> Vecteur:
        return Vecteur(*(self.v - other.v))

    def __mul__(self, scalar: float) -> Vecteur:
        return Vecteur(*(scalar * self.v))

    def __rmul__(self, scalar: float) -> Vecteur:
        return self.__mul__(scalar)

    def __neg__(self) -> Vecteur:
        return Vecteur(*(-self.v))

    def norme(self) -> float:
        """||v||₂ = √(Σ vᵢ²)."""
        return float(np.sqrt(np.sum(self.v**2)))

    def normaliser(self) -> Vecteur:
        """v / ||v||."""
        n = self.norme()
        if n == 0:
            raise ValueError("Impossible de normaliser le vecteur nul.")
        return Vecteur(*(self.v / n))

    @staticmethod
    def combinaison_lineaire(scalaires: list[float], vecteurs: list[Vecteur]) -> Vecteur:
        """α₁v₁ + α₂v₂ + ... + αₖvₖ."""
        result = scalaires[0] * vecteurs[0]
        for s, v in zip(scalaires[1:], vecteurs[1:]):
            result = result + s * v
        return result


# ======================================================================
#  Visualisation
# ======================================================================

def tracer_vecteurs_2d(
    vecteurs: list[tuple[str, Vecteur]],
    origine: tuple[float, float] = (0, 0),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Trace des vecteurs 2D avec flèches."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(vecteurs)))
    for (nom, v), c in zip(vecteurs, colors):
        ax.quiver(origine[0], origine[1], v.v[0], v.v[1],
                  angles="xy", scale_units="xy", scale=1,
                  color=c, label=f"{nom} = {v}", width=0.015)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Vecteurs en R²")
    # Auto-limites
    all_coords = [v.v for _, v in vecteurs]
    lim = max(max(abs(c)) for c in all_coords) * 1.3
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    return ax


def tracer_vecteurs_3d(
    vecteurs: list[tuple[str, Vecteur]],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Trace des vecteurs 3D."""
    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 1, len(vecteurs)))
    for (nom, v), c in zip(vecteurs, colors):
        ax.quiver(0, 0, 0, v.v[0], v.v[1], v.v[2],
                  color=c, label=f"{nom}", arrow_length_ratio=0.1)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Vecteurs en R³")
    ax.legend(fontsize=9)
    return ax


def tracer_combinaison_lineaire(ax: plt.Axes | None = None) -> plt.Axes:
    """Visualise une combinaison linéaire en R²."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    v1 = Vecteur(2, 1)
    v2 = Vecteur(-1, 2)
    alpha, beta = 1.5, 0.8
    result = Vecteur.combinaison_lineaire([alpha, beta], [v1, v2])

    ax.quiver(0, 0, v1.v[0], v1.v[1], angles="xy", scale_units="xy", scale=1,
              color="blue", width=0.012, label=f"v₁ = {v1}")
    ax.quiver(0, 0, v2.v[0], v2.v[1], angles="xy", scale_units="xy", scale=1,
              color="red", width=0.012, label=f"v₂ = {v2}")
    # αv₁
    av1 = alpha * v1
    ax.quiver(0, 0, av1.v[0], av1.v[1], angles="xy", scale_units="xy", scale=1,
              color="blue", alpha=0.4, width=0.008,
              label=f"{alpha}v₁")
    # βv₂ décalé
    bv2 = beta * v2
    ax.quiver(av1.v[0], av1.v[1], bv2.v[0], bv2.v[1],
              angles="xy", scale_units="xy", scale=1,
              color="red", alpha=0.4, width=0.008)
    # Résultat
    ax.quiver(0, 0, result.v[0], result.v[1], angles="xy", scale_units="xy", scale=1,
              color="green", width=0.015, label=f"{alpha}v₁ + {beta}v₂ = {result}")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"Combinaison linéaire : {alpha}v₁ + {beta}v₂")
    ax.set_xlim(-2, 5); ax.set_ylim(-1, 4)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    return ax


if __name__ == "__main__":
    print("=== Opérations vectorielles ===")
    a = Vecteur(3, -4)
    b = Vecteur(1, 2)
    print(f"a = {a}, b = {b}")
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"2·a = {2 * a}")
    print(f"||a|| = {a.norme()}")
    print(f"â = {a.normaliser()}, ||â|| = {a.normaliser().norme():.10f}")

    print(f"\n=== Combinaison linéaire ===")
    v1, v2 = Vecteur(2, 1), Vecteur(-1, 2)
    cl = Vecteur.combinaison_lineaire([1.5, 0.8], [v1, v2])
    print(f"1.5·{v1} + 0.8·{v2} = {cl}")

    print(f"\n=== 3D ===")
    u = Vecteur(1, 0, 0)
    v = Vecteur(0, 1, 0)
    w = Vecteur(0, 0, 1)
    print(f"e₁ = {u}, e₂ = {v}, e₃ = {w}")
    print(f"e₁ + e₂ + e₃ = {u + v + w}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tracer_vecteurs_2d([("a", a), ("b", b), ("a+b", a + b)], ax=axes[0])
    tracer_combinaison_lineaire(ax=axes[1])
    plt.tight_layout()
    plt.savefig("vectors_2d_3d_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
