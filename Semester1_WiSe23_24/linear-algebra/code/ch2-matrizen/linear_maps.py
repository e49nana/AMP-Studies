"""
linear_maps.py
==============

Applications linéaires : matrice associée, noyau, image.

Couvre :
    - Définition : f(αx + βy) = αf(x) + βf(y)
    - Construction de la matrice : A = [f(e₁) | f(e₂) | ... | f(eₙ)]
    - Noyau Kern(f) et image Bild(f)
    - Composition = multiplication matricielle
    - Exemples : rotations, projections, réflexions, cisaillement

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ApplicationLineaire:
    """Encapsule une application linéaire f : R^n → R^m."""
    matrice: np.ndarray
    nom: str = ""

    @property
    def dim_depart(self) -> int:
        return self.matrice.shape[1]

    @property
    def dim_arrivee(self) -> int:
        return self.matrice.shape[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.matrice @ np.asarray(x, float)

    def noyau_dim(self) -> int:
        return self.dim_depart - np.linalg.matrix_rank(self.matrice)

    def image_dim(self) -> int:
        return np.linalg.matrix_rank(self.matrice)

    def est_injective(self) -> bool:
        """Kern = {0} ssi injective."""
        return self.noyau_dim() == 0

    def est_surjective(self) -> bool:
        """Bild = R^m ssi surjective."""
        return self.image_dim() == self.dim_arrivee

    def est_bijective(self) -> bool:
        return self.est_injective() and self.est_surjective()

    def composer(self, other: ApplicationLineaire) -> ApplicationLineaire:
        """(f ∘ g)(x) = f(g(x)) → matrice = A_f · A_g."""
        return ApplicationLineaire(
            matrice=self.matrice @ other.matrice,
            nom=f"{self.nom} ∘ {other.nom}",
        )

    @classmethod
    def from_function(cls, f: Callable, n: int, m: int, nom: str = "") -> ApplicationLineaire:
        """Construit la matrice en appliquant f aux vecteurs de base."""
        A = np.column_stack([f(e) for e in np.eye(n)])
        return cls(matrice=A, nom=nom)


# ======================================================================
#  Exemples classiques en R²
# ======================================================================

def rotation_2d(theta: float) -> ApplicationLineaire:
    """Rotation d'angle θ (en radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return ApplicationLineaire(
        matrice=np.array([[c, -s], [s, c]]),
        nom=f"Rot({np.degrees(theta):.0f}°)",
    )


def reflexion_2d(theta: float) -> ApplicationLineaire:
    """Réflexion par rapport à la droite d'angle θ avec l'axe x."""
    c, s = np.cos(2*theta), np.sin(2*theta)
    return ApplicationLineaire(
        matrice=np.array([[c, s], [s, -c]]),
        nom=f"Ref({np.degrees(theta):.0f}°)",
    )


def projection_2d(theta: float) -> ApplicationLineaire:
    """Projection orthogonale sur la droite d'angle θ."""
    c, s = np.cos(theta), np.sin(theta)
    return ApplicationLineaire(
        matrice=np.array([[c**2, c*s], [c*s, s**2]]),
        nom=f"Proj({np.degrees(theta):.0f}°)",
    )


def cisaillement_2d(k: float, horizontal: bool = True) -> ApplicationLineaire:
    """Cisaillement (Scherung)."""
    if horizontal:
        return ApplicationLineaire(np.array([[1, k], [0, 1]]), f"Scher_x({k})")
    return ApplicationLineaire(np.array([[1, 0], [k, 1]]), f"Scher_y({k})")


def homothetie_2d(sx: float, sy: float) -> ApplicationLineaire:
    """Homothétie (Skalierung)."""
    return ApplicationLineaire(np.array([[sx, 0], [0, sy]]), f"Skal({sx},{sy})")


# ======================================================================
#  Visualisation
# ======================================================================

def tracer_transformation(
    app: ApplicationLineaire,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Montre l'effet d'une transformation sur le carré unité et un cercle."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    # Cercle unité
    theta = np.linspace(0, 2*np.pi, 100)
    cercle = np.array([np.cos(theta), np.sin(theta)])
    image = app.matrice @ cercle

    ax.plot(cercle[0], cercle[1], "b-", alpha=0.4, linewidth=1, label="original")
    ax.plot(image[0], image[1], "r-", linewidth=2, label=f"{app.nom}(·)")

    # Vecteurs de base
    for i, (c, nom) in enumerate(zip(["blue", "cyan"], ["e₁", "e₂"])):
        e = np.zeros(2); e[i] = 1
        fe = app(e)
        ax.quiver(0, 0, e[0], e[1], angles="xy", scale_units="xy", scale=1,
                  color=c, alpha=0.5, width=0.01)
        ax.quiver(0, 0, fe[0], fe[1], angles="xy", scale_units="xy", scale=1,
                  color="red", width=0.012)

    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title(app.nom)
    ax.legend(fontsize=9)
    lim = 2.0
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    return ax


if __name__ == "__main__":
    print("=== Construction depuis une fonction ===")
    f = lambda x: np.array([x[0] + x[1], 2*x[0] - x[1], x[0]])
    app = ApplicationLineaire.from_function(f, n=2, m=3, nom="f")
    print(f"f : R² → R³")
    print(f"Matrice =\n{app.matrice}")
    print(f"f([1,0]) = {app(np.array([1, 0]))}")
    print(f"f([0,1]) = {app(np.array([0, 1]))}")
    print(f"dim Kern = {app.noyau_dim()}, dim Bild = {app.image_dim()}")
    print(f"Injective ? {app.est_injective()}, Surjective ? {app.est_surjective()}")

    print(f"\n=== Exemples R² ===")
    transformations = [
        rotation_2d(np.pi/4),
        reflexion_2d(np.pi/6),
        projection_2d(np.pi/4),
        cisaillement_2d(0.5),
        homothetie_2d(2, 0.5),
    ]
    for t in transformations:
        print(f"  {t.nom:15s} : inj={t.est_injective()}, surj={t.est_surjective()}, "
              f"det={np.linalg.det(t.matrice):.4f}")

    print(f"\n=== Composition ===")
    R = rotation_2d(np.pi/2)
    S = reflexion_2d(0)
    RS = R.composer(S)
    print(f"{R.nom} ∘ {S.nom} = {RS.nom}")
    print(f"Matrice =\n{RS.matrice}")

    # Tracés
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, t in zip(axes.flat[:5], transformations):
        tracer_transformation(t, ax=ax)
    axes[1, 2].set_visible(False)
    plt.tight_layout()
    plt.savefig("linear_maps_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
