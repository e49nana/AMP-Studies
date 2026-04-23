"""
gershgorin_circles.py
=====================

Cercles de Gershgorin et localisation des valeurs propres.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", section 6.6.

Couvre :
    - Cercles de Gershgorin par lignes (Définition 6.11, Satz 6.12)
    - Cercles de Gershgorin par colonnes (Korollar 6.14)
    - Intersection des deux familles pour encadrement plus serré
    - Visualisation dans le plan complexe
    - Reproduction des Beispiele 6.16, 6.17, 6.18 et Übung 6.15
    - Application : estimation de ρ(A) pour initialiser Wielandt

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ======================================================================
#  1. Calcul des cercles (Définition 6.11)
# ======================================================================

@dataclass
class GershgorinKreis:
    """Un cercle de Gershgorin : centre a_ii, rayon r_i."""
    centre: complex
    rayon: float
    index: int

    def contient(self, z: complex) -> bool:
        return abs(z - self.centre) <= self.rayon + 1e-12


def cercles_lignes(A: np.ndarray) -> list[GershgorinKreis]:
    """
    Cercles de Gershgorin par lignes (Définition 6.11) :
        K_i = {z ∈ ℂ : |z - a_ii| ≤ r_i},  r_i = Σ_{k≠i} |a_ik|.
    """
    A = np.asarray(A)
    n = A.shape[0]
    circles = []
    for i in range(n):
        centre = complex(A[i, i])
        rayon = float(np.sum(np.abs(A[i, :])) - np.abs(A[i, i]))
        circles.append(GershgorinKreis(centre=centre, rayon=rayon, index=i))
    return circles


def cercles_colonnes(A: np.ndarray) -> list[GershgorinKreis]:
    """
    Cercles de Gershgorin par colonnes (Korollar 6.14) :
        K_i = {z ∈ ℂ : |z - a_ii| ≤ r̃_i},  r̃_i = Σ_{k≠i} |a_ki|.

    Équivaut à appliquer Gershgorin à Aᵀ. Les valeurs propres sont
    les mêmes, mais les cercles sont (en général) différents.
    """
    return cercles_lignes(A.T)


def encadrement_spectral(A: np.ndarray) -> tuple[float, float]:
    """
    Borne sur le rayon spectral par Gershgorin :
        ρ(A) ≤ max_i (|a_ii| + r_i).

    Renvoie (borne_inf, borne_sup) pour le rayon spectral.
    """
    circles = cercles_lignes(A)
    sup = max(abs(c.centre) + c.rayon for c in circles)
    inf = max(0, min(abs(c.centre) - c.rayon for c in circles))
    return inf, sup


# ======================================================================
#  2. Vérification du théorème (Satz 6.12)
# ======================================================================

def verifier_gershgorin(A: np.ndarray) -> dict[str, bool]:
    """
    Vérifie que chaque valeur propre est dans au moins un cercle.
    """
    eigvals = np.linalg.eigvals(A)
    circles = cercles_lignes(A)
    result = {}
    for i, lam in enumerate(eigvals):
        dans_un_cercle = any(c.contient(lam) for c in circles)
        result[f"λ_{i+1} = {lam:.4f}"] = dans_un_cercle
    return result


# ======================================================================
#  3. Visualisation
# ======================================================================

def tracer_gershgorin(
    A: np.ndarray,
    titre: str = "Cercles de Gershgorin",
    lignes: bool = True,
    colonnes: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Trace les cercles de Gershgorin et les valeurs propres exactes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    eigvals = np.linalg.eigvals(A)
    couleurs = plt.cm.tab10(np.linspace(0, 1, A.shape[0]))

    # Cercles par lignes
    if lignes:
        circles_l = cercles_lignes(A)
        for c, color in zip(circles_l, couleurs):
            patch = Circle(
                (c.centre.real, c.centre.imag), c.rayon,
                fill=True, facecolor=(*color[:3], 0.15),
                edgecolor=color, linewidth=2,
                label=f"$K_{c.index+1}$ (ligne, $a_{{{c.index+1}{c.index+1}}}={c.centre:.2f}$, $r={c.rayon:.2f}$)",
            )
            ax.add_patch(patch)
            ax.plot(c.centre.real, c.centre.imag, "+", color=color, markersize=10)

    # Cercles par colonnes
    if colonnes:
        circles_c = cercles_colonnes(A)
        for c, color in zip(circles_c, couleurs):
            patch = Circle(
                (c.centre.real, c.centre.imag), c.rayon,
                fill=False, edgecolor=color, linewidth=1.5, linestyle="--",
            )
            ax.add_patch(patch)

    # Valeurs propres exactes
    ax.plot(eigvals.real, eigvals.imag, "k*", markersize=15, zorder=5,
            label="valeurs propres exactes")

    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(titre)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)

    # Auto-ajuster les limites
    all_circles = cercles_lignes(A)
    if colonnes:
        all_circles += cercles_colonnes(A)
    xmin = min(c.centre.real - c.rayon for c in all_circles) - 0.5
    xmax = max(c.centre.real + c.rayon for c in all_circles) + 0.5
    ymin = min(c.centre.imag - c.rayon for c in all_circles) - 0.5
    ymax = max(c.centre.imag + c.rayon for c in all_circles) + 0.5
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return ax


# ======================================================================
#  Démo
# ======================================================================

if __name__ == "__main__":
    # --- Übung 6.15 ---
    print("=== Übung 6.15 ===")
    A1 = np.array([[2, 0.1, -0.1], [0.3, 4, -0.2], [0, 0.8, 5]], dtype=float)
    circles = cercles_lignes(A1)
    for c in circles:
        print(f"  K_{c.index+1}: centre = {c.centre:.1f}, rayon = {c.rayon:.1f}")
    print(f"  Valeurs propres exactes : {np.linalg.eigvals(A1).real}")
    verif = verifier_gershgorin(A1)
    for k, v in verif.items():
        print(f"  {k} dans un cercle : {'✓' if v else '✗'}")

    # --- Beispiel 6.16 (matrice symétrique) ---
    print("\n=== Beispiel 6.16 (symétrique) ===")
    A2 = np.array([[2, 0.4, -0.1], [0.4, 3, 0.3], [-0.1, 0.3, 5]], dtype=float)
    circles2 = cercles_lignes(A2)
    for c in circles2:
        lo, hi = c.centre.real - c.rayon, c.centre.real + c.rayon
        print(f"  K_{c.index+1}: [{lo:.1f}, {hi:.1f}]")
    print(f"  Valeurs propres : {np.sort(np.linalg.eigvalsh(A2))}")

    # --- Beispiel 6.17 ---
    print("\n=== Beispiel 6.17 ===")
    A3 = np.array([[2, 0.7], [0.2, 3]], dtype=float)
    print(f"  Valeurs propres : {np.linalg.eigvals(A3)}")

    # --- Beispiel 6.18 (complexe) ---
    print("\n=== Beispiel 6.18 (complexe) ===")
    A4 = np.array([[3 + 1j, 0.3 + 0.4j], [0.2, 4 - 0.5j]])
    print(f"  Valeurs propres : {np.linalg.eigvals(A4)}")

    # --- Tracés ---
    print("\n=== Tracés ===")
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    tracer_gershgorin(A1, "Übung 6.15", lignes=True, colonnes=True, ax=axes[0, 0])
    tracer_gershgorin(A2, "Beispiel 6.16 (symétrique)", ax=axes[0, 1])
    tracer_gershgorin(A3, "Beispiel 6.17", lignes=True, colonnes=True, ax=axes[1, 0])
    tracer_gershgorin(A4, "Beispiel 6.18 (complexe)", ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig("gershgorin_demo.png", dpi=120)
    print("Figure sauvegardée : gershgorin_demo.png")
