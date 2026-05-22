"""
coulomb.py
==========

Loi de Coulomb et force électrostatique.

Couvre :
    - F = k·q₁q₂/r² (force entre charges ponctuelles)
    - Superposition : force résultante de plusieurs charges
    - Équilibre électrostatique
    - Comparaison force gravitationnelle vs électrique
    - Visualisation des forces

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


K = 8.9875e9       # constante de Coulomb (N·m²/C²)
E0 = 8.854e-12     # permittivité du vide (C²/N·m²)
E_ELECTRON = 1.602e-19  # charge élémentaire (C)


def force_coulomb(q1: float, q2: float, r: float) -> float:
    """F = k·|q₁q₂|/r². Positif = répulsion, négatif = attraction."""
    return K * q1 * q2 / r**2


def force_coulomb_vecteur(
    q1: float, pos1: np.ndarray, q2: float, pos2: np.ndarray,
) -> np.ndarray:
    """Force exercée par q₂ sur q₁ (vecteur)."""
    r_vec = pos1 - pos2
    r = np.linalg.norm(r_vec)
    if r < 1e-15:
        return np.zeros_like(r_vec)
    return K * q1 * q2 / r**3 * r_vec


def force_resultante(
    q_test: float, pos_test: np.ndarray,
    charges: list[tuple[float, np.ndarray]],
) -> np.ndarray:
    """Force totale sur q_test due à une collection de charges."""
    F_total = np.zeros_like(pos_test, dtype=float)
    for q, pos in charges:
        F_total += force_coulomb_vecteur(q_test, pos_test, q, pos)
    return F_total


def equilibre_1d(q1: float, q2: float, d: float) -> float | None:
    """
    Position d'équilibre d'une charge test entre q₁ (en x=0) et q₂ (en x=d).
    F₁ + F₂ = 0 → k·q₁/x² = k·q₂/(d-x)² → x = d·√|q₁| / (√|q₁| + √|q₂|).
    Valable seulement si q₁ et q₂ sont de même signe.
    """
    if q1 * q2 <= 0:
        return None  # pas d'équilibre entre charges de signes opposés
    return d * np.sqrt(abs(q1)) / (np.sqrt(abs(q1)) + np.sqrt(abs(q2)))


def comparer_gravitationnelle_electrique() -> dict:
    """Compare F_grav et F_elec entre un proton et un électron."""
    m_p = 1.673e-27  # kg
    m_e = 9.109e-31  # kg
    G_const = 6.674e-11
    r = 5.29e-11  # rayon de Bohr

    F_grav = G_const * m_p * m_e / r**2
    F_elec = K * E_ELECTRON**2 / r**2
    return {"F_grav": F_grav, "F_elec": F_elec, "ratio": F_elec / F_grav}


# ======================================================================
#  Tracés
# ======================================================================

def tracer_force_vs_distance(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    r = np.linspace(0.01, 1, 300)
    for q_pair, nom in [((1e-6, 1e-6), "+/+"), ((1e-6, -1e-6), "+/−")]:
        F = [force_coulomb(*q_pair, ri) for ri in r]
        ax.plot(r*100, F, linewidth=2, label=f"{nom}")

    ax.set_xlabel("$r$ (cm)"); ax.set_ylabel("$F$ (N)")
    ax.set_title("Loi de Coulomb : $F \\propto 1/r^2$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_superposition(ax: plt.Axes | None = None) -> plt.Axes:
    """Force sur une charge test due à plusieurs charges sources."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    charges = [
        (2e-6, np.array([-0.3, 0])),
        (-1e-6, np.array([0.3, 0])),
        (1e-6, np.array([0, 0.4])),
    ]

    # Champ de force sur une grille
    x = np.linspace(-0.8, 0.8, 15)
    y = np.linspace(-0.8, 0.8, 15)
    X, Y = np.meshgrid(x, y)
    Fx, Fy = np.zeros_like(X), np.zeros_like(Y)

    q_test = 1e-9
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pos = np.array([X[i,j], Y[i,j]])
            F = force_resultante(q_test, pos, charges)
            Fx[i,j], Fy[i,j] = F[0], F[1]

    # Normaliser pour la visualisation
    F_mag = np.sqrt(Fx**2 + Fy**2)
    F_mag = np.maximum(F_mag, 1e-10)
    ax.quiver(X, Y, Fx/F_mag, Fy/F_mag, F_mag, cmap="hot_r", alpha=0.6)

    # Charges sources
    for q, pos in charges:
        color = "red" if q > 0 else "blue"
        ax.plot(pos[0], pos[1], "o", color=color, markersize=15)
        ax.annotate(f"{q*1e6:.0f} μC", pos, textcoords="offset points",
                    xytext=(10, 10), fontsize=10, color=color)

    ax.set_xlabel("$x$ (m)"); ax.set_ylabel("$y$ (m)")
    ax.set_title("Superposition des forces de Coulomb")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Loi de Coulomb ===\n")
    q1, q2, r = 1e-6, 2e-6, 0.1
    F = force_coulomb(q1, q2, r)
    print(f"  q₁ = {q1*1e6} μC, q₂ = {q2*1e6} μC, r = {r*100} cm")
    print(f"  F = {F:.4f} N (répulsion)")

    F_att = force_coulomb(q1, -q2, r)
    print(f"  q₂ = -{q2*1e6} μC → F = {F_att:.4f} N (attraction)")

    print(f"\n=== Superposition ===\n")
    charges = [(1e-6, np.array([0, 0])), (-1e-6, np.array([0.2, 0]))]
    pos_test = np.array([0.1, 0.1])
    F = force_resultante(1e-9, pos_test, charges)
    print(f"  F_résultante = ({F[0]:.6f}, {F[1]:.6f}) N")
    print(f"  |F| = {np.linalg.norm(F):.6f} N")

    print(f"\n=== Équilibre ===\n")
    x_eq = equilibre_1d(4e-6, 1e-6, 0.3)
    print(f"  q₁ = 4μC (x=0), q₂ = 1μC (x=30cm)")
    print(f"  Équilibre en x = {x_eq*100:.1f} cm (= 2d/3)")

    print(f"\n=== Gravité vs électricité ===\n")
    comp = comparer_gravitationnelle_electrique()
    print(f"  F_grav (p-e) = {comp['F_grav']:.3e} N")
    print(f"  F_elec (p-e) = {comp['F_elec']:.3e} N")
    print(f"  Ratio F_elec/F_grav = {comp['ratio']:.2e}")
    print(f"  → L'électricité est ~10³⁹ × plus forte que la gravité !")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tracer_force_vs_distance(ax=axes[0])
    tracer_superposition(ax=axes[1])
    plt.tight_layout()
    plt.savefig("coulomb_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
