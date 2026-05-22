"""
electric_field.py
=================

Champ électrique et lignes de champ.

Couvre :
    - E = F/q = k·Q/r² (champ d'une charge ponctuelle)
    - Superposition des champs
    - Dipôle électrique
    - Lignes de champ (intégration numérique)
    - Visualisation 2D avec streamplot

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


K = 8.9875e9


def champ_charge_ponctuelle(
    Q: float, pos_Q: np.ndarray, pos: np.ndarray,
) -> np.ndarray:
    """E = kQ/r² · r̂ (champ créé par Q en position pos)."""
    r_vec = pos - pos_Q
    r = np.linalg.norm(r_vec)
    if r < 1e-15:
        return np.zeros_like(pos, dtype=float)
    return K * Q / r**3 * r_vec


def champ_total(
    charges: list[tuple[float, np.ndarray]], pos: np.ndarray,
) -> np.ndarray:
    """Superposition des champs de plusieurs charges."""
    E = np.zeros_like(pos, dtype=float)
    for Q, pos_Q in charges:
        E += champ_charge_ponctuelle(Q, pos_Q, pos)
    return E


def champ_sur_grille(
    charges: list[tuple[float, np.ndarray]],
    x_range: tuple, y_range: tuple, n: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calcule Ex, Ey sur une grille 2D."""
    x = np.linspace(*x_range, n)
    y = np.linspace(*y_range, n)
    X, Y = np.meshgrid(x, y)
    Ex, Ey = np.zeros_like(X), np.zeros_like(Y)

    for i in range(n):
        for j in range(n):
            E = champ_total(charges, np.array([X[i,j], Y[i,j]]))
            Ex[i,j], Ey[i,j] = E[0], E[1]

    return X, Y, Ex, Ey


def moment_dipolaire(q: float, d: np.ndarray) -> np.ndarray:
    """p = q·d (vecteur du - vers le +)."""
    return q * np.asarray(d, dtype=float)


# ======================================================================
#  Tracés
# ======================================================================

def tracer_champ_streamplot(
    charges: list[tuple[float, np.ndarray]],
    x_range: tuple = (-2, 2), y_range: tuple = (-2, 2),
    titre: str = "", ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    X, Y, Ex, Ey = champ_sur_grille(charges, x_range, y_range, 40)
    E_mag = np.sqrt(Ex**2 + Ey**2)
    E_mag = np.maximum(E_mag, 1e-10)

    # Limiter pour la visualisation
    lw = 2 * np.log1p(E_mag / E_mag.max())
    ax.streamplot(X, Y, Ex, Ey, color=np.log1p(E_mag), cmap="inferno",
                   linewidth=1, density=2, arrowsize=1.5)

    for Q, pos in charges:
        color = "red" if Q > 0 else "blue"
        size = min(max(abs(Q) * 1e7, 100), 500)
        ax.plot(pos[0], pos[1], "o", color=color, markersize=15,
                markeredgecolor="black", markeredgewidth=1)
        sign = "+" if Q > 0 else "−"
        ax.annotate(sign, pos, ha="center", va="center", fontsize=14,
                    fontweight="bold", color="white")

    ax.set_xlabel("$x$ (m)"); ax.set_ylabel("$y$ (m)")
    ax.set_title(titre if titre else "Lignes de champ électrique")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
    return ax


def tracer_champ_1d(Q: float = 1e-6, ax: plt.Axes | None = None) -> plt.Axes:
    """|E| en fonction de r pour une charge ponctuelle."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    r = np.linspace(0.01, 0.5, 300)
    E = K * abs(Q) / r**2

    ax.plot(r*100, E, "b-", linewidth=2)
    ax.set_xlabel("$r$ (cm)"); ax.set_ylabel("$|E|$ (N/C)")
    ax.set_title(f"$|E| = kQ/r^2$ ($Q = {Q*1e6:.0f}$ μC)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Champ d'une charge ponctuelle ===\n")
    Q = 1e-6  # 1 μC
    for r in [0.01, 0.05, 0.1, 0.5]:
        E = K * Q / r**2
        print(f"  r = {r*100:>5.0f} cm : |E| = {E:.2e} N/C")

    print(f"\n=== Dipôle ===\n")
    q = 1e-6
    d = 0.1
    p = moment_dipolaire(q, np.array([d, 0]))
    print(f"  q = {q*1e6} μC, d = {d*100} cm")
    print(f"  p = {np.linalg.norm(p):.2e} C·m")

    # Champ sur l'axe du dipôle
    for x in [0.2, 0.5, 1.0]:
        E = champ_total([(q, np.array([d/2, 0])), (-q, np.array([-d/2, 0]))],
                        np.array([x, 0]))
        print(f"  E(x={x}) = ({E[0]:.4f}, {E[1]:.4f}) N/C, |E| = {np.linalg.norm(E):.4f}")

    # Tracés
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Charge unique +
    tracer_champ_streamplot([(1e-6, np.array([0, 0]))],
                             titre="Charge positive", ax=axes[0, 0])

    # Dipôle
    tracer_champ_streamplot(
        [(1e-6, np.array([0.3, 0])), (-1e-6, np.array([-0.3, 0]))],
        titre="Dipôle", ax=axes[0, 1])

    # Deux charges +
    tracer_champ_streamplot(
        [(1e-6, np.array([-0.3, 0])), (1e-6, np.array([0.3, 0]))],
        titre="Deux charges +", ax=axes[1, 0])

    # Quadrupôle
    tracer_champ_streamplot(
        [(1e-6, np.array([0.3, 0.3])), (-1e-6, np.array([-0.3, 0.3])),
         (-1e-6, np.array([0.3, -0.3])), (1e-6, np.array([-0.3, -0.3]))],
        titre="Quadrupôle", ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig("electric_field_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
