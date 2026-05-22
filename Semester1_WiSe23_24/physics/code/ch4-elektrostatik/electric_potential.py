"""
electric_potential.py
=====================

Potentiel électrique et équipotentielles.

Couvre :
    - V = kQ/r (potentiel d'une charge ponctuelle)
    - Superposition des potentiels
    - E = -∇V (le champ est le gradient négatif du potentiel)
    - Énergie potentielle U = qV
    - Équipotentielles (perpendiculaires aux lignes de champ)
    - Condensateur plan : V linéaire entre les plaques

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


K = 8.9875e9
E0 = 8.854e-12


def potentiel_charge(Q: float, pos_Q: np.ndarray, pos: np.ndarray) -> float:
    """V = kQ/r."""
    r = np.linalg.norm(pos - pos_Q)
    if r < 1e-15:
        return np.sign(Q) * 1e15
    return K * Q / r


def potentiel_total(
    charges: list[tuple[float, np.ndarray]], pos: np.ndarray,
) -> float:
    """V = Σ kQᵢ/rᵢ (superposition)."""
    return sum(potentiel_charge(Q, pQ, pos) for Q, pQ in charges)


def potentiel_sur_grille(
    charges: list[tuple[float, np.ndarray]],
    x_range: tuple, y_range: tuple, n: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcule V sur une grille 2D."""
    x = np.linspace(*x_range, n)
    y = np.linspace(*y_range, n)
    X, Y = np.meshgrid(x, y)
    V = np.zeros_like(X)

    for i in range(n):
        for j in range(n):
            V[i, j] = potentiel_total(charges, np.array([X[i,j], Y[i,j]]))

    return X, Y, V


def champ_depuis_potentiel(
    X: np.ndarray, Y: np.ndarray, V: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """E = -∇V (gradient numérique)."""
    dy = Y[1, 0] - Y[0, 0]
    dx = X[0, 1] - X[0, 0]
    Ey, Ex = np.gradient(-V, dy, dx)
    return Ex, Ey


def energie_potentielle(q: float, V: float) -> float:
    """U = qV."""
    return q * V


def condensateur_plan(V0: float, d: float, n_points: int = 100) -> dict:
    """
    Condensateur plan : V varie linéairement, E = V₀/d constant.
    """
    x = np.linspace(0, d, n_points)
    V = V0 * (1 - x / d)
    E = V0 / d
    C_par_A = E0 / d  # C/A = ε₀/d (capacité par unité de surface)
    return {"x": x, "V": V, "E": E, "C_par_A": C_par_A}


# ======================================================================
#  Tracés
# ======================================================================

def tracer_equipotentielles(
    charges: list[tuple[float, np.ndarray]],
    x_range: tuple = (-2, 2), y_range: tuple = (-2, 2),
    titre: str = "", ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    X, Y, V = potentiel_sur_grille(charges, x_range, y_range, 150)
    Ex, Ey = champ_depuis_potentiel(X, Y, V)

    # Limiter V pour la visualisation
    V_clip = np.clip(V, -1e6, 1e6)

    # Équipotentielles
    levels = np.linspace(np.percentile(V_clip, 5), np.percentile(V_clip, 95), 20)
    cs = ax.contour(X, Y, V_clip, levels=levels, cmap="RdBu_r", linewidths=1)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")

    # Lignes de champ (streamplot)
    E_mag = np.sqrt(Ex**2 + Ey**2)
    E_mag = np.maximum(E_mag, 1e-10)
    ax.streamplot(X, Y, Ex, Ey, color="grey", linewidth=0.5, density=1.5, arrowsize=1)

    for Q, pos in charges:
        color = "red" if Q > 0 else "blue"
        ax.plot(pos[0], pos[1], "o", color=color, markersize=12,
                markeredgecolor="black", markeredgewidth=1)

    ax.set_xlabel("$x$ (m)"); ax.set_ylabel("$y$ (m)")
    ax.set_title(titre if titre else "Équipotentielles (couleurs) + lignes de champ (gris)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
    return ax


def tracer_V_1d(ax: plt.Axes | None = None) -> plt.Axes:
    """V(r) pour une charge ponctuelle."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    r = np.linspace(0.01, 0.5, 300)
    for Q_uC in [1, -1, 2]:
        Q = Q_uC * 1e-6
        V = K * Q / r
        ax.plot(r*100, V/1e3, linewidth=2, label=f"$Q = {Q_uC}$ μC")

    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("$r$ (cm)"); ax.set_ylabel("$V$ (kV)")
    ax.set_title("Potentiel $V = kQ/r$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_gradient_verification(ax: plt.Axes | None = None) -> plt.Axes:
    """Vérifie E = -dV/dr pour une charge ponctuelle."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    Q = 1e-6
    r = np.linspace(0.02, 0.5, 300)
    V = K * Q / r
    E_exact = K * Q / r**2

    # E numérique = -dV/dr
    dr = r[1] - r[0]
    E_num = -np.gradient(V, dr)

    ax.plot(r*100, E_exact, "b-", linewidth=2, label="$E = kQ/r^2$ (exact)")
    ax.plot(r*100, E_num, "r--", linewidth=2, label="$E = -dV/dr$ (numérique)")
    ax.set_xlabel("$r$ (cm)"); ax.set_ylabel("$E$ (N/C)")
    ax.set_title("Vérification $E = -\\nabla V$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Potentiel d'une charge ===\n")
    Q = 1e-6
    for r in [0.01, 0.05, 0.1, 0.5]:
        V = K * Q / r
        print(f"  r = {r*100:>5.0f} cm : V = {V/1e3:.2f} kV")

    print(f"\n=== Superposition (dipôle) ===\n")
    charges = [(1e-6, np.array([0.1, 0])), (-1e-6, np.array([-0.1, 0]))]
    for pos in [np.array([0.5, 0]), np.array([0, 0.5]), np.array([0, 0])]:
        V = potentiel_total(charges, pos)
        print(f"  V({pos}) = {V:.2f} V")
    print(f"  V(0,0) = 0 (symétrie du dipôle) ✓")

    print(f"\n=== E = -∇V ===\n")
    print(f"  V = kQ/r → -dV/dr = kQ/r² = E ✓")

    print(f"\n=== Condensateur plan ===\n")
    c = condensateur_plan(1000, 0.01)
    print(f"  V₀ = 1000 V, d = 1 cm")
    print(f"  E = V₀/d = {c['E']:.0f} V/m = {c['E']/100:.0f} V/cm")
    print(f"  C/A = ε₀/d = {c['C_par_A']*1e12:.2f} pF/m²")

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    tracer_V_1d(ax=axes[0, 0])
    tracer_gradient_verification(ax=axes[0, 1])
    tracer_equipotentielles(
        [(1e-6, np.array([0.3, 0])), (-1e-6, np.array([-0.3, 0]))],
        titre="Dipôle : équipotentielles + champ", ax=axes[1, 0])
    tracer_equipotentielles(
        [(1e-6, np.array([-0.3, 0])), (1e-6, np.array([0.3, 0]))],
        titre="Deux charges + : V > 0 partout", ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig("electric_potential_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
