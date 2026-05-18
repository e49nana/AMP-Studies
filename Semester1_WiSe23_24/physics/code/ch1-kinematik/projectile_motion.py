"""
projectile_motion.py
====================

Mouvement d'un projectile en 2D (Wurfbewegung).

Couvre :
    - Tir oblique sans frottement : x(t) = v₀ cos(α)·t, y(t) = v₀ sin(α)·t - g/2·t²
    - Portée, hauteur maximale, temps de vol
    - Angle optimal (45° sans frottement)
    - Tir oblique AVEC frottement (résolution par RK4)
    - Enveloppe de sécurité (parabole de sûreté)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


G = 9.81  # m/s²


# ======================================================================
#  1. Sans frottement (analytique)
# ======================================================================

def trajectoire_analytique(
    v0: float, alpha: float, g: float = G, dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Trajectoire parabolique : renvoie (x, y) jusqu'au sol."""
    t_vol = 2 * v0 * np.sin(alpha) / g
    t = np.arange(0, t_vol + dt, dt)
    x = v0 * np.cos(alpha) * t
    y = v0 * np.sin(alpha) * t - 0.5 * g * t**2
    y = np.maximum(y, 0)
    return x, y


def portee(v0: float, alpha: float, g: float = G) -> float:
    """R = v₀² sin(2α) / g."""
    return v0**2 * np.sin(2 * alpha) / g


def hauteur_max(v0: float, alpha: float, g: float = G) -> float:
    """H = v₀² sin²(α) / (2g)."""
    return v0**2 * np.sin(alpha)**2 / (2 * g)


def temps_vol(v0: float, alpha: float, g: float = G) -> float:
    """T = 2v₀ sin(α) / g."""
    return 2 * v0 * np.sin(alpha) / g


def angle_optimal(g: float = G) -> float:
    """45° = π/4 (maximise la portée sans frottement)."""
    return np.pi / 4


# ======================================================================
#  2. Avec frottement (numérique)
# ======================================================================

def trajectoire_frottement(
    v0: float, alpha: float, k: float = 0.1, m: float = 1.0,
    g: float = G, dt: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Avec frottement proportionnel à v² :
        F_frott = -k·|v|·v (force de traînée).
    Résolu par RK4.
    """
    def f(t, state):
        x, y, vx, vy = state
        v = np.sqrt(vx**2 + vy**2)
        ax = -(k/m) * v * vx
        ay = -g - (k/m) * v * vy
        return np.array([vx, vy, ax, ay])

    state = np.array([0, 0, v0*np.cos(alpha), v0*np.sin(alpha)])
    xs, ys = [0], [0]

    t = 0
    while state[1] >= 0 or t < 0.1:
        k1 = f(t, state)
        k2 = f(t+dt/2, state+dt/2*k1)
        k3 = f(t+dt/2, state+dt/2*k2)
        k4 = f(t+dt, state+dt*k3)
        state = state + dt/6*(k1+2*k2+2*k3+k4)
        t += dt
        if state[1] < 0:
            break
        xs.append(state[0])
        ys.append(max(state[1], 0))
        if t > 100:
            break

    return np.array(xs), np.array(ys)


# ======================================================================
#  3. Enveloppe de sécurité
# ======================================================================

def enveloppe_securite(v0: float, g: float = G, n_points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """
    Parabole de sûreté : y_max(x) = v₀²/(2g) - g·x²/(2v₀²).
    Tous les projectiles tirés à v₀ restent sous cette courbe.
    """
    x_max = v0**2 / g
    x = np.linspace(0, x_max, n_points)
    y = v0**2 / (2*g) - g * x**2 / (2 * v0**2)
    return x, np.maximum(y, 0)


# ======================================================================
#  4. Tracés
# ======================================================================

def tracer_angles(v0: float = 20, ax: plt.Axes | None = None) -> plt.Axes:
    """Trajectoires pour différents angles de tir."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    angles = [15, 30, 45, 60, 75]
    for deg in angles:
        alpha = np.radians(deg)
        x, y = trajectoire_analytique(v0, alpha)
        R = portee(v0, alpha)
        ax.plot(x, y, linewidth=2, label=f"{deg}° (R={R:.1f} m)")

    # Enveloppe
    xe, ye = enveloppe_securite(v0)
    ax.plot(xe, ye, "k--", linewidth=1.5, alpha=0.5, label="enveloppe")

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Tir oblique ($v_0 = {v0}$ m/s)")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    return ax


def tracer_avec_sans_frottement(
    v0: float = 30, alpha_deg: float = 45,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    alpha = np.radians(alpha_deg)
    x1, y1 = trajectoire_analytique(v0, alpha)
    x2, y2 = trajectoire_frottement(v0, alpha, k=0.05, m=1.0)
    x3, y3 = trajectoire_frottement(v0, alpha, k=0.2, m=1.0)

    ax.plot(x1, y1, "b-", linewidth=2, label="sans frottement")
    ax.plot(x2, y2, "r--", linewidth=2, label="k = 0.05")
    ax.plot(x3, y3, "g:", linewidth=2, label="k = 0.2")

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Effet du frottement ($v_0 = {v0}$, α = {alpha_deg}°)")
    ax.set_ylim(0, None)
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    v0 = 20  # m/s

    print(f"=== Tir oblique sans frottement (v₀ = {v0} m/s) ===\n")
    print(f"  {'angle':>6} | {'portée':>8} | {'H_max':>8} | {'T_vol':>6}")
    print("  " + "-" * 38)
    for deg in [15, 30, 45, 60, 75]:
        alpha = np.radians(deg)
        print(f"  {deg:>5}° | {portee(v0, alpha):>7.2f}m | {hauteur_max(v0, alpha):>7.2f}m | {temps_vol(v0, alpha):>5.2f}s")

    print(f"\n  Angle optimal : {np.degrees(angle_optimal()):.0f}°")
    print(f"  Portée max : {portee(v0, angle_optimal()):.2f} m = v₀²/g = {v0**2/G:.2f} m")

    print(f"\n=== Avec frottement (v₀ = 30, α = 45°) ===\n")
    for k in [0, 0.05, 0.1, 0.2]:
        if k == 0:
            R = portee(30, np.pi/4)
        else:
            x, y = trajectoire_frottement(30, np.pi/4, k=k)
            R = x[-1]
        print(f"  k = {k:<4} : portée ≈ {R:.1f} m")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    tracer_angles(v0=20, ax=axes[0])
    tracer_avec_sans_frottement(v0=30, ax=axes[1])
    plt.tight_layout()
    plt.savefig("projectile_motion_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
