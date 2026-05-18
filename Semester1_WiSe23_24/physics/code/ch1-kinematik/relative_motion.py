"""
relative_motion.py
==================

Référentiels et mouvement relatif (Galileo).

Couvre :
    - Changement de référentiel : r_B = r_A + r_{A→B}
    - Composition des vitesses : v_B = v_A + v_{A→B}
    - Transformation de Galilée
    - Exemples : bateau traversant une rivière, avion dans le vent
    - Forces fictives : Coriolis (introduction qualitative)
    - Visualisation des trajectoires dans deux référentiels

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Composition des vitesses
# ======================================================================

def vitesse_resultante(v_objet: np.ndarray, v_referentiel: np.ndarray) -> np.ndarray:
    """v_absolue = v_relative + v_référentiel."""
    return np.asarray(v_objet) + np.asarray(v_referentiel)


def trajectoire_dans_referentiel(
    x_abs: np.ndarray, y_abs: np.ndarray,
    vx_ref: float, vy_ref: float, t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Position dans un référentiel se déplaçant à (vx_ref, vy_ref)."""
    return x_abs - vx_ref * t, y_abs - vy_ref * t


# ======================================================================
#  2. Exemples classiques
# ======================================================================

def bateau_riviere(
    v_bateau: float, v_courant: float, angle: float, largeur: float,
) -> dict:
    """
    Bateau traversant une rivière :
        - v_bateau : vitesse du bateau dans l'eau (perpendiculaire visée)
        - v_courant : vitesse du courant (horizontal)
        - angle : angle de visée par rapport à la perpendiculaire
    """
    vx = v_bateau * np.sin(angle) + v_courant
    vy = v_bateau * np.cos(angle)
    t_traversee = largeur / vy if vy > 0 else float("inf")
    derive = vx * t_traversee
    v_sol = np.sqrt(vx**2 + vy**2)

    return {
        "v_sol": v_sol,
        "t_traversee": t_traversee,
        "derive": derive,
        "angle_resultant": np.degrees(np.arctan2(vx, vy)),
    }


def correction_angle_riviere(v_bateau: float, v_courant: float) -> float:
    """Angle pour traverser droit : sin(α) = v_courant / v_bateau."""
    if v_courant >= v_bateau:
        return float("nan")  # impossible
    return np.arcsin(v_courant / v_bateau)


def avion_vent(
    v_avion: float, cap: float, v_vent: float, dir_vent: float,
) -> dict:
    """
    Avion dans le vent :
        cap = direction visée (rad depuis le nord)
        dir_vent = direction D'OÙ vient le vent
    """
    vx_avion = v_avion * np.sin(cap)
    vy_avion = v_avion * np.cos(cap)
    vx_vent = -v_vent * np.sin(dir_vent)
    vy_vent = -v_vent * np.cos(dir_vent)

    vx_sol = vx_avion + vx_vent
    vy_sol = vy_avion + vy_vent
    v_sol = np.sqrt(vx_sol**2 + vy_sol**2)
    route = np.arctan2(vx_sol, vy_sol)

    return {"v_sol": v_sol, "route_deg": np.degrees(route), "derive_deg": np.degrees(route - cap)}


# ======================================================================
#  3. Tracés
# ======================================================================

def tracer_bateau(v_b: float = 5, v_c: float = 2, L: float = 100,
                   ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Cas 1 : vise droit
    t_max = L / v_b
    t = np.linspace(0, t_max, 200)
    x1 = v_c * t
    y1 = v_b * t
    ax.plot(x1, y1, "b-", linewidth=2, label=f"vise droit (dérive = {v_c*t_max:.0f} m)")

    # Cas 2 : corrige l'angle
    alpha = correction_angle_riviere(v_b, v_c)
    if not np.isnan(alpha):
        vy = v_b * np.cos(alpha)
        t_max2 = L / vy
        t2 = np.linspace(0, t_max2, 200)
        x2 = np.zeros_like(t2)
        y2 = vy * t2
        ax.plot(x2, y2, "r--", linewidth=2,
                label=f"corrigé α={np.degrees(alpha):.1f}° (dérive = 0)")

    # Rivière
    ax.fill_between([-20, 80], [0, 0], [L, L], alpha=0.1, color="cyan")
    ax.axhline(0, color="brown", linewidth=2)
    ax.axhline(L, color="brown", linewidth=2)

    # Courant
    for y_arrow in np.linspace(10, 90, 5):
        ax.annotate("", xy=(20, y_arrow), xytext=(0, y_arrow),
                    arrowprops=dict(arrowstyle="->", color="blue", alpha=0.3))
    ax.text(10, 50, f"courant\n{v_c} m/s", ha="center", color="blue", alpha=0.5)

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Bateau : $v_b={v_b}$, $v_c={v_c}$, largeur={L}m")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    return ax


def tracer_deux_referentiels(ax: plt.Axes | None = None) -> plt.Axes:
    """Même mouvement vu de deux référentiels."""
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    else:
        axes = [ax, ax]

    # Objet : tir parabolique
    v0, alpha = 10, np.radians(60)
    g = 9.81
    t_vol = 2*v0*np.sin(alpha)/g
    t = np.linspace(0, t_vol, 200)
    x = v0*np.cos(alpha)*t
    y = v0*np.sin(alpha)*t - 0.5*g*t**2

    # Référentiel fixe
    axes[0].plot(x, y, "b-", linewidth=2, label="trajectoire (sol)")
    axes[0].set_title("Référentiel du sol")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_aspect("equal"); axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Référentiel en translation à vx = v0 cos(α)
    vx_ref = v0 * np.cos(alpha)
    x_rel = x - vx_ref * t
    axes[1].plot(x_rel, y, "r-", linewidth=2, label="trajectoire (mobile)")
    axes[1].set_title(f"Référentiel mobile ($v_x = {vx_ref:.1f}$ m/s)")
    axes[1].set_xlabel("x'"); axes[1].set_ylabel("y")
    axes[1].set_aspect("equal"); axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    return axes[0]


if __name__ == "__main__":
    print("=== Bateau traversant une rivière ===\n")
    v_b, v_c, L = 5, 2, 100
    r1 = bateau_riviere(v_b, v_c, 0, L)
    print(f"  Vise droit : v_sol={r1['v_sol']:.2f} m/s, t={r1['t_traversee']:.1f}s, "
          f"dérive={r1['derive']:.0f}m")

    alpha = correction_angle_riviere(v_b, v_c)
    r2 = bateau_riviere(v_b, v_c, -alpha, L)
    print(f"  Corrigé (α={np.degrees(alpha):.1f}°) : t={r2['t_traversee']:.1f}s, "
          f"dérive={r2['derive']:.1f}m")

    print(f"\n=== Avion dans le vent ===\n")
    r = avion_vent(250, np.radians(0), 50, np.radians(270))
    print(f"  Cap Nord, vent d'Ouest 50 km/h :")
    print(f"    v_sol = {r['v_sol']:.1f} km/h, route = {r['route_deg']:.1f}°, "
          f"dérive = {r['derive_deg']:.1f}°")

    print(f"\n=== Composition des vitesses ===")
    print(f"  Train 100 km/h + passager 5 km/h :")
    print(f"    même sens : {100+5} km/h")
    print(f"    sens opposé : {100-5} km/h")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tracer_bateau(ax=axes[0])
    tracer_deux_referentiels.__wrapped__ = None
    # Simplified dual plot
    v0, alpha = 10, np.radians(60)
    g = 9.81
    t_vol = 2*v0*np.sin(alpha)/g
    t = np.linspace(0, t_vol, 200)
    x = v0*np.cos(alpha)*t
    y = v0*np.sin(alpha)*t - 0.5*g*t**2
    vx_ref = v0*np.cos(alpha)
    axes[1].plot(x, y, "b-", linewidth=2, label="vue du sol")
    axes[1].plot(x - vx_ref*t, y, "r--", linewidth=2, label="vue du mobile")
    axes[1].set_title("Même tir, deux référentiels")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("relative_motion_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
