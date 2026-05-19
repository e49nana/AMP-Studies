"""
newton_laws.py
==============

Les trois lois de Newton et applications classiques.

Couvre :
    - F = ma : résolution numérique du mouvement
    - Plan incliné avec et sans frottement
    - Système de poulies (Atwood)
    - Frottement statique vs dynamique
    - Force normale, tension, poids

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


G = 9.81


def rk4_1d(f, t0, state0, t_end, h):
    """RK4 pour état = [x, v]."""
    state = np.array(state0, dtype=float)
    n = int((t_end - t0) / h)
    t = np.zeros(n+1); xs = np.zeros(n+1); vs = np.zeros(n+1)
    t[0] = t0; xs[0] = state[0]; vs[0] = state[1]
    for k in range(n):
        k1 = np.array(f(t[k], state))
        k2 = np.array(f(t[k]+h/2, state+h/2*k1))
        k3 = np.array(f(t[k]+h/2, state+h/2*k2))
        k4 = np.array(f(t[k]+h, state+h*k3))
        state = state + h/6*(k1+2*k2+2*k3+k4)
        t[k+1] = t[k]+h; xs[k+1] = state[0]; vs[k+1] = state[1]
    return t, xs, vs


# ======================================================================
#  1. Plan incliné
# ======================================================================

def plan_incline(
    m: float, theta: float, mu_k: float = 0, g: float = G,
) -> dict:
    """
    Bloc de masse m sur plan incliné d'angle θ.
    mu_k = coefficient de frottement cinétique.
    """
    N = m * g * np.cos(theta)
    F_gravity_parallel = m * g * np.sin(theta)
    F_frottement = mu_k * N
    a = g * (np.sin(theta) - mu_k * np.cos(theta))

    return {
        "N": N,
        "F_grav_par": F_gravity_parallel,
        "F_frottement": F_frottement,
        "a": max(a, 0),  # ne remonte pas si frottement > gravité
        "glisse": a > 0,
    }


def angle_critique(mu_s: float) -> float:
    """Angle à partir duquel le bloc glisse : tan(θ_c) = μ_s."""
    return np.arctan(mu_s)


def simuler_plan_incline(
    m: float, theta: float, mu_k: float, L: float, g: float = G,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simule le glissement sur un plan de longueur L."""
    a = g * (np.sin(theta) - mu_k * np.cos(theta))
    if a <= 0:
        return np.array([0]), np.array([0]), np.array([0])

    def f(t, state):
        x, v = state
        if x >= L:
            return [0, 0]
        return [v, a]

    t_end = np.sqrt(2*L/a) * 1.2
    return rk4_1d(f, 0, [0, 0], t_end, 0.001)


# ======================================================================
#  2. Machine d'Atwood
# ======================================================================

def atwood(m1: float, m2: float, g: float = G) -> dict:
    """
    Deux masses reliées par une corde sur une poulie.
    a = (m₂ - m₁)g / (m₁ + m₂), T = 2m₁m₂g / (m₁ + m₂).
    """
    a = (m2 - m1) * g / (m1 + m2)
    T = 2 * m1 * m2 * g / (m1 + m2)
    return {"a": a, "T": T, "a/g": a/g}


# ======================================================================
#  3. Frottement statique vs dynamique
# ======================================================================

def force_frottement(F_app: float, m: float, mu_s: float, mu_k: float, g: float = G) -> dict:
    """
    Force de frottement en fonction de la force appliquée.
    Statique tant que F_app < μ_s·mg, puis cinétique.
    """
    N = m * g
    F_s_max = mu_s * N
    F_k = mu_k * N

    if F_app <= F_s_max:
        return {"regime": "statique", "F_frott": F_app, "a": 0, "bouge": False}
    else:
        a = (F_app - F_k) / m
        return {"regime": "cinétique", "F_frott": F_k, "a": a, "bouge": True}


# ======================================================================
#  4. Tracés
# ======================================================================

def tracer_plan_incline(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    thetas = np.linspace(0, np.pi/3, 100)
    for mu in [0, 0.1, 0.3, 0.5]:
        a_vals = [G * (np.sin(t) - mu * np.cos(t)) for t in thetas]
        a_vals = [max(a, 0) for a in a_vals]
        ax.plot(np.degrees(thetas), a_vals, linewidth=2, label=f"μ = {mu}")
        if mu > 0:
            tc = np.degrees(angle_critique(mu))
            ax.axvline(tc, color="grey", linestyle=":", alpha=0.3)

    ax.set_xlabel("angle θ (°)"); ax.set_ylabel("accélération (m/s²)")
    ax.set_title("Plan incliné : accélération vs angle")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_frottement_statique_dynamique(
    m: float = 5, mu_s: float = 0.5, mu_k: float = 0.3,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    F_range = np.linspace(0, 60, 300)
    F_frott = []
    for F in F_range:
        r = force_frottement(F, m, mu_s, mu_k)
        F_frott.append(r["F_frott"])

    ax.plot(F_range, F_frott, "b-", linewidth=2, label="$F_{frott}$")
    ax.plot(F_range, F_range, "k:", alpha=0.3, label="$F_{app}$ (référence)")
    ax.axhline(mu_s * m * G, color="red", linestyle="--", alpha=0.5,
                label=f"$\\mu_s mg = {mu_s*m*G:.1f}$ N")
    ax.axhline(mu_k * m * G, color="green", linestyle="--", alpha=0.5,
                label=f"$\\mu_k mg = {mu_k*m*G:.1f}$ N")

    ax.set_xlabel("$F_{app}$ (N)"); ax.set_ylabel("$F_{frott}$ (N)")
    ax.set_title(f"Frottement : statique → cinétique (m={m} kg)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Plan incliné ===\n")
    m = 10  # kg
    for theta_deg in [15, 30, 45]:
        theta = np.radians(theta_deg)
        for mu in [0, 0.2]:
            r = plan_incline(m, theta, mu)
            print(f"  θ={theta_deg}°, μ={mu} : a={r['a']:.2f} m/s², "
                  f"N={r['N']:.1f} N, glisse={r['glisse']}")

    print(f"\n  Angle critique (μ_s=0.5) : {np.degrees(angle_critique(0.5)):.1f}°")

    print(f"\n=== Machine d'Atwood ===\n")
    for m1, m2 in [(1, 2), (5, 5.5), (3, 7)]:
        r = atwood(m1, m2)
        print(f"  m₁={m1}, m₂={m2} : a={r['a']:.3f} m/s² ({r['a/g']:.3f}g), T={r['T']:.2f} N")

    print(f"\n=== Frottement statique → dynamique ===\n")
    for F in [5, 20, 24.5, 25, 30]:
        r = force_frottement(F, 5, 0.5, 0.3)
        print(f"  F={F:>5.1f} N : {r['regime']:>10}, F_frott={r['F_frott']:.1f} N, a={r['a']:.2f} m/s²")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_plan_incline(ax=axes[0])
    tracer_frottement_statique_dynamique(ax=axes[1])
    plt.tight_layout()
    plt.savefig("newton_laws_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
