"""
gravity.py
==========

Gravitation et orbites (Kepler).

Couvre :
    - Loi de la gravitation universelle : F = G·m₁m₂/r²
    - Champ gravitationnel g(r) = GM/r²
    - Orbites circulaires et elliptiques (simulation RK4)
    - Lois de Kepler (vérification numérique)
    - Vitesses cosmiques : v₁ (orbitale), v₂ (libération), v₃ (solaire)
    - Énergie orbitale : E = -GMm/(2a)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


G_CONST = 6.674e-11
M_TERRE = 5.972e24
R_TERRE = 6.371e6
M_SOLEIL = 1.989e30
UA = 1.496e11  # unité astronomique


def force_gravitationnelle(m1: float, m2: float, r: float) -> float:
    """F = G·m₁m₂/r²."""
    return G_CONST * m1 * m2 / r**2


def champ_gravitationnel(M: float, r: float) -> float:
    """g(r) = GM/r²."""
    return G_CONST * M / r**2


def vitesse_orbitale(M: float, r: float) -> float:
    """v₁ = √(GM/r) (orbite circulaire)."""
    return np.sqrt(G_CONST * M / r)


def vitesse_liberation(M: float, r: float) -> float:
    """v₂ = √(2GM/r) = v₁·√2."""
    return np.sqrt(2 * G_CONST * M / r)


def periode_orbitale(M: float, a: float) -> float:
    """T = 2π√(a³/(GM)) (3e loi de Kepler)."""
    return 2 * np.pi * np.sqrt(a**3 / (G_CONST * M))


def energie_orbitale(M: float, m: float, a: float) -> float:
    """E = -GMm/(2a)."""
    return -G_CONST * M * m / (2 * a)


# ======================================================================
#  Simulation d'orbites
# ======================================================================

def simuler_orbite(
    M: float, r0: float, v0: float, t_end: float,
    h: float = 100, angle_v0: float = np.pi/2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simule une orbite par RK4. État = [x, y, vx, vy].
    La masse centrale M est en (0,0).
    """
    state = np.array([r0, 0, v0*np.cos(angle_v0), v0*np.sin(angle_v0)])

    def f(t, s):
        x, y, vx, vy = s
        r = np.sqrt(x**2 + y**2)
        if r < R_TERRE * 0.1:
            return np.array([vx, vy, 0, 0])
        a = -G_CONST * M / r**3
        return np.array([vx, vy, a*x, a*y])

    n = int(t_end / h)
    xs, ys = [state[0]], [state[1]]
    for _ in range(n):
        k1 = f(0, state)
        k2 = f(0, state+h/2*k1)
        k3 = f(0, state+h/2*k2)
        k4 = f(0, state+h*k3)
        state = state + h/6*(k1+2*k2+2*k3+k4)
        xs.append(state[0])
        ys.append(state[1])

    return np.array(xs), np.array(ys)


def verifier_kepler_3(planetes: list[tuple[str, float, float]]) -> None:
    """Vérifie T² ∝ a³ (3e loi de Kepler)."""
    print("  Vérification T² / a³ = constante :\n")
    print(f"  {'planète':>10} | {'a (UA)':>8} | {'T (ans)':>8} | {'T²/a³':>10}")
    print("  " + "-" * 45)
    for nom, a_ua, T_ans in planetes:
        ratio = T_ans**2 / a_ua**3
        print(f"  {nom:>10} | {a_ua:>8.3f} | {T_ans:>8.3f} | {ratio:>10.4f}")


# ======================================================================
#  Tracés
# ======================================================================

def tracer_orbites(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    r0 = R_TERRE + 400e3  # ISS altitude
    v_circ = vitesse_orbitale(M_TERRE, r0)

    for factor, nom in [(1.0, "circulaire"), (1.2, "elliptique"), (0.8, "elliptique (bas)")]:
        v0 = v_circ * factor
        T = periode_orbitale(M_TERRE, r0) * 1.5
        x, y = simuler_orbite(M_TERRE, r0, v0, T, h=10)
        ax.plot(x/1e6, y/1e6, linewidth=1.5, label=f"v₀ = {factor}v₁ ({nom})")

    # Terre
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(R_TERRE/1e6 * np.cos(theta), R_TERRE/1e6 * np.sin(theta),
            "b-", linewidth=2, alpha=0.3)
    ax.plot(0, 0, "bo", markersize=10, label="Terre")

    ax.set_xlabel("x (10³ km)"); ax.set_ylabel("y (10³ km)")
    ax.set_title("Orbites terrestres")
    ax.set_aspect("equal"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_champ_g(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    r = np.linspace(R_TERRE, 10*R_TERRE, 300)
    g = [champ_gravitationnel(M_TERRE, ri) for ri in r]

    ax.plot(r/R_TERRE, g, "b-", linewidth=2)
    ax.axhline(9.81, color="red", linestyle="--", alpha=0.5, label="$g_0 = 9.81$ m/s²")
    ax.set_xlabel("$r / R_{Terre}$"); ax.set_ylabel("$g$ (m/s²)")
    ax.set_title("Champ gravitationnel $g(r) = GM/r^2$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Gravitation terrestre ===\n")
    print(f"  g surface = {champ_gravitationnel(M_TERRE, R_TERRE):.3f} m/s²")
    print(f"  v₁ (surface) = {vitesse_orbitale(M_TERRE, R_TERRE):.0f} m/s = {vitesse_orbitale(M_TERRE, R_TERRE)/1000:.1f} km/s")
    print(f"  v₂ (surface) = {vitesse_liberation(M_TERRE, R_TERRE):.0f} m/s = {vitesse_liberation(M_TERRE, R_TERRE)/1000:.1f} km/s")

    print(f"\n=== Orbites satellites ===\n")
    for h_km in [200, 400, 35786]:
        r = R_TERRE + h_km * 1e3
        v = vitesse_orbitale(M_TERRE, r)
        T = periode_orbitale(M_TERRE, r)
        E = energie_orbitale(M_TERRE, 1, r)
        print(f"  h = {h_km:>6} km : v = {v/1000:.2f} km/s, T = {T/3600:.2f} h, E/m = {E:.2e} J/kg")

    print(f"\n=== 3e loi de Kepler ===\n")
    planetes = [
        ("Mercure", 0.387, 0.241),
        ("Vénus", 0.723, 0.615),
        ("Terre", 1.000, 1.000),
        ("Mars", 1.524, 1.881),
        ("Jupiter", 5.203, 11.86),
        ("Saturne", 9.537, 29.46),
    ]
    verifier_kepler_3(planetes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tracer_orbites(ax=axes[0])
    tracer_champ_g(ax=axes[1])
    plt.tight_layout()
    plt.savefig("gravity_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
