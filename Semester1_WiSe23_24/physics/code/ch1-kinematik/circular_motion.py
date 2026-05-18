"""
circular_motion.py
==================

Mouvement circulaire uniforme et non uniforme.

Couvre :
    - MCU : ω, v = ωr, a_c = ω²r = v²/r
    - Période T = 2π/ω, fréquence f = 1/T
    - Accélération centripète (direction radiale)
    - Force centripète F_c = mω²r
    - Mouvement circulaire non uniforme : composantes tangentielle et radiale
    - Applications : virage, manège, orbite satellite

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. MCU (Gleichförmige Kreisbewegung)
# ======================================================================

def position_mcu(R: float, omega: float, t: np.ndarray, phi0: float = 0) -> tuple[np.ndarray, np.ndarray]:
    """x(t) = R cos(ωt + φ₀), y(t) = R sin(ωt + φ₀)."""
    return R * np.cos(omega * t + phi0), R * np.sin(omega * t + phi0)


def vitesse_mcu(R: float, omega: float, t: np.ndarray, phi0: float = 0) -> tuple[np.ndarray, np.ndarray]:
    """vx = -Rω sin(ωt + φ₀), vy = Rω cos(ωt + φ₀)."""
    return -R * omega * np.sin(omega * t + phi0), R * omega * np.cos(omega * t + phi0)


def acceleration_mcu(R: float, omega: float, t: np.ndarray, phi0: float = 0) -> tuple[np.ndarray, np.ndarray]:
    """ax = -Rω² cos(ωt + φ₀), ay = -Rω² sin(ωt + φ₀). Toujours vers le centre."""
    return -R * omega**2 * np.cos(omega * t + phi0), -R * omega**2 * np.sin(omega * t + phi0)


def vitesse_tangentielle(R: float, omega: float) -> float:
    """v = ωR."""
    return omega * R


def acceleration_centripete(R: float, omega: float = None, v: float = None) -> float:
    """a_c = ω²R = v²/R."""
    if omega is not None:
        return omega**2 * R
    elif v is not None:
        return v**2 / R
    raise ValueError("Fournir omega ou v.")


def force_centripete(m: float, R: float, omega: float) -> float:
    """F_c = mω²R."""
    return m * omega**2 * R


def periode(omega: float) -> float:
    """T = 2π/ω."""
    return 2 * np.pi / omega


# ======================================================================
#  2. Applications
# ======================================================================

def virage_vitesse_max(R: float, mu: float, g: float = 9.81) -> float:
    """Vitesse max dans un virage plat : v_max = √(μgR)."""
    return np.sqrt(mu * g * R)


def orbite_satellite(h: float, M: float = 5.972e24, R_terre: float = 6.371e6) -> dict:
    """Orbite circulaire à altitude h."""
    G_const = 6.674e-11
    r = R_terre + h
    v = np.sqrt(G_const * M / r)
    T = 2 * np.pi * r / v
    return {"altitude": h, "rayon": r, "vitesse": v, "periode_h": T/3600}


# ======================================================================
#  3. Tracés
# ======================================================================

def tracer_mcu(R: float = 1.0, omega: float = 2.0, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    t = np.linspace(0, 2*np.pi/omega, 200)
    x, y = position_mcu(R, omega, t)
    vx, vy = vitesse_mcu(R, omega, t)
    ax_x, ax_y = acceleration_mcu(R, omega, t)

    ax.plot(x, y, "b-", linewidth=2, label="trajectoire")

    # Quelques vecteurs vitesse et accélération
    for i in range(0, len(t), len(t)//6):
        scale_v = 0.3 / (R * omega)
        scale_a = 0.3 / (R * omega**2)
        ax.quiver(x[i], y[i], vx[i]*scale_v, vy[i]*scale_v,
                  color="green", width=0.008, label="$\\vec{v}$" if i == 0 else "")
        ax.quiver(x[i], y[i], ax_x[i]*scale_a, ax_y[i]*scale_a,
                  color="red", width=0.008, label="$\\vec{a}_c$" if i == 0 else "")

    ax.plot(0, 0, "k+", markersize=15)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title(f"MCU : $R={R}$, $\\omega={omega}$, $v={R*omega:.1f}$, $a_c={R*omega**2:.1f}$")
    ax.legend(fontsize=9)
    return ax


def tracer_composantes(R: float = 1.0, omega: float = 2.0, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    T = 2 * np.pi / omega
    t = np.linspace(0, 2*T, 500)
    x, y = position_mcu(R, omega, t)
    vx, vy = vitesse_mcu(R, omega, t)

    ax.plot(t, x, "b-", linewidth=2, label="$x(t) = R\\cos(\\omega t)$")
    ax.plot(t, y, "r-", linewidth=2, label="$y(t) = R\\sin(\\omega t)$")
    ax.plot(t, np.sqrt(vx**2 + vy**2), "g--", linewidth=1.5, label=f"$|v| = {R*omega:.2f}$ (constant)")

    ax.set_xlabel("$t$ (s)"); ax.set_ylabel("position / vitesse")
    ax.set_title("Composantes du MCU")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== MCU : R = 2 m, ω = 3 rad/s ===\n")
    R, omega = 2.0, 3.0
    print(f"  v = ωR = {vitesse_tangentielle(R, omega):.1f} m/s")
    print(f"  a_c = ω²R = {acceleration_centripete(R, omega=omega):.1f} m/s²")
    print(f"  T = 2π/ω = {periode(omega):.3f} s")
    print(f"  F_c (m=1kg) = {force_centripete(1, R, omega):.1f} N")

    print(f"\n=== Virage ===")
    for R_virage in [50, 100, 200]:
        v_max = virage_vitesse_max(R_virage, 0.7)
        print(f"  R = {R_virage}m, μ = 0.7 : v_max = {v_max:.1f} m/s = {v_max*3.6:.0f} km/h")

    print(f"\n=== Orbites satellites ===")
    for h in [200e3, 400e3, 35786e3]:
        orb = orbite_satellite(h)
        print(f"  h = {h/1e3:.0f} km : v = {orb['vitesse']:.0f} m/s, "
              f"T = {orb['periode_h']:.2f} h")
    print(f"  (h = 35786 km → T ≈ 24h = orbite géostationnaire)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tracer_mcu(R=1, omega=2, ax=axes[0])
    tracer_composantes(R=1, omega=2, ax=axes[1])
    plt.tight_layout()
    plt.savefig("circular_motion_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
