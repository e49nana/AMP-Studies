"""
ode_applications.py
===================

Applications physiques des EDO.

Couvre :
    - Croissance exponentielle et logistique
    - Pendule simple (linéaire et non-linéaire)
    - Circuit RC
    - Chute libre avec frottement
    - Comparaison solution numérique vs analytique

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def rk4(f, t0, y0, t_end, h):
    """RK4 scalaire ou vectoriel."""
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    n = int((t_end - t0) / h)
    t = np.zeros(n+1); y = np.zeros((n+1, len(y0)))
    t[0] = t0; y[0] = y0
    for k in range(n):
        k1 = np.atleast_1d(f(t[k], y[k]))
        k2 = np.atleast_1d(f(t[k]+h/2, y[k]+h/2*k1))
        k3 = np.atleast_1d(f(t[k]+h/2, y[k]+h/2*k2))
        k4 = np.atleast_1d(f(t[k]+h, y[k]+h*k3))
        y[k+1] = y[k] + h/6*(k1+2*k2+2*k3+k4)
        t[k+1] = t[k] + h
    return t, y


# ======================================================================
#  1. Croissance exponentielle et logistique
# ======================================================================

def croissance_exponentielle(r: float = 0.5):
    """y' = r·y → y(t) = y₀·e^{rt}."""
    return lambda t, y: r * y


def croissance_logistique(r: float = 0.5, K: float = 100):
    """y' = r·y·(1 - y/K) → y(t) = K / (1 + (K/y₀-1)·e^{-rt})."""
    return lambda t, y: r * y * (1 - y / K)


def solution_logistique(t, y0, r, K):
    """Solution analytique de la logistique."""
    return K / (1 + (K/y0 - 1) * np.exp(-r * t))


# ======================================================================
#  2. Pendule
# ======================================================================

def pendule_lineaire(g_L: float = 1.0):
    """θ'' + (g/L)θ = 0 → [θ', -(g/L)θ]."""
    return lambda t, y: np.array([y[1], -g_L * y[0]])


def pendule_nonlineaire(g_L: float = 1.0):
    """θ'' + (g/L)sin(θ) = 0 → [θ', -(g/L)sin(θ)]."""
    return lambda t, y: np.array([y[1], -g_L * np.sin(y[0])])


# ======================================================================
#  3. Circuit RC
# ======================================================================

def circuit_rc(R: float = 1000, C: float = 1e-3, V_source: float = 5.0):
    """
    Charge du condensateur : V_C' = (V_s - V_C) / (RC).
    Solution : V_C(t) = V_s · (1 - e^{-t/RC}).
    """
    tau = R * C
    return lambda t, y: (V_source - y) / tau


def solution_rc(t, R, C, V_source):
    tau = R * C
    return V_source * (1 - np.exp(-t / tau))


# ======================================================================
#  4. Chute libre avec frottement
# ======================================================================

def chute_frottement(g: float = 9.81, k: float = 0.1, m: float = 1.0):
    """
    v' = g - (k/m)·v² (frottement quadratique).
    Vitesse terminale : v_t = √(mg/k).
    """
    return lambda t, y: np.array([y[1], g - (k/m) * y[1]**2])


# ======================================================================
#  Tracés
# ======================================================================

def tracer_croissance(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    r, K, y0 = 0.5, 100, 5
    t, y_exp = rk4(croissance_exponentielle(r), 0, [y0], 15, 0.01)
    t, y_log = rk4(croissance_logistique(r, K), 0, [y0], 15, 0.01)
    t_fine = np.linspace(0, 15, 300)

    ax.plot(t, y_exp[:, 0], "r-", linewidth=2, label="exponentielle (sans limite)")
    ax.plot(t, y_log[:, 0], "b-", linewidth=2, label="logistique ($K=100$)")
    ax.plot(t_fine, solution_logistique(t_fine, y0, r, K), "g--", alpha=0.5, label="logistique (exacte)")
    ax.axhline(K, color="grey", linestyle=":", alpha=0.5, label=f"capacité $K={K}$")
    ax.set_xlabel("$t$"); ax.set_ylabel("population $y$")
    ax.set_title("Croissance exponentielle vs logistique")
    ax.set_ylim(0, 200)
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_pendule(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for theta0, nom in [(0.1, "petit angle"), (np.pi/2, "π/2"), (np.pi - 0.1, "≈π")]:
        t, y_lin = rk4(pendule_lineaire(), 0, [theta0, 0], 20, 0.01)
        t, y_nl = rk4(pendule_nonlineaire(), 0, [theta0, 0], 20, 0.01)
        if theta0 < 0.5:
            ax.plot(t, y_lin[:, 0], "b--", linewidth=1, alpha=0.5, label=f"linéaire θ₀={theta0:.1f}")
        ax.plot(t, y_nl[:, 0], linewidth=2, label=f"non-lin. θ₀={theta0:.1f}")

    ax.set_xlabel("$t$"); ax.set_ylabel("$\\theta$ (rad)")
    ax.set_title("Pendule : linéaire vs non-linéaire")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax


def tracer_circuit_rc(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    R, C, Vs = 1000, 1e-3, 5.0
    tau = R * C
    t, y = rk4(circuit_rc(R, C, Vs), 0, [0], 5*tau, tau/100)
    t_fine = np.linspace(0, 5*tau, 300)

    ax.plot(t, y[:, 0], "b-", linewidth=2, label="RK4")
    ax.plot(t_fine, solution_rc(t_fine, R, C, Vs), "r--", linewidth=1.5, label="exacte")
    ax.axhline(Vs, color="grey", linestyle=":", alpha=0.5, label=f"$V_s = {Vs}$ V")
    ax.axvline(tau, color="green", linestyle=":", alpha=0.5, label=f"$\\tau = RC = {tau}$ s")
    ax.set_xlabel("$t$ (s)"); ax.set_ylabel("$V_C$ (V)")
    ax.set_title(f"Circuit RC : $R={R}$ Ω, $C={C*1e6:.0f}$ μF")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Croissance logistique ===")
    r, K, y0 = 0.5, 100, 5
    t, y = rk4(croissance_logistique(r, K), 0, [y0], 20, 0.01)
    exact = solution_logistique(t, y0, r, K)
    print(f"  r={r}, K={K}, y₀={y0}")
    print(f"  y(10) = {y[1000, 0]:.4f} (exact: {exact[1000]:.4f})")
    print(f"  y(20) = {y[-1, 0]:.4f} → K = {K}")

    print(f"\n=== Pendule ===")
    for theta0 in [0.1, np.pi/4, np.pi/2]:
        t, y = rk4(pendule_nonlineaire(), 0, [theta0, 0], 20, 0.001)
        # Estimer la période
        crossings = np.where(np.diff(np.sign(y[:, 0])))[0]
        if len(crossings) >= 2:
            T = 2 * (t[crossings[1]] - t[crossings[0]])
        else:
            T = float("nan")
        T_lin = 2 * np.pi  # période linéaire
        print(f"  θ₀ = {theta0:.2f} rad : T = {T:.4f} (linéaire: {T_lin:.4f}, "
              f"écart {abs(T-T_lin)/T_lin*100:.1f}%)")

    print(f"\n=== Circuit RC ===")
    R, C, Vs = 1000, 1e-3, 5.0
    tau = R * C
    t, y = rk4(circuit_rc(R, C, Vs), 0, [0], 5*tau, tau/100)
    print(f"  τ = RC = {tau} s")
    print(f"  V_C(τ) = {y[100, 0]:.4f} V (exact: {Vs*(1-np.exp(-1)):.4f} = 63.2% de V_s)")
    print(f"  V_C(5τ) = {y[-1, 0]:.4f} V (≈ V_s = {Vs} V)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_croissance(ax=axes[0])
    tracer_pendule(ax=axes[1])
    tracer_circuit_rc(ax=axes[2])
    plt.tight_layout()
    plt.savefig("ode_applications_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
