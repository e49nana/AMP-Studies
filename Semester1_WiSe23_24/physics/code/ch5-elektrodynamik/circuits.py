"""
circuits.py
===========

Circuits électriques : Ohm, Kirchhoff, RC/RL/RLC.

Couvre :
    - Loi d'Ohm : V = RI
    - Kirchhoff : loi des nœuds (Σ I = 0), loi des mailles (Σ V = 0)
    - Résistances en série et parallèle
    - Circuit RC : charge/décharge du condensateur
    - Circuit RL : établissement/coupure du courant
    - Circuit RLC : oscillations amorties
    - Simulation par RK4

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def rk4_scalar(f, t0, y0, t_end, h):
    n = int((t_end - t0) / h)
    t = np.zeros(n+1); y = np.zeros(n+1)
    t[0] = t0; y[0] = y0
    for k in range(n):
        k1 = f(t[k], y[k])
        k2 = f(t[k]+h/2, y[k]+h/2*k1)
        k3 = f(t[k]+h/2, y[k]+h/2*k2)
        k4 = f(t[k]+h, y[k]+h*k3)
        y[k+1] = y[k] + h/6*(k1+2*k2+2*k3+k4)
        t[k+1] = t[k]+h
    return t, y


def rk4_sys(f, t0, y0, t_end, h):
    y = np.array(y0, dtype=float)
    n = int((t_end-t0)/h)
    t = np.zeros(n+1); ys = np.zeros((n+1, len(y)))
    t[0] = t0; ys[0] = y
    for k in range(n):
        k1 = np.array(f(t[k], y))
        k2 = np.array(f(t[k]+h/2, y+h/2*k1))
        k3 = np.array(f(t[k]+h/2, y+h/2*k2))
        k4 = np.array(f(t[k]+h, y+h*k3))
        y = y + h/6*(k1+2*k2+2*k3+k4)
        t[k+1] = t[k]+h; ys[k+1] = y
    return t, ys


# ======================================================================
#  1. Résistances
# ======================================================================

def resistance_serie(*Rs: float) -> float:
    """R_tot = R₁ + R₂ + ..."""
    return sum(Rs)


def resistance_parallele(*Rs: float) -> float:
    """1/R_tot = 1/R₁ + 1/R₂ + ..."""
    return 1 / sum(1/R for R in Rs)


# ======================================================================
#  2. Circuit RC
# ======================================================================

def rc_charge(V_s: float, R: float, C: float, t: np.ndarray) -> dict:
    """
    Charge du condensateur :
        V_C(t) = V_s(1 - e^{-t/τ}), I(t) = (V_s/R)e^{-t/τ}.
    τ = RC.
    """
    tau = R * C
    V_C = V_s * (1 - np.exp(-t / tau))
    I = (V_s / R) * np.exp(-t / tau)
    return {"V_C": V_C, "I": I, "tau": tau}


def rc_decharge(V_0: float, R: float, C: float, t: np.ndarray) -> dict:
    """
    Décharge : V_C(t) = V₀ e^{-t/τ}.
    """
    tau = R * C
    V_C = V_0 * np.exp(-t / tau)
    I = -(V_0 / R) * np.exp(-t / tau)
    return {"V_C": V_C, "I": I, "tau": tau}


# ======================================================================
#  3. Circuit RL
# ======================================================================

def rl_etablissement(V_s: float, R: float, L: float, t: np.ndarray) -> dict:
    """
    Établissement du courant :
        I(t) = (V_s/R)(1 - e^{-t/τ}), V_L(t) = V_s e^{-t/τ}.
    τ = L/R.
    """
    tau = L / R
    I = (V_s / R) * (1 - np.exp(-t / tau))
    V_L = V_s * np.exp(-t / tau)
    return {"I": I, "V_L": V_L, "tau": tau}


# ======================================================================
#  4. Circuit RLC
# ======================================================================

def rlc_simulation(R: float, L: float, C: float, V_0: float, t_end: float,
                    h: float = 1e-5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Circuit RLC série : L·q'' + R·q' + q/C = 0.
    État = [q, I=q'].
        q' = I
        I' = (-R·I - q/C) / L
    """
    def f(t, state):
        q, I = state
        return np.array([I, (-R*I - q/C) / L])

    q0 = C * V_0  # charge initiale
    t, ys = rk4_sys(f, 0, [q0, 0], t_end, h)
    return t, ys[:, 0] / C, ys[:, 1]  # t, V_C, I


def rlc_regime(R: float, L: float, C: float) -> str:
    """Classifie le régime : sous-amorti, critique, sur-amorti."""
    omega0 = 1 / np.sqrt(L * C)
    gamma = R / (2 * L)
    if gamma < omega0:
        return f"sous-amorti (ω₀={omega0:.1f}, γ={gamma:.1f})"
    elif abs(gamma - omega0) < 0.01 * omega0:
        return f"critique (ω₀ = γ = {omega0:.1f})"
    else:
        return f"sur-amorti (ω₀={omega0:.1f}, γ={gamma:.1f})"


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_rc(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    R, C, Vs = 1000, 1e-3, 5.0
    tau = R * C
    t = np.linspace(0, 5*tau, 500)
    ch = rc_charge(Vs, R, C, t)

    ax.plot(t/tau, ch["V_C"], "b-", linewidth=2, label="$V_C$ (charge)")
    ax.plot(t/tau, Vs - ch["V_C"], "r--", linewidth=2, label="$V_R$")
    ax.axhline(Vs, color="grey", linestyle=":", alpha=0.3)
    ax.axhline(0.632*Vs, color="blue", linestyle=":", alpha=0.3)
    ax.axvline(1, color="green", linestyle=":", alpha=0.3, label=f"$t = \\tau = {tau}$ s")

    ax.set_xlabel("$t / \\tau$"); ax.set_ylabel("tension (V)")
    ax.set_title(f"Circuit RC : charge ($R={R}$ Ω, $C={C*1e6:.0f}$ μF)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_rlc(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    L, C = 0.1, 1e-4  # 100 mH, 100 μF
    V0 = 10

    for R, style in [(5, "-"), (20, "--"), (63, ":"), (100, "-.")]:
        t, V_C, I = rlc_simulation(R, L, C, V0, 0.1)
        regime = rlc_regime(R, L, C)
        ax.plot(t*1000, V_C, style, linewidth=2, label=f"R={R}Ω ({regime.split('(')[0].strip()})")

    ax.set_xlabel("$t$ (ms)"); ax.set_ylabel("$V_C$ (V)")
    ax.set_title(f"Circuit RLC ($L={L*1e3:.0f}$ mH, $C={C*1e6:.0f}$ μF)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Résistances ===\n")
    print(f"  Série (100, 200, 300) : {resistance_serie(100, 200, 300):.0f} Ω")
    print(f"  Parallèle (100, 200)  : {resistance_parallele(100, 200):.1f} Ω")

    print(f"\n=== Circuit RC (R=1kΩ, C=1mF) ===\n")
    R, C, Vs = 1000, 1e-3, 5
    tau = R * C
    print(f"  τ = RC = {tau} s")
    for n_tau in [1, 2, 3, 5]:
        V = Vs * (1 - np.exp(-n_tau))
        print(f"  V_C({n_tau}τ) = {V:.4f} V = {V/Vs*100:.1f}% de V_s")

    print(f"\n=== Circuit RL (V=12V, R=10Ω, L=0.5H) ===\n")
    R_rl, L, Vs_rl = 10, 0.5, 12
    tau_rl = L / R_rl
    print(f"  τ = L/R = {tau_rl} s")
    print(f"  I_max = V/R = {Vs_rl/R_rl} A")

    print(f"\n=== Circuit RLC ===\n")
    L, C = 0.1, 1e-4
    for R in [5, 20, 63, 100]:
        print(f"  R={R:>3}Ω : {rlc_regime(R, L, C)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_rc(ax=axes[0])
    tracer_rlc(ax=axes[1])
    plt.tight_layout()
    plt.savefig("circuits_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
