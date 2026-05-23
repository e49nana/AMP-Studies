"""
induction.py
============

Induction électromagnétique : Faraday, Lenz, inductance.

Couvre :
    - Loi de Faraday : ε = -dΦ_B/dt
    - Loi de Lenz : le courant induit s'oppose à la variation de flux
    - Inductance propre : L = Φ/I, ε = -L dI/dt
    - Inductance d'un solénoïde : L = μ₀n²V
    - Énergie magnétique : U = ½LI²
    - Transformateur idéal : V₂/V₁ = N₂/N₁

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


MU0 = 4 * np.pi * 1e-7


# ======================================================================
#  1. Flux et FEM
# ======================================================================

def flux_magnetique(B: float, A: float, theta: float = 0) -> float:
    """Φ = B·A·cos(θ)."""
    return B * A * np.cos(theta)


def fem_faraday(dPhi_dt: float) -> float:
    """ε = -dΦ/dt."""
    return -dPhi_dt


def fem_generateur(B: float, A: float, omega: float, t: np.ndarray, N: int = 1) -> np.ndarray:
    """
    Générateur AC : bobine de N tours tournant dans un champ B.
        ε(t) = NBAω sin(ωt).
    """
    return N * B * A * omega * np.sin(omega * t)


# ======================================================================
#  2. Inductance
# ======================================================================

def inductance_solenoide(n: float, A: float, l: float) -> float:
    """L = μ₀n²·A·l pour un solénoïde (n tours/m, A section, l longueur)."""
    return MU0 * n**2 * A * l


def energie_magnetique(L: float, I: float) -> float:
    """U = ½LI²."""
    return 0.5 * L * I**2


def densite_energie_magnetique(B: float) -> float:
    """u = B²/(2μ₀) (J/m³)."""
    return B**2 / (2 * MU0)


# ======================================================================
#  3. Transformateur
# ======================================================================

def transformateur(V1: float, N1: int, N2: int, I1: float = None) -> dict:
    """
    Transformateur idéal :
        V₂/V₁ = N₂/N₁, I₂/I₁ = N₁/N₂, P₁ = P₂.
    """
    ratio = N2 / N1
    V2 = V1 * ratio
    result = {"V1": V1, "V2": V2, "ratio": ratio, "N1": N1, "N2": N2}
    if I1 is not None:
        I2 = I1 / ratio
        result["I1"] = I1
        result["I2"] = I2
        result["P"] = V1 * I1
    return result


# ======================================================================
#  4. Courant de Foucault et applications
# ======================================================================

def fem_rail(B: float, L: float, v: float) -> float:
    """FEM induite dans un rail : ε = BLv."""
    return B * L * v


def freinage_magnetique(B: float, L: float, v: float, R: float, m: float) -> dict:
    """
    Force de freinage sur un rail conducteur.
    ε = BLv, I = ε/R, F = BIL = B²L²v/R.
    a = -B²L²v/(mR) → décroissance exponentielle de v.
    """
    eps = B * L * v
    I = eps / R
    F = B * I * L
    tau = m * R / (B**2 * L**2)
    return {"eps": eps, "I": I, "F_frein": F, "tau": tau}


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_generateur(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    B, A, N = 0.5, 0.01, 100  # 0.5 T, 100 cm², 100 tours
    for f in [50, 60, 100]:
        omega = 2 * np.pi * f
        t = np.linspace(0, 3/f, 500)
        eps = fem_generateur(B, A, omega, t, N)
        eps_max = N * B * A * omega
        ax.plot(t*1000, eps, linewidth=2, label=f"$f = {f}$ Hz ($\\varepsilon_{{max}} = {eps_max:.1f}$ V)")

    ax.set_xlabel("$t$ (ms)"); ax.set_ylabel("$\\varepsilon$ (V)")
    ax.set_title("Générateur AC : $\\varepsilon = NBA\\omega \\sin(\\omega t)$")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_energie_inductance(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    I_range = np.linspace(0, 10, 200)
    for L_mH in [1, 10, 100]:
        L = L_mH * 1e-3
        U = [energie_magnetique(L, I) for I in I_range]
        ax.plot(I_range, U, linewidth=2, label=f"$L = {L_mH}$ mH")

    ax.set_xlabel("$I$ (A)"); ax.set_ylabel("$U$ (J)")
    ax.set_title("Énergie magnétique $U = \\frac{1}{2}LI^2$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_transformateur(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    N1 = 100
    N2_range = np.arange(10, 500)
    V1 = 230

    ax.plot(N2_range/N1, V1 * N2_range/N1, "b-", linewidth=2, label="$V_2$")
    ax.axhline(V1, color="red", linestyle="--", alpha=0.5, label=f"$V_1 = {V1}$ V")
    ax.axvline(1, color="grey", linestyle=":", alpha=0.3, label="$N_2/N_1 = 1$")

    ax.set_xlabel("$N_2 / N_1$"); ax.set_ylabel("$V_2$ (V)")
    ax.set_title("Transformateur : $V_2 = V_1 \\cdot N_2/N_1$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Loi de Faraday ===\n")
    B, A = 0.5, 0.01
    Phi = flux_magnetique(B, A)
    print(f"  Φ = BA = {B}×{A} = {Phi:.4f} Wb")
    print(f"  Si Φ passe de {Phi} à 0 en 0.01s : ε = {fem_faraday(-Phi/0.01):.2f} V")

    print(f"\n=== Générateur AC ===\n")
    B, A, N, f = 0.5, 0.01, 100, 50
    omega = 2*np.pi*f
    eps_max = N*B*A*omega
    print(f"  N={N}, B={B}T, A={A*1e4}cm², f={f}Hz")
    print(f"  ε_max = NBAω = {eps_max:.1f} V")

    print(f"\n=== Inductance (solénoïde) ===\n")
    n, A_sol, l = 1000, 1e-4, 0.1
    L = inductance_solenoide(n, A_sol, l)
    print(f"  n={n}/m, A={A_sol*1e4}cm², l={l*100}cm")
    print(f"  L = μ₀n²Al = {L*1e3:.4f} mH")
    print(f"  U(I=1A) = {energie_magnetique(L, 1)*1e3:.4f} mJ")

    print(f"\n=== Transformateur ===\n")
    for N2 in [50, 100, 200, 1000]:
        r = transformateur(230, 100, N2, 1)
        print(f"  N₂={N2:>4} : V₂={r['V2']:>7.1f} V, I₂={r['I2']:.3f} A")

    print(f"\n=== Freinage magnétique ===\n")
    r = freinage_magnetique(1.0, 0.5, 10, 0.1, 1.0)
    print(f"  B=1T, L=0.5m, v=10m/s, R=0.1Ω, m=1kg")
    print(f"  ε = {r['eps']:.1f} V, I = {r['I']:.1f} A, F = {r['F_frein']:.1f} N")
    print(f"  τ = mR/(B²L²) = {r['tau']:.3f} s")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_generateur(ax=axes[0])
    tracer_energie_inductance(ax=axes[1])
    tracer_transformateur(ax=axes[2])
    plt.tight_layout()
    plt.savefig("induction_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
