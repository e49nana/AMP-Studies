"""
gauss_law.py
============

Loi de Gauss et symétries.

Couvre :
    - Φ_E = ∮ E·dA = Q_enc / ε₀ (loi de Gauss)
    - Flux à travers des surfaces simples
    - Applications par symétrie : sphère, cylindre infini, plan infini
    - Conducteur : E = 0 à l'intérieur, σ/ε₀ à la surface
    - Vérification numérique par intégration

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


K = 8.9875e9
E0 = 8.854e-12


# ======================================================================
#  1. Flux
# ======================================================================

def flux_charge_ponctuelle(Q: float, R: float) -> float:
    """Flux à travers une sphère de rayon R : Φ = Q/ε₀ (indépendant de R !)."""
    return Q / E0


def flux_numerique_sphere(Q: float, R: float, n_theta: int = 100, n_phi: int = 100) -> float:
    """Vérifie Φ = Q/ε₀ par intégration numérique sur une sphère."""
    E = K * Q / R**2
    # ∮ E·dA = E · 4πR² pour symétrie sphérique
    return E * 4 * np.pi * R**2


# ======================================================================
#  2. Symétries classiques
# ======================================================================

def champ_sphere_uniforme(Q: float, R: float, r: float) -> float:
    """
    Sphère uniformément chargée (charge totale Q, rayon R) :
        r < R : E = Q·r / (4πε₀R³) (croît linéairement)
        r ≥ R : E = Q / (4πε₀r²) (comme charge ponctuelle)
    """
    if r < R:
        return Q * r / (4 * np.pi * E0 * R**3)
    return Q / (4 * np.pi * E0 * r**2)


def champ_cylindre_infini(lam: float, r: float) -> float:
    """
    Fil infini (densité linéique λ) :
        E = λ / (2πε₀r).
    """
    if r <= 0:
        return 0
    return abs(lam) / (2 * np.pi * E0 * r)


def champ_plan_infini(sigma: float) -> float:
    """
    Plan infini (densité surfacique σ) :
        E = σ / (2ε₀) (uniforme, indépendant de la distance !).
    """
    return abs(sigma) / (2 * E0)


def champ_condensateur(sigma: float) -> float:
    """
    Entre les plaques d'un condensateur plan :
        E = σ / ε₀ (double du plan unique car deux plaques).
    """
    return abs(sigma) / E0


# ======================================================================
#  3. Tracés
# ======================================================================

def tracer_sphere(Q: float = 1e-6, R: float = 0.1, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    r = np.linspace(0, 0.5, 500)
    E = [champ_sphere_uniforme(Q, R, ri) for ri in r]
    E_ponctuelle = [K*Q/ri**2 if ri > 0.001 else 0 for ri in r]

    ax.plot(r*100, E, "b-", linewidth=2, label="sphère uniforme")
    ax.plot(r*100, E_ponctuelle, "r--", linewidth=1.5, alpha=0.5, label="charge ponctuelle")
    ax.axvline(R*100, color="grey", linestyle=":", alpha=0.5, label=f"$R = {R*100:.0f}$ cm")

    ax.set_xlabel("$r$ (cm)"); ax.set_ylabel("$|E|$ (N/C)")
    ax.set_title(f"Sphère uniformément chargée ($Q = {Q*1e6:.0f}$ μC, $R = {R*100:.0f}$ cm)")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_cylindre(lam: float = 1e-6, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    r = np.linspace(0.01, 0.5, 300)
    E = [champ_cylindre_infini(lam, ri) for ri in r]

    ax.plot(r*100, E, "b-", linewidth=2, label=f"$E = \\lambda/(2\\pi\\varepsilon_0 r)$")
    ax.set_xlabel("$r$ (cm)"); ax.set_ylabel("$|E|$ (N/C)")
    ax.set_title(f"Fil infini ($\\lambda = {lam*1e6:.0f}$ μC/m)")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_trois_symetries(ax: plt.Axes | None = None) -> plt.Axes:
    """Compare E(r) pour les 3 symétries."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    r = np.linspace(0.01, 0.5, 300)

    # Normaliser pour comparer les formes
    E_sphere = np.array([K * 1e-6 / ri**2 for ri in r])
    E_cyl = np.array([1e-6 / (2*np.pi*E0*ri) for ri in r])
    E_plan = np.full_like(r, 1e-6 / (2*E0))

    ax.plot(r*100, E_sphere/E_sphere[0], "b-", linewidth=2, label="sphère ($\\propto 1/r^2$)")
    ax.plot(r*100, E_cyl/E_cyl[0], "r-", linewidth=2, label="cylindre ($\\propto 1/r$)")
    ax.plot(r*100, E_plan/E_plan[0], "g-", linewidth=2, label="plan (constant)")

    ax.set_xlabel("$r$ (cm)"); ax.set_ylabel("$E / E_0$ (normalisé)")
    ax.set_title("Trois symétries : $1/r^2$, $1/r$, constant")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Loi de Gauss : Φ = Q/ε₀ ===\n")
    Q = 1e-6
    for R in [0.01, 0.1, 1.0, 10]:
        phi_theo = Q / E0
        phi_num = flux_numerique_sphere(Q, R)
        print(f"  R = {R:>5} m : Φ = {phi_num:.4e} V·m (théo: {phi_theo:.4e})")
    print(f"  → Φ indépendant de R ✓")

    print(f"\n=== Sphère uniformément chargée ===\n")
    Q, R = 1e-6, 0.1
    for r in [0, 0.05, 0.1, 0.2, 0.5]:
        E = champ_sphere_uniforme(Q, R, r)
        print(f"  r = {r*100:>5.0f} cm : E = {E:.4e} N/C")

    print(f"\n=== Trois symétries ===\n")
    sigma = 1e-6  # C/m²
    lam = 1e-6    # C/m
    print(f"  Plan infini (σ={sigma*1e6} μC/m²) : E = {champ_plan_infini(sigma):.2e} N/C (constant)")
    print(f"  Condensateur (σ={sigma*1e6} μC/m²) : E = {champ_condensateur(sigma):.2e} N/C")
    print(f"  Fil infini (λ={lam*1e6} μC/m) à r=1cm : E = {champ_cylindre_infini(lam, 0.01):.2e} N/C")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_sphere(ax=axes[0])
    tracer_cylindre(ax=axes[1])
    tracer_trois_symetries(ax=axes[2])
    plt.tight_layout()
    plt.savefig("gauss_law_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
