"""
electromagnetic_waves.py
========================

Ondes électromagnétiques et spectre.

Couvre :
    - Équations de Maxwell (résumé qualitatif)
    - Onde EM plane : E et B perpendiculaires, en phase, c = 1/√(μ₀ε₀)
    - Relation E/B = c
    - Spectre électromagnétique (radio → gamma)
    - Énergie : vecteur de Poynting S = E × B / μ₀
    - Intensité et pression de radiation

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


C = 2.998e8           # vitesse de la lumière (m/s)
MU0 = 4*np.pi*1e-7
E0 = 8.854e-12


def vitesse_lumiere() -> float:
    """c = 1/√(μ₀ε₀). Vérification numérique."""
    return 1 / np.sqrt(MU0 * E0)


def longueur_onde(f: float) -> float:
    """λ = c/f."""
    return C / f


def frequence(lam: float) -> float:
    """f = c/λ."""
    return C / lam


def energie_photon(f: float) -> float:
    """E = hf."""
    h = 6.626e-34
    return h * f


def intensite_onde(E0_amp: float) -> float:
    """I = ½ε₀cE₀² (intensité moyenne)."""
    return 0.5 * E0 * C * E0_amp**2


def pression_radiation(I: float, absorption: bool = True) -> float:
    """P_rad = I/c (absorption) ou 2I/c (réflexion)."""
    return I / C if absorption else 2 * I / C


# ======================================================================
#  Spectre électromagnétique
# ======================================================================

SPECTRE = [
    ("Radio", 1e3, 1e9),
    ("Micro-ondes", 1e9, 3e11),
    ("Infrarouge", 3e11, 4.3e14),
    ("Visible", 4.3e14, 7.5e14),
    ("Ultraviolet", 7.5e14, 3e16),
    ("Rayons X", 3e16, 3e19),
    ("Rayons γ", 3e19, 3e22),
]

VISIBLE = [
    ("rouge", 700e-9, 620e-9),
    ("orange", 620e-9, 590e-9),
    ("jaune", 590e-9, 570e-9),
    ("vert", 570e-9, 495e-9),
    ("bleu", 495e-9, 450e-9),
    ("violet", 450e-9, 380e-9),
]


def afficher_spectre() -> None:
    """Affiche le spectre EM avec λ et E."""
    print(f"  {'Domaine':>14} | {'f_min':>10} | {'f_max':>10} | {'λ_max':>10} | {'λ_min':>10}")
    print("  " + "-" * 60)
    for nom, f_min, f_max in SPECTRE:
        lam_max = longueur_onde(f_min)
        lam_min = longueur_onde(f_max)
        print(f"  {nom:>14} | {f_min:>10.2e} | {f_max:>10.2e} | "
              f"{lam_max:>10.2e} | {lam_min:>10.2e}")


# ======================================================================
#  Tracés
# ======================================================================

def tracer_onde_em(ax: plt.Axes | None = None) -> plt.Axes:
    """Visualise une onde EM plane (E et B perpendiculaires)."""
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")

    z = np.linspace(0, 4*np.pi, 300)
    E = np.sin(z)
    B = np.sin(z)

    ax.plot(z, E, np.zeros_like(z), "b-", linewidth=2, label="$\\vec{E}$ (vertical)")
    ax.plot(z, np.zeros_like(z), B, "r-", linewidth=2, label="$\\vec{B}$ (horizontal)")
    ax.plot(z, np.zeros_like(z), np.zeros_like(z), "k--", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("$z$ (propagation)"); ax.set_ylabel("$E$"); ax.set_zlabel("$B$")
    ax.set_title("Onde EM plane : $E \\perp B \\perp$ propagation")
    ax.legend()
    return ax


def tracer_spectre(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    colors = ["grey", "brown", "red", "rainbow", "violet", "blue", "black"]
    for i, (nom, f_min, f_max) in enumerate(SPECTRE):
        ax.barh(0, np.log10(f_max) - np.log10(f_min), left=np.log10(f_min),
                height=0.5, alpha=0.6, color=plt.cm.rainbow(i/6),
                edgecolor="black", linewidth=0.5)
        ax.text((np.log10(f_min) + np.log10(f_max))/2, 0, nom,
                ha="center", va="center", fontsize=8, fontweight="bold")

    ax.set_xlabel("$\\log_{10}(f)$ (Hz)")
    ax.set_yticks([]); ax.set_title("Spectre électromagnétique")
    ax.grid(True, alpha=0.3, axis="x")
    return ax


def tracer_poynting(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    E0_range = np.linspace(0, 1000, 300)
    I = [intensite_onde(E) for E in E0_range]

    ax.plot(E0_range, [i/1e3 for i in I], "b-", linewidth=2)
    ax.set_xlabel("$E_0$ (V/m)"); ax.set_ylabel("Intensité $I$ (kW/m²)")
    ax.set_title("$I = \\frac{1}{2}\\varepsilon_0 c E_0^2$")
    ax.grid(True, alpha=0.3)

    # Annoter le soleil
    I_soleil = 1361  # W/m²
    E_soleil = np.sqrt(2*I_soleil / (E0*C))
    ax.axhline(I_soleil/1e3, color="orange", linestyle="--", alpha=0.5,
                label=f"Soleil ({I_soleil} W/m²)")
    ax.legend()
    return ax


if __name__ == "__main__":
    print("=== Vitesse de la lumière ===\n")
    c_calc = vitesse_lumiere()
    print(f"  c = 1/√(μ₀ε₀) = {c_calc:.6e} m/s")
    print(f"  c exact        = {C:.6e} m/s")
    print(f"  erreur         = {abs(c_calc - C)/C:.2e}")

    print(f"\n=== Spectre EM ===\n")
    afficher_spectre()

    print(f"\n=== Lumière visible ===\n")
    for nom, lam_max, lam_min in VISIBLE:
        E = energie_photon(frequence(lam_min))
        print(f"  {nom:>6} : {lam_min*1e9:.0f}-{lam_max*1e9:.0f} nm, "
              f"E = {E/E_CHARGE:.2f} eV" if 'E_CHARGE' in dir() else "")

    print(f"\n=== Énergie et pression ===\n")
    I_soleil = 1361  # constante solaire
    print(f"  Intensité solaire : {I_soleil} W/m²")
    print(f"  E₀ = √(2I/ε₀c) = {np.sqrt(2*I_soleil/(E0*C)):.1f} V/m")
    print(f"  Pression (absorption) : {pression_radiation(I_soleil)*1e6:.2f} μPa")
    print(f"  Pression (réflexion)  : {pression_radiation(I_soleil, False)*1e6:.2f} μPa")

    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(221, projection="3d")
    tracer_onde_em(ax1)
    ax2 = fig.add_subplot(222)
    tracer_poynting(ax2)
    ax3 = fig.add_subplot(212)
    tracer_spectre(ax3)
    plt.tight_layout()
    plt.savefig("electromagnetic_waves_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
