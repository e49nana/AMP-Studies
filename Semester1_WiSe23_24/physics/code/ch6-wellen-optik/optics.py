"""
optics.py
=========

Optique géométrique : lentilles, miroirs, Snell-Descartes.

Couvre :
    - Loi de Snell-Descartes : n₁ sin θ₁ = n₂ sin θ₂
    - Angle critique et réflexion totale interne
    - Lentilles minces : 1/f = 1/p + 1/q (formule de conjugaison)
    - Grandissement : γ = -q/p
    - Miroirs sphériques : même formule avec conventions de signe
    - Construction géométrique de l'image
    - Fibres optiques : angle d'acceptance

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Réfraction (Snell-Descartes)
# ======================================================================

def snell(n1: float, theta1: float, n2: float) -> float | None:
    """
    n₁ sin θ₁ = n₂ sin θ₂ → θ₂ = arcsin(n₁ sin θ₁ / n₂).
    Renvoie None si réflexion totale.
    """
    sin_theta2 = n1 * np.sin(theta1) / n2
    if abs(sin_theta2) > 1:
        return None  # réflexion totale
    return np.arcsin(sin_theta2)


def angle_critique(n1: float, n2: float) -> float | None:
    """θ_c = arcsin(n₂/n₁). Existe seulement si n₁ > n₂."""
    if n1 <= n2:
        return None
    return np.arcsin(n2 / n1)


def angle_brewster(n1: float, n2: float) -> float:
    """θ_B = arctan(n₂/n₁). Angle de polarisation."""
    return np.arctan(n2 / n1)


# ======================================================================
#  2. Lentilles minces
# ======================================================================

@dataclass
class ImageResult:
    """Résultat de la construction d'image."""
    q: float          # position de l'image
    gamma: float      # grandissement
    reelle: bool      # image réelle (q > 0) ou virtuelle (q < 0)
    droite: bool      # image droite (γ > 0) ou renversée (γ < 0)
    taille_image: float


def lentille_mince(f: float, p: float, taille_objet: float = 1.0) -> ImageResult:
    """
    Formule de conjugaison : 1/f = 1/p + 1/q → q = fp/(p-f).
    Convention : p > 0 (objet réel), f > 0 (convergente), f < 0 (divergente).
    """
    if abs(p - f) < 1e-15:
        q = float("inf")
        gamma = float("inf")
    else:
        q = f * p / (p - f)
        gamma = -q / p

    return ImageResult(
        q=q,
        gamma=gamma,
        reelle=q > 0,
        droite=gamma > 0,
        taille_image=abs(gamma) * taille_objet,
    )


def puissance_dioptrique(f: float) -> float:
    """D = 1/f (en dioptries, si f en mètres)."""
    return 1 / f


def lentilles_accolees(*fs: float) -> float:
    """1/f_eq = 1/f₁ + 1/f₂ + ... (lentilles accolées)."""
    return 1 / sum(1/f for f in fs)


# ======================================================================
#  3. Miroirs sphériques
# ======================================================================

def miroir_spherique(R: float, p: float, taille_objet: float = 1.0) -> ImageResult:
    """
    Miroir sphérique : f = R/2.
    Convention : R > 0 (concave), R < 0 (convexe).
    """
    f = R / 2
    return lentille_mince(f, p, taille_objet)


# ======================================================================
#  4. Fibres optiques
# ======================================================================

def ouverture_numerique(n_coeur: float, n_gaine: float) -> float:
    """ON = √(n_coeur² - n_gaine²) = n_air · sin(θ_max)."""
    return np.sqrt(n_coeur**2 - n_gaine**2)


def angle_acceptance(n_coeur: float, n_gaine: float) -> float:
    """θ_max = arcsin(ON / n_air)."""
    ON = ouverture_numerique(n_coeur, n_gaine)
    return np.arcsin(min(ON, 1))


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_snell(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    theta1 = np.linspace(0, np.pi/2 - 0.01, 200)

    for n1, n2, nom in [(1, 1.5, "air→verre"), (1.5, 1, "verre→air"), (1, 1.33, "air→eau")]:
        theta2 = []
        for th in theta1:
            th2 = snell(n1, th, n2)
            theta2.append(np.degrees(th2) if th2 is not None else np.nan)
        ax.plot(np.degrees(theta1), theta2, linewidth=2, label=f"{nom} ({n1}→{n2})")

    tc = angle_critique(1.5, 1)
    if tc:
        ax.axvline(np.degrees(tc), color="red", linestyle=":", alpha=0.5,
                    label=f"θ_c (verre→air) = {np.degrees(tc):.1f}°")

    ax.plot([0, 90], [0, 90], "k:", alpha=0.2)
    ax.set_xlabel("$\\theta_1$ (°)"); ax.set_ylabel("$\\theta_2$ (°)")
    ax.set_title("Loi de Snell-Descartes")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_lentille_image(
    f: float, p: float, taille: float = 1.0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    img = lentille_mince(f, p, taille)
    q = img.q
    gamma = img.gamma

    # Axe optique
    x_range = max(abs(p), abs(q), abs(f)) * 1.5
    ax.axhline(0, color="black", linewidth=1)

    # Lentille
    ax.plot([0, 0], [-1.5, 1.5], "b-", linewidth=3, alpha=0.5)
    ax.plot(f, 0, "bv", markersize=10, label=f"F (f={f:.1f})")
    ax.plot(-f, 0, "b^", markersize=10, label=f"F'")

    # Objet
    ax.annotate("", xy=(-p, taille), xytext=(-p, 0),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax.text(-p, taille*1.1, "objet", ha="center", color="green", fontsize=10)

    # Image
    if np.isfinite(q):
        img_color = "red" if img.reelle else "orange"
        style = "->" if img.reelle else "->"
        ax.annotate("", xy=(q, gamma*taille), xytext=(q, 0),
                    arrowprops=dict(arrowstyle=style, color=img_color, lw=2,
                                     linestyle="-" if img.reelle else "--"))
        typ = "réelle" if img.reelle else "virtuelle"
        ax.text(q, gamma*taille*1.1, f"image ({typ})", ha="center",
                color=img_color, fontsize=10)

    # Rayons principaux
    if np.isfinite(q):
        # Rayon parallèle → passe par F'
        ax.plot([-p, 0, q], [taille, taille, gamma*taille], "r-", linewidth=0.8, alpha=0.5)
        # Rayon par le centre
        ax.plot([-p, q], [taille, gamma*taille], "g-", linewidth=0.8, alpha=0.5)

    ax.set_xlim(-x_range, x_range)
    ylim = max(abs(taille), abs(gamma*taille) if np.isfinite(gamma) else 1) * 1.5
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel("$x$ (unités de f)"); ax.set_ylabel("$y$")
    ax.set_title(f"Lentille : $f={f}$, $p={p}$, $q={q:.1f}$, $\\gamma={gamma:.2f}$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax


def tracer_conjugaison(f: float = 10, ax: plt.Axes | None = None) -> plt.Axes:
    """Trace q vs p pour une lentille convergente."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    p = np.linspace(f*1.01, 5*f, 300)
    q = f * p / (p - f)

    ax.plot(p/f, q/f, "b-", linewidth=2, label="$q(p)$")
    ax.axhline(1, color="red", linestyle=":", alpha=0.3, label="$q = f$")
    ax.axvline(1, color="green", linestyle=":", alpha=0.3, label="$p = f$")
    ax.axvline(2, color="grey", linestyle=":", alpha=0.3)
    ax.axhline(2, color="grey", linestyle=":", alpha=0.3, label="$p = q = 2f$")

    ax.set_xlabel("$p / f$"); ax.set_ylabel("$q / f$")
    ax.set_title("Formule de conjugaison : $1/f = 1/p + 1/q$")
    ax.set_xlim(1, 5); ax.set_ylim(1, 10)
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Snell-Descartes ===\n")
    for n1, n2, th1_deg in [(1, 1.5, 30), (1, 1.5, 60), (1.5, 1, 30), (1.5, 1, 42)]:
        th2 = snell(n1, np.radians(th1_deg), n2)
        result = f"{np.degrees(th2):.1f}°" if th2 is not None else "réflexion totale !"
        print(f"  n₁={n1}, n₂={n2}, θ₁={th1_deg}° → θ₂ = {result}")

    tc = angle_critique(1.5, 1)
    print(f"\n  Angle critique (verre→air) : {np.degrees(tc):.1f}°")
    print(f"  Angle de Brewster (air→verre) : {np.degrees(angle_brewster(1, 1.5)):.1f}°")

    print(f"\n=== Lentille convergente (f = 10 cm) ===\n")
    f = 10
    for p in [30, 20, 15, 10, 5]:
        img = lentille_mince(f, p)
        print(f"  p={p:>2} : q={img.q:>7.1f}, γ={img.gamma:>6.2f}, "
              f"{'réelle' if img.reelle else 'virtuelle':>9}, "
              f"{'droite' if img.droite else 'renversée'}")

    print(f"\n=== Fibre optique ===\n")
    n_c, n_g = 1.48, 1.46
    ON = ouverture_numerique(n_c, n_g)
    th = angle_acceptance(n_c, n_g)
    print(f"  n_coeur={n_c}, n_gaine={n_g}")
    print(f"  ON = {ON:.4f}, θ_max = {np.degrees(th):.1f}°")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tracer_snell(ax=axes[0, 0])
    tracer_conjugaison(ax=axes[0, 1])
    tracer_lentille_image(10, 25, 2, ax=axes[1, 0])
    tracer_lentille_image(10, 5, 2, ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig("optics_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
