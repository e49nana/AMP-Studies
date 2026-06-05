"""
multiple_integrals.py
=====================

Intégrales multiples et changement de variables.

Couvre :
    - Intégrales doubles ∬ f(x,y) dA
    - Intégrales triples ∭ f(x,y,z) dV
    - Coordonnées polaires : dA = r dr dθ
    - Coordonnées cylindriques : dV = r dr dθ dz
    - Coordonnées sphériques : dV = r² sin φ dr dφ dθ
    - Applications : aire, volume, centre de masse, moments d'inertie

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


# ======================================================================
#  1. Intégrales doubles
# ======================================================================

def integrale_double(
    f: Callable, x_range: tuple, y_range_fn: Callable | tuple,
) -> float:
    """
    ∬ f(x,y) dA sur le domaine D.
    y_range_fn(x) = (y_min, y_max) ou constante.
    """
    if callable(y_range_fn):
        result, _ = integrate.dblquad(
            f, x_range[0], x_range[1],
            lambda x: y_range_fn(x)[0], lambda x: y_range_fn(x)[1],
        )
    else:
        result, _ = integrate.dblquad(
            f, x_range[0], x_range[1],
            lambda x: y_range_fn[0], lambda x: y_range_fn[1],
        )
    return result


def integrale_polaire(
    f_polar: Callable, r_range: tuple, theta_range: tuple = (0, 2*np.pi),
) -> float:
    """∬ f(r,θ) · r dr dθ (Jacobien = r)."""
    result, _ = integrate.dblquad(
        lambda theta, r: f_polar(r, theta) * r,
        r_range[0], r_range[1],
        lambda r: theta_range[0], lambda r: theta_range[1],
    )
    return result


# ======================================================================
#  2. Intégrales triples
# ======================================================================

def integrale_triple(
    f: Callable, x_range: tuple, y_range: tuple, z_range: tuple,
) -> float:
    """∭ f(x,y,z) dV sur un pavé."""
    result, _ = integrate.tplquad(
        f, x_range[0], x_range[1],
        lambda x: y_range[0], lambda x: y_range[1],
        lambda x, y: z_range[0], lambda x, y: z_range[1],
    )
    return result


def integrale_spherique(
    f_sph: Callable, r_range: tuple,
    phi_range: tuple = (0, np.pi), theta_range: tuple = (0, 2*np.pi),
) -> float:
    """∭ f(r,φ,θ) · r² sin φ dr dφ dθ (Jacobien = r² sin φ)."""
    result, _ = integrate.tplquad(
        lambda theta, phi, r: f_sph(r, phi, theta) * r**2 * np.sin(phi),
        r_range[0], r_range[1],
        lambda r: phi_range[0], lambda r: phi_range[1],
        lambda r, phi: theta_range[0], lambda r, phi: theta_range[1],
    )
    return result


# ======================================================================
#  3. Applications
# ======================================================================

def aire_domaine(x_range: tuple, y_range_fn: Callable | tuple) -> float:
    """A = ∬ 1 dA."""
    return integrale_double(lambda y, x: 1, x_range, y_range_fn)


def volume_solide(f_top: Callable, f_bot: Callable,
                    x_range: tuple, y_range: tuple) -> float:
    """V = ∬ (f_top - f_bot) dA."""
    return integrale_double(
        lambda y, x: f_top(x, y) - f_bot(x, y),
        x_range, y_range,
    )


def centre_de_masse_2d(
    rho: Callable, x_range: tuple, y_range: tuple,
) -> tuple[float, float]:
    """(x̄, ȳ) = (∬ xρ dA / M, ∬ yρ dA / M)."""
    M = integrale_double(lambda y, x: rho(x, y), x_range, y_range)
    Mx = integrale_double(lambda y, x: x * rho(x, y), x_range, y_range)
    My = integrale_double(lambda y, x: y * rho(x, y), x_range, y_range)
    return Mx / M, My / M


def moment_inertie_z(
    rho: Callable, x_range: tuple, y_range: tuple,
) -> float:
    """I_z = ∬ (x² + y²) ρ dA."""
    return integrale_double(
        lambda y, x: (x**2 + y**2) * rho(x, y),
        x_range, y_range,
    )


# ======================================================================
#  4. Tracés
# ======================================================================

def tracer_domaine_polaire(ax: plt.Axes | None = None) -> plt.Axes:
    """Visualise le passage cartésien → polaire."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    theta = np.linspace(0, 2*np.pi, 100)
    R = 1

    # Cercle
    ax.plot(R*np.cos(theta), R*np.sin(theta), "b-", linewidth=2)
    ax.fill(R*np.cos(theta), R*np.sin(theta), alpha=0.15, color="blue")

    # Grille polaire
    for r in np.linspace(0.2, 1, 5):
        ax.plot(r*np.cos(theta), r*np.sin(theta), "grey", linewidth=0.5, alpha=0.3)
    for th in np.linspace(0, 2*np.pi, 13)[:-1]:
        ax.plot([0, np.cos(th)], [0, np.sin(th)], "grey", linewidth=0.5, alpha=0.3)

    # Élément d'aire
    r0, th0, dr, dth = 0.6, np.pi/4, 0.15, np.pi/8
    th_arc = np.linspace(th0, th0+dth, 20)
    patch_x = np.concatenate([r0*np.cos(th_arc), (r0+dr)*np.cos(th_arc[::-1])])
    patch_y = np.concatenate([r0*np.sin(th_arc), (r0+dr)*np.sin(th_arc[::-1])])
    ax.fill(patch_x, patch_y, color="red", alpha=0.5)
    ax.annotate("$r\\,dr\\,d\\theta$", (r0*np.cos(th0+dth/2)*1.15,
                r0*np.sin(th0+dth/2)*1.15), fontsize=12, color="red")

    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title("Coordonnées polaires : $dA = r\\,dr\\,d\\theta$")
    return ax


def tracer_changement_variables(ax: plt.Axes | None = None) -> plt.Axes:
    """Compare intégrale cartésienne vs polaire pour le disque."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Aire du disque de rayon R par intégrale
    Rs = np.linspace(0.5, 3, 20)
    aires_pol = []
    aires_exact = []
    for R in Rs:
        A = integrale_polaire(lambda r, th: 1, (0, R))
        aires_pol.append(A)
        aires_exact.append(np.pi * R**2)

    ax.plot(Rs, aires_pol, "bo-", markersize=5, label="polaire (numérique)")
    ax.plot(Rs, aires_exact, "r--", linewidth=2, label="$\\pi R^2$ (exact)")
    ax.set_xlabel("$R$"); ax.set_ylabel("aire")
    ax.set_title("Aire du disque : $\\int_0^{2\\pi}\\int_0^R r\\,dr\\,d\\theta = \\pi R^2$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Intégrales doubles ===\n")

    # ∬ xy dA sur [0,1]×[0,1]
    I = integrale_double(lambda y, x: x*y, (0, 1), (0, 1))
    print(f"  ∬ xy dA sur [0,1]² = {I:.6f} (exact: 1/4 = 0.25)")

    # Aire du cercle unité
    A = integrale_polaire(lambda r, th: 1, (0, 1))
    print(f"  Aire disque R=1 : {A:.6f} (exact: π = {np.pi:.6f})")

    print(f"\n=== Changement de variables ===\n")
    # ∬ e^{-(x²+y²)} dA sur R² → polaire
    I_gauss = integrale_polaire(lambda r, th: np.exp(-r**2), (0, 10))
    print(f"  ∬ e^{{-(x²+y²)}} dA = {I_gauss:.6f} (exact: π = {np.pi:.6f})")
    print(f"  → Ceci prouve ∫ e^{{-x²}} dx = √π")

    print(f"\n=== Intégrales triples ===\n")
    # Volume du cube [0,1]³
    V = integrale_triple(lambda z, y, x: 1, (0, 1), (0, 1), (0, 1))
    print(f"  Volume cube [0,1]³ = {V:.6f}")

    # Volume de la sphère unité
    V_sphere = integrale_spherique(lambda r, phi, th: 1, (0, 1))
    print(f"  Volume sphère R=1 = {V_sphere:.6f} (exact: 4π/3 = {4*np.pi/3:.6f})")

    print(f"\n=== Centre de masse ===\n")
    # Plaque carrée [0,1]² avec ρ = x + 1
    cx, cy = centre_de_masse_2d(lambda x, y: x + 1, (0, 1), (0, 1))
    print(f"  ρ(x,y) = x+1 sur [0,1]² : centre = ({cx:.4f}, {cy:.4f})")
    print(f"  (décalé vers x car ρ croît avec x)")

    print(f"\n=== Moment d'inertie ===\n")
    # Disque uniforme ρ = 1, rayon R = 1
    Iz = integrale_polaire(lambda r, th: r**2, (0, 1))  # ∬ r² · r dr dθ
    print(f"  Disque R=1, ρ=1 : I_z = {Iz:.6f} (exact: π/2 = {np.pi/2:.6f})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tracer_domaine_polaire(ax=axes[0])
    tracer_changement_variables(ax=axes[1])
    plt.tight_layout()
    plt.savefig("multiple_integrals_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
