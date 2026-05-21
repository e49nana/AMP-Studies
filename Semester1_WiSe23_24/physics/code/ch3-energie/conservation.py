"""
conservation.py
===============

Conservation de l'énergie mécanique et diagrammes de potentiel.

Couvre :
    - E_mec = E_cin + E_pot = constante (si forces conservatives)
    - Diagramme E_pot(x) : analyse qualitative du mouvement
    - Points d'équilibre : stable (minimum), instable (maximum)
    - Pendule : échange E_cin ↔ E_pot
    - Montagne russe : hauteur minimale pour passer un looping
    - Perte d'énergie par frottement

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


G = 9.81


# ======================================================================
#  1. Conservation
# ======================================================================

def energie_mecanique(m: float, v: float, h: float, g: float = G) -> dict:
    """E_mec = E_cin + E_pot."""
    Ecin = 0.5 * m * v**2
    Epot = m * g * h
    return {"E_cin": Ecin, "E_pot": Epot, "E_mec": Ecin + Epot}


def vitesse_depuis_hauteur(h_depart: float, h: float, v0: float = 0, g: float = G) -> float:
    """v = √(v₀² + 2g(h_depart - h)). Conservation E_mec."""
    v2 = v0**2 + 2 * g * (h_depart - h)
    return np.sqrt(max(v2, 0))


# ======================================================================
#  2. Diagramme de potentiel
# ======================================================================

def analyser_potentiel(
    Ep: Callable, x_range: tuple[float, float], E_total: float, n_points: int = 1000,
) -> dict:
    """
    Analyse qualitative du mouvement dans un potentiel Ep(x).
    Avec E_total fixée, le mouvement est possible là où Ep(x) ≤ E_total.
    """
    x = np.linspace(*x_range, n_points)
    Ep_vals = np.array([Ep(xi) for xi in x])

    # Points de rebroussement : Ep(x) = E_total
    accessible = Ep_vals <= E_total

    # Équilibres : Ep'(x) = 0
    dEp = np.gradient(Ep_vals, x)
    d2Ep = np.gradient(dEp, x)

    equilibres = []
    for i in range(1, len(dEp) - 1):
        if dEp[i-1] * dEp[i] < 0:
            x_eq = x[i]
            if d2Ep[i] > 0:
                equilibres.append((x_eq, "stable"))
            else:
                equilibres.append((x_eq, "instable"))

    return {
        "x": x,
        "Ep": Ep_vals,
        "accessible": accessible,
        "equilibres": equilibres,
    }


def tracer_diagramme_potentiel(
    Ep: Callable, x_range: tuple[float, float],
    E_totals: list[float], nom: str = "E_p(x)",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    x = np.linspace(*x_range, 500)
    Ep_vals = [Ep(xi) for xi in x]
    ax.plot(x, Ep_vals, "b-", linewidth=2.5, label="$E_p(x)$")

    colors = plt.cm.Set1(np.linspace(0, 0.5, len(E_totals)))
    for E, c in zip(E_totals, colors):
        ax.axhline(E, color=c, linestyle="--", alpha=0.6, label=f"$E = {E:.1f}$ J")
        # Zone accessible
        accessible = [ep <= E + 0.01 for ep in Ep_vals]
        for i in range(len(x)-1):
            if accessible[i]:
                ax.axvspan(x[i], x[i+1], alpha=0.05, color=c)

    # Équilibres
    analyse = analyser_potentiel(Ep, x_range, max(E_totals))
    for x_eq, typ in analyse["equilibres"]:
        marker = "v" if typ == "stable" else "^"
        color = "green" if typ == "stable" else "red"
        ax.plot(x_eq, Ep(x_eq), marker, color=color, markersize=12,
                label=f"éq. {typ} ({x_eq:.1f})")

    ax.set_xlabel("$x$"); ax.set_ylabel("Énergie (J)")
    ax.set_title(f"Diagramme de potentiel : ${nom}$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax


# ======================================================================
#  3. Applications
# ======================================================================

def pendule_conservation(m: float, L: float, theta0: float) -> dict:
    """Pendule : échange E_cin ↔ E_pot."""
    h_max = L * (1 - np.cos(theta0))
    v_max = np.sqrt(2 * G * h_max)
    E_mec = m * G * h_max
    return {"h_max": h_max, "v_max": v_max, "E_mec": E_mec}


def looping_hauteur_min(R: float) -> float:
    """
    Hauteur minimale pour passer un looping de rayon R.
    Au sommet : mg = mv²/R → v² = gR (minimum).
    Conservation : mgh = mg(2R) + ½m(gR) → h = 5R/2.
    """
    return 2.5 * R


def montagne_russe(
    profil_h: Callable, m: float, v0: float, mu: float = 0,
    x_range: tuple = (0, 20), h: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simule un chariot sur un profil h(x) avec frottement optionnel."""
    x_vals = np.arange(x_range[0], x_range[1], h)
    v = np.zeros_like(x_vals)
    E = np.zeros_like(x_vals)
    v[0] = v0
    h0 = profil_h(x_vals[0])

    for i in range(1, len(x_vals)):
        dh = profil_h(x_vals[i]) - profil_h(x_vals[i-1])
        dx = h
        # Conservation (avec frottement)
        v2 = v[i-1]**2 - 2*G*dh - 2*mu*G*dx
        v[i] = np.sqrt(max(v2, 0))
        E[i] = 0.5*m*v[i]**2 + m*G*profil_h(x_vals[i])

    return x_vals, v, E


# ======================================================================
#  4. Tracés
# ======================================================================

def tracer_pendule_energie(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    m, L = 1, 1
    theta0 = np.pi/3
    t = np.linspace(0, 4, 500)
    omega0 = np.sqrt(G/L)
    theta = theta0 * np.cos(omega0 * t)

    h = L * (1 - np.cos(theta))
    v = L * omega0 * theta0 * np.abs(np.sin(omega0 * t))

    Ecin = 0.5 * m * (L*omega0*theta0*np.sin(omega0*t))**2
    Epot = m * G * h
    Emec = Ecin + Epot

    ax.plot(t, Ecin, "r-", linewidth=2, label="$E_{cin}$")
    ax.plot(t, Epot, "b-", linewidth=2, label="$E_{pot}$")
    ax.plot(t, Emec, "k--", linewidth=1.5, label="$E_{mec}$ (constante)")

    ax.set_xlabel("$t$ (s)"); ax.set_ylabel("énergie (J)")
    ax.set_title(f"Pendule : échange $E_{{cin}} \\leftrightarrow E_{{pot}}$ ($\\theta_0 = {np.degrees(theta0):.0f}°$)")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_montagne_russe(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    profil = lambda x: 5 + 3*np.sin(x/2) - 1.5*np.cos(x)
    x, v, E = montagne_russe(profil, 1, 0, mu=0, x_range=(0, 20))
    x_f, v_f, E_f = montagne_russe(profil, 1, 0, mu=0.05, x_range=(0, 20))

    ax.plot(x, [profil(xi) for xi in x], "k-", linewidth=2, label="profil $h(x)$")
    ax.plot(x, v, "b-", linewidth=1.5, alpha=0.7, label="$v(x)$ sans frott.")
    ax.plot(x_f, v_f, "r--", linewidth=1.5, alpha=0.7, label="$v(x)$ avec frott.")

    ax.set_xlabel("$x$ (m)"); ax.set_ylabel("$h$ (m) / $v$ (m/s)")
    ax.set_title("Montagne russe : conservation vs dissipation")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Conservation de l'énergie ===\n")
    m = 2  # kg
    for h in [0, 5, 10]:
        v = vitesse_depuis_hauteur(10, h)
        e = energie_mecanique(m, v, h)
        print(f"  h={h:>2}m : v={v:.2f} m/s, E_cin={e['E_cin']:.1f}, "
              f"E_pot={e['E_pot']:.1f}, E_mec={e['E_mec']:.1f} J")

    print(f"\n=== Pendule (m=1, L=1, θ₀=60°) ===\n")
    r = pendule_conservation(1, 1, np.pi/3)
    print(f"  h_max = {r['h_max']:.4f} m")
    print(f"  v_max = {r['v_max']:.4f} m/s (en bas)")
    print(f"  E_mec = {r['E_mec']:.4f} J")

    print(f"\n=== Looping ===\n")
    for R in [1, 5, 10]:
        h_min = looping_hauteur_min(R)
        print(f"  R = {R}m : h_min = {h_min:.1f}m = 5R/2")

    print(f"\n=== Diagramme de potentiel ===\n")
    Ep = lambda x: x**4 - 8*x**2 + 12
    analyse = analyser_potentiel(Ep, (-3.5, 3.5), 20)
    for x_eq, typ in analyse["equilibres"]:
        print(f"  x = {x_eq:.2f} : équilibre {typ}, Ep = {Ep(x_eq):.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_pendule_energie(ax=axes[0])
    tracer_diagramme_potentiel(
        Ep, (-3.5, 3.5), [5, 12, 16], "x^4 - 8x^2 + 12", ax=axes[1])
    tracer_montagne_russe(ax=axes[2])
    plt.tight_layout()
    plt.savefig("conservation_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
