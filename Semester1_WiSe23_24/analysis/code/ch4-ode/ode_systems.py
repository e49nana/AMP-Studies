"""
ode_systems.py
==============

Systèmes d'EDO et portraits de phase.

Couvre :
    - Systèmes y' = f(t, y) avec y ∈ R^n
    - RK4 vectoriel
    - Portraits de phase pour systèmes autonomes 2D
    - Classification des points fixes : nœud, selle, spirale, centre
    - Exemples : oscillateur harmonique, Lotka-Volterra, Van der Pol

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def rk4_systeme(
    f: Callable[[float, np.ndarray], np.ndarray],
    t0: float, y0: np.ndarray, t_end: float, h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """RK4 pour systèmes y' = f(t, y), y ∈ R^n."""
    n_steps = int((t_end - t0) / h)
    y0 = np.asarray(y0, dtype=float)
    dim = len(y0)
    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, dim))
    t[0] = t0
    y[0] = y0

    for k in range(n_steps):
        k1 = f(t[k], y[k])
        k2 = f(t[k] + h/2, y[k] + h/2 * k1)
        k3 = f(t[k] + h/2, y[k] + h/2 * k2)
        k4 = f(t[k] + h, y[k] + h * k3)
        y[k+1] = y[k] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t[k+1] = t[k] + h

    return t, y


# ======================================================================
#  Systèmes classiques
# ======================================================================

def oscillateur_harmonique(omega: float = 1.0):
    """y'' + ω²y = 0 → système : y₁' = y₂, y₂' = -ω²y₁."""
    return lambda t, y: np.array([y[1], -omega**2 * y[0]])


def oscillateur_amorti(omega: float = 1.0, gamma: float = 0.2):
    """y'' + 2γy' + ω²y = 0."""
    return lambda t, y: np.array([y[1], -2*gamma*y[1] - omega**2*y[0]])


def lotka_volterra(alpha=1.0, beta=0.5, gamma=0.5, delta=0.2):
    """Proie-prédateur : x' = αx - βxy, y' = δxy - γy."""
    return lambda t, y: np.array([
        alpha*y[0] - beta*y[0]*y[1],
        delta*y[0]*y[1] - gamma*y[1],
    ])


def van_der_pol(mu: float = 1.0):
    """Oscillateur de Van der Pol : y'' - μ(1-y²)y' + y = 0."""
    return lambda t, y: np.array([y[1], mu*(1 - y[0]**2)*y[1] - y[0]])


# ======================================================================
#  Classification des points fixes
# ======================================================================

def classifier_point_fixe(A: np.ndarray) -> str:
    """Classifie le point fixe du système linéaire y' = Ay."""
    eigvals = np.linalg.eigvals(A)
    re = eigvals.real
    im = eigvals.imag

    if np.all(np.abs(im) < 1e-10):  # réelles
        if np.all(re < 0):
            return "nœud stable" if re[0] != re[1] else "nœud dégénéré stable"
        elif np.all(re > 0):
            return "nœud instable"
        elif re[0] * re[1] < 0:
            return "selle (instable)"
    else:  # complexes
        if np.all(re < -1e-10):
            return "spirale stable"
        elif np.all(re > 1e-10):
            return "spirale instable"
        else:
            return "centre"

    return "indéterminé"


# ======================================================================
#  Portraits de phase
# ======================================================================

def portrait_de_phase(
    f: Callable, y0s: list[np.ndarray], t_end: float = 20, h: float = 0.01,
    ax: plt.Axes | None = None, titre: str = "",
) -> plt.Axes:
    """Trace le portrait de phase 2D pour plusieurs conditions initiales."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    for y0 in y0s:
        t, y = rk4_systeme(f, 0, y0, t_end, h)
        ax.plot(y[:, 0], y[:, 1], linewidth=1, alpha=0.7)
        ax.plot(y0[0], y0[1], "ko", markersize=3)

    # Champ de vecteurs
    y1_range = ax.get_xlim() if ax.get_xlim() != (0, 1) else (-3, 3)
    y2_range = ax.get_ylim() if ax.get_ylim() != (0, 1) else (-3, 3)
    Y1, Y2 = np.meshgrid(np.linspace(-3, 3, 15), np.linspace(-3, 3, 15))
    U, V = np.zeros_like(Y1), np.zeros_like(Y2)
    for i in range(Y1.shape[0]):
        for j in range(Y1.shape[1]):
            dy = f(0, np.array([Y1[i,j], Y2[i,j]]))
            U[i,j], V[i,j] = dy[0], dy[1]
    ax.quiver(Y1, Y2, U, V, alpha=0.2, color="grey")

    ax.set_xlabel("$y_1$"); ax.set_ylabel("$y_2$")
    ax.set_title(titre)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Oscillateur harmonique ===")
    f = oscillateur_harmonique(1.0)
    t, y = rk4_systeme(f, 0, np.array([1.0, 0.0]), 20, 0.01)
    print(f"  y(0) = [1, 0], période = 2π ≈ {2*np.pi:.4f}")
    print(f"  y(2π) = [{y[int(2*np.pi/0.01), 0]:.6f}, {y[int(2*np.pi/0.01), 1]:.6f}]")
    print(f"  → Retour quasi-exact à l'état initial ✓")

    print(f"\n=== Classification des points fixes ===")
    matrices = [
        ("Nœud stable", np.array([[-2, 0], [0, -1]])),
        ("Selle", np.array([[1, 0], [0, -1]])),
        ("Spirale stable", np.array([[-0.1, 1], [-1, -0.1]])),
        ("Centre", np.array([[0, 1], [-1, 0]])),
    ]
    for nom, A in matrices:
        eigvals = np.linalg.eigvals(A)
        print(f"  {nom:20s} : λ = {np.round(eigvals, 3)}, type = {classifier_point_fixe(A)}")

    print(f"\n=== Lotka-Volterra (proie-prédateur) ===")
    f_lv = lotka_volterra()
    t, y = rk4_systeme(f_lv, 0, np.array([2.0, 1.0]), 30, 0.01)
    print(f"  Proies oscillent entre {y[:,0].min():.2f} et {y[:,0].max():.2f}")
    print(f"  Prédateurs oscillent entre {y[:,1].min():.2f} et {y[:,1].max():.2f}")

    # Tracés
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Oscillateur harmonique
    y0s_osc = [np.array([r, 0]) for r in [0.5, 1, 1.5, 2]]
    portrait_de_phase(oscillateur_harmonique(), y0s_osc, 10, 0.01,
                       ax=axes[0, 0], titre="Oscillateur harmonique (centre)")

    # Amorti
    y0s_am = [np.array([r, 0]) for r in [1, 2, 3]]
    portrait_de_phase(oscillateur_amorti(1, 0.2), y0s_am, 30, 0.01,
                       ax=axes[0, 1], titre="Oscillateur amorti (spirale stable)")

    # Lotka-Volterra
    y0s_lv = [np.array([2, 1]), np.array([1, 2]), np.array([3, 0.5])]
    portrait_de_phase(lotka_volterra(), y0s_lv, 30, 0.01,
                       ax=axes[1, 0], titre="Lotka-Volterra")
    axes[1, 0].set_xlim(0, 6); axes[1, 0].set_ylim(0, 4)

    # Van der Pol
    y0s_vdp = [np.array([0.1, 0]), np.array([3, 0]), np.array([-2, 2])]
    portrait_de_phase(van_der_pol(1.0), y0s_vdp, 30, 0.01,
                       ax=axes[1, 1], titre="Van der Pol (cycle limite)")

    plt.tight_layout()
    plt.savefig("ode_systems_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
