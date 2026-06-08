"""
heat_equation.py
================

Équation de la chaleur : ∂u/∂t = α ∂²u/∂x².

Couvre :
    - Discrétisation par différences finies (schéma explicite)
    - Condition de stabilité : r = αΔt/Δx² ≤ 0.5
    - Conditions aux limites : Dirichlet, Neumann
    - Conditions initiales variées
    - Solution analytique (séparation de variables)
    - Visualisation spatio-temporelle

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Schéma explicite (FTCS)
# ======================================================================

def heat_explicite(
    u0: np.ndarray, alpha: float, dx: float, dt: float, n_steps: int,
    bc_left: float = 0, bc_right: float = 0,
) -> np.ndarray:
    """
    Forward Time, Central Space (FTCS) :
        u_j^{n+1} = u_j^n + r(u_{j-1}^n - 2u_j^n + u_{j+1}^n)
    avec r = αΔt/Δx². Stable ssi r ≤ 0.5.
    Renvoie u[temps, espace].
    """
    r = alpha * dt / dx**2
    nx = len(u0)
    u = np.zeros((n_steps + 1, nx))
    u[0] = u0.copy()

    for n in range(n_steps):
        u[n+1, 1:-1] = u[n, 1:-1] + r * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
        u[n+1, 0] = bc_left
        u[n+1, -1] = bc_right

    return u


# ======================================================================
#  2. Schéma implicite (BTCS)
# ======================================================================

def heat_implicite(
    u0: np.ndarray, alpha: float, dx: float, dt: float, n_steps: int,
    bc_left: float = 0, bc_right: float = 0,
) -> np.ndarray:
    """
    Backward Time, Central Space (BTCS) — inconditionnellement stable.
    Résout un système tridiagonal à chaque pas de temps.
    """
    r = alpha * dt / dx**2
    nx = len(u0)
    u = np.zeros((n_steps + 1, nx))
    u[0] = u0.copy()

    # Matrice tridiagonale (intérieur seulement)
    n_int = nx - 2
    A = np.zeros((n_int, n_int))
    for i in range(n_int):
        A[i, i] = 1 + 2*r
        if i > 0:
            A[i, i-1] = -r
        if i < n_int - 1:
            A[i, i+1] = -r

    for n in range(n_steps):
        b = u[n, 1:-1].copy()
        b[0] += r * bc_left
        b[-1] += r * bc_right
        u[n+1, 1:-1] = np.linalg.solve(A, b)
        u[n+1, 0] = bc_left
        u[n+1, -1] = bc_right

    return u


# ======================================================================
#  3. Solution analytique
# ======================================================================

def heat_analytique(
    x: np.ndarray, t: float, alpha: float, L: float, n_modes: int = 50,
) -> np.ndarray:
    """
    Solution pour u(x,0) = sin(πx/L), u(0,t) = u(L,t) = 0 :
        u(x,t) = sin(πx/L) · exp(-α(π/L)²t).
    Cas général (série de Fourier) :
        u = Σ bₙ sin(nπx/L) exp(-α(nπ/L)²t).
    """
    return np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi / L)**2 * t)


def heat_analytique_carre(
    x: np.ndarray, t: float, alpha: float, L: float, n_modes: int = 50,
) -> np.ndarray:
    """Solution pour condition initiale carrée : u₀ = 1 sur [L/4, 3L/4]."""
    u = np.zeros_like(x)
    for n in range(1, n_modes + 1):
        bn = (2/L) * (L/(n*np.pi)) * (np.cos(n*np.pi/4) - np.cos(3*n*np.pi/4))
        u += bn * np.sin(n*np.pi*x/L) * np.exp(-alpha*(n*np.pi/L)**2 * t)
    return u


# ======================================================================
#  4. Tracés
# ======================================================================

def tracer_evolution(u: np.ndarray, x: np.ndarray, dt: float,
                      indices_t: list[int] | None = None,
                      titre: str = "", ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if indices_t is None:
        n_total = u.shape[0]
        indices_t = [0, n_total//10, n_total//4, n_total//2, n_total-1]

    for i in indices_t:
        t = i * dt
        ax.plot(x, u[i], linewidth=2, label=f"$t = {t:.3f}$")

    ax.set_xlabel("$x$"); ax.set_ylabel("$u(x, t)$")
    ax.set_title(titre if titre else "Équation de la chaleur")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_heatmap(u: np.ndarray, x: np.ndarray, dt: float,
                     titre: str = "", ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    t = np.arange(u.shape[0]) * dt
    im = ax.imshow(u.T, aspect="auto", origin="lower", cmap="hot",
                    extent=[t[0], t[-1], x[0], x[-1]])
    plt.colorbar(im, ax=ax, label="$u(x,t)$")
    ax.set_xlabel("$t$"); ax.set_ylabel("$x$")
    ax.set_title(titre if titre else "Heatmap $u(x,t)$")
    return ax


def tracer_stabilite(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre instabilité quand r > 0.5."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    L, nx = 1.0, 50
    x = np.linspace(0, L, nx)
    u0 = np.sin(np.pi * x / L)
    alpha = 0.01

    for r_target, style in [(0.4, "-"), (0.5, "--"), (0.6, ":")]:
        dx = L / (nx - 1)
        dt = r_target * dx**2 / alpha
        n_steps = int(0.5 / dt)
        u = heat_explicite(u0, alpha, dx, dt, min(n_steps, 5000))
        label = f"$r = {r_target}$ ({'stable' if r_target <= 0.5 else 'INSTABLE'})"
        ax.plot(x, u[min(n_steps, 5000)//2], style, linewidth=2, label=label)

    ax.set_xlabel("$x$"); ax.set_ylabel("$u$")
    ax.set_title("Stabilité : $r = \\alpha\\Delta t / \\Delta x^2 \\leq 0.5$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    L, nx = 1.0, 51
    x = np.linspace(0, L, nx)
    dx = L / (nx - 1)
    alpha = 0.01

    print("=== Équation de la chaleur ===\n")
    print(f"  Domaine : [0, {L}], nx = {nx}, α = {alpha}")

    # Condition de stabilité
    r = 0.4
    dt = r * dx**2 / alpha
    print(f"  r = {r}, dt = {dt:.6f}")
    print(f"  Condition r ≤ 0.5 : {'✓' if r <= 0.5 else '✗'}")

    # Explicite
    u0 = np.sin(np.pi * x / L)
    n_steps = 2000
    u_exp = heat_explicite(u0, alpha, dx, dt, n_steps)

    # Vérification vs analytique
    t_final = n_steps * dt
    u_exact = heat_analytique(x, t_final, alpha, L)
    erreur = np.max(np.abs(u_exp[-1] - u_exact))
    print(f"\n  Explicite (t = {t_final:.3f}) :")
    print(f"    ||u_num - u_exact||_∞ = {erreur:.2e}")

    # Implicite
    r_imp = 2.0  # r >> 0.5, instable pour explicite mais OK pour implicite
    dt_imp = r_imp * dx**2 / alpha
    n_steps_imp = int(t_final / dt_imp) + 1
    u_imp = heat_implicite(u0, alpha, dx, dt_imp, n_steps_imp)
    u_exact_imp = heat_analytique(x, n_steps_imp * dt_imp, alpha, L)
    err_imp = np.max(np.abs(u_imp[-1] - u_exact_imp))
    print(f"\n  Implicite (r = {r_imp}, inconditionnellement stable) :")
    print(f"    ||u_num - u_exact||_∞ = {err_imp:.2e}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tracer_evolution(u_exp, x, dt, titre="Explicite (sin initial)", ax=axes[0, 0])
    tracer_heatmap(u_exp, x, dt, titre="Heatmap", ax=axes[0, 1])

    # Condition initiale carrée
    u0_sq = np.where((x > 0.25) & (x < 0.75), 1.0, 0.0)
    u_sq = heat_explicite(u0_sq, alpha, dx, dt, n_steps)
    tracer_evolution(u_sq, x, dt, titre="Diffusion d'un créneau", ax=axes[1, 0])
    tracer_stabilite(ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig("heat_equation_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
