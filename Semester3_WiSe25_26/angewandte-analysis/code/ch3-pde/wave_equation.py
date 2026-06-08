"""
wave_equation.py
================

Équation des ondes 1D : ∂²u/∂t² = c² ∂²u/∂x².

Couvre :
    - Discrétisation explicite (schéma centré en temps et espace)
    - Condition CFL : c·Δt/Δx ≤ 1
    - Solution de d'Alembert : u = f(x-ct) + g(x+ct)
    - Réflexion aux bords (Dirichlet = fixe, Neumann = libre)
    - Ondes stationnaires vs progressives

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def wave_explicite(
    u0: np.ndarray, v0: np.ndarray, c: float, dx: float, dt: float,
    n_steps: int, bc: str = "dirichlet",
) -> np.ndarray:
    """
    Schéma centré :
        u_j^{n+1} = 2u_j^n - u_j^{n-1} + r²(u_{j+1}^n - 2u_j^n + u_{j-1}^n)
    avec r = cΔt/Δx. Stable ssi r ≤ 1 (CFL).
    """
    r = c * dt / dx
    nx = len(u0)
    u = np.zeros((n_steps + 1, nx))
    u[0] = u0.copy()

    # Premier pas : u^1 = u^0 + dt·v0 + 0.5·r²·(u_{j+1} - 2u_j + u_{j-1})
    u[1, 1:-1] = (u0[1:-1] + dt * v0[1:-1] +
                   0.5 * r**2 * (u0[2:] - 2*u0[1:-1] + u0[:-2]))
    _apply_bc(u, 1, bc)

    for n in range(1, n_steps):
        u[n+1, 1:-1] = (2*u[n, 1:-1] - u[n-1, 1:-1] +
                          r**2 * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2]))
        _apply_bc(u, n+1, bc)

    return u


def _apply_bc(u: np.ndarray, n: int, bc: str) -> None:
    if bc == "dirichlet":
        u[n, 0] = 0; u[n, -1] = 0
    elif bc == "neumann":
        u[n, 0] = u[n, 1]; u[n, -1] = u[n, -2]


def dalembert(x: np.ndarray, t: float, c: float,
               f: callable, g_integral: callable = None) -> np.ndarray:
    """
    Solution de d'Alembert pour corde infinie :
        u(x,t) = [f(x-ct) + f(x+ct)] / 2 + 1/(2c) ∫_{x-ct}^{x+ct} g(s) ds.
    Si v₀ = 0 : u = [f(x-ct) + f(x+ct)] / 2.
    """
    return 0.5 * (f(x - c*t) + f(x + c*t))


def condition_cfl(c: float, dx: float, dt: float) -> dict:
    """Vérifie la condition CFL : r = cΔt/Δx ≤ 1."""
    r = c * dt / dx
    return {"r": r, "stable": r <= 1, "dt_max": dx / c}


# ======================================================================
#  Tracés
# ======================================================================

def tracer_propagation(u: np.ndarray, x: np.ndarray, dt: float,
                        indices_t: list[int] | None = None,
                        titre: str = "", ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if indices_t is None:
        nt = u.shape[0]
        indices_t = [0, nt//8, nt//4, nt//2, 3*nt//4, nt-1]

    for i in indices_t:
        ax.plot(x, u[i], linewidth=1.5, label=f"$t = {i*dt:.3f}$")

    ax.set_xlabel("$x$"); ax.set_ylabel("$u(x,t)$")
    ax.set_title(titre if titre else "Équation des ondes")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax


def tracer_xt_diagram(u: np.ndarray, x: np.ndarray, dt: float,
                        titre: str = "", ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    t = np.arange(u.shape[0]) * dt
    im = ax.imshow(u.T, aspect="auto", origin="lower", cmap="RdBu_r",
                    extent=[t[0], t[-1], x[0], x[-1]],
                    vmin=-np.max(np.abs(u)), vmax=np.max(np.abs(u)))
    plt.colorbar(im, ax=ax, label="$u$")
    ax.set_xlabel("$t$"); ax.set_ylabel("$x$")
    ax.set_title(titre if titre else "Diagramme $(x, t)$")
    return ax


def tracer_cfl(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre l'effet de la condition CFL."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    L, nx, c = 1.0, 100, 1.0
    x = np.linspace(0, L, nx)
    dx = L / (nx - 1)
    u0 = np.sin(2*np.pi*x/L)
    v0 = np.zeros(nx)

    for r_target in [0.5, 1.0, 1.01]:
        dt = r_target * dx / c
        n_steps = min(int(1.0 / dt), 10000)
        u = wave_explicite(u0, v0, c, dx, dt, n_steps)
        y = u[n_steps//2]
        y_clip = np.clip(y, -3, 3)
        label = f"$r = {r_target}$ ({'stable' if r_target <= 1 else 'INSTABLE'})"
        ax.plot(x, y_clip, linewidth=2, label=label)

    ax.set_xlabel("$x$"); ax.set_ylabel("$u$")
    ax.set_title("Condition CFL : $r = c\\Delta t / \\Delta x \\leq 1$")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    L, nx, c = 1.0, 201, 1.0
    x = np.linspace(0, L, nx)
    dx = L / (nx - 1)

    print("=== Équation des ondes 1D ===\n")
    r = 0.8
    dt = r * dx / c
    cfl = condition_cfl(c, dx, dt)
    print(f"  c = {c}, dx = {dx:.4f}, dt = {dt:.6f}")
    print(f"  CFL : r = {cfl['r']:.2f}, stable = {cfl['stable']} ✓")

    # Gaussienne initiale
    u0 = np.exp(-200*(x - 0.5)**2)
    v0 = np.zeros(nx)
    n_steps = 1000

    u_dir = wave_explicite(u0, v0, c, dx, dt, n_steps, "dirichlet")
    u_neu = wave_explicite(u0, v0, c, dx, dt, n_steps, "neumann")

    print(f"\n  Énergie conservée (Dirichlet) :")
    for k in [0, n_steps//4, n_steps//2, n_steps]:
        E = 0.5 * dx * np.sum(u_dir[k]**2)
        print(f"    t = {k*dt:.3f} : E ∝ {E:.6f}")

    # d'Alembert
    print(f"\n=== d'Alembert ===\n")
    f_init = lambda x: np.exp(-200*(x - 0.5)**2)
    t_test = 0.2
    u_da = dalembert(x, t_test, c, f_init)
    print(f"  u(x, {t_test}) = [f(x-ct) + f(x+ct)] / 2")
    print(f"  → Deux bosses se propageant en sens opposé")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tracer_propagation(u_dir, x, dt, titre="Dirichlet (bords fixes)", ax=axes[0, 0])
    tracer_propagation(u_neu, x, dt, titre="Neumann (bords libres)", ax=axes[0, 1])
    tracer_xt_diagram(u_dir, x, dt, "Diagramme (x,t) — Dirichlet", ax=axes[1, 0])
    tracer_cfl(ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig("wave_equation_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
