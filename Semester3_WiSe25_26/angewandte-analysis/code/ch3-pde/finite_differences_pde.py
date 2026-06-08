"""
finite_differences_pde.py
=========================

Méthodes de différences finies pour les EDP — synthèse.

Couvre :
    - Classification : elliptique (Laplace), parabolique (chaleur), hyperbolique (ondes)
    - Stencils : 3 points (1D), 5 points (2D)
    - Analyse de stabilité : Von Neumann
    - Ordre de précision : O(Δx²), O(Δt)
    - Méthode de Crank-Nicolson (implicite O(Δt²))
    - Comparaison des schémas

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Classification des EDP
# ======================================================================

def classifier_edp(A: float, B: float, C: float) -> str:
    """
    EDP linéaire du 2e ordre : A·u_xx + B·u_xy + C·u_yy + ... = 0.
    Discriminant D = B² - 4AC.
    """
    D = B**2 - 4*A*C
    if D < -1e-10:
        return "elliptique (Laplace, Poisson)"
    elif abs(D) < 1e-10:
        return "parabolique (chaleur, diffusion)"
    else:
        return "hyperbolique (ondes, advection)"


# ======================================================================
#  2. Stencils
# ======================================================================

def stencil_1d_second(u: np.ndarray, j: int, h: float) -> float:
    """u''(xⱼ) ≈ (u_{j-1} - 2u_j + u_{j+1}) / h². Erreur O(h²)."""
    return (u[j-1] - 2*u[j] + u[j+1]) / h**2


def stencil_2d_laplacien(u: np.ndarray, i: int, j: int, h: float) -> float:
    """Δu ≈ (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / h². 5 points."""
    return (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / h**2


def ordre_precision(errors: list[float], hs: list[float]) -> float:
    """Estime l'ordre p tel que err ∝ h^p par régression log-log."""
    log_h = np.log(hs)
    log_e = np.log(errors)
    p, _ = np.polyfit(log_h, log_e, 1)
    return p


# ======================================================================
#  3. Crank-Nicolson
# ======================================================================

def crank_nicolson_heat(
    u0: np.ndarray, alpha: float, dx: float, dt: float, n_steps: int,
    bc_left: float = 0, bc_right: float = 0,
) -> np.ndarray:
    """
    Crank-Nicolson pour l'eq. de la chaleur :
        (I - r/2 · D²) u^{n+1} = (I + r/2 · D²) u^n.
    Implicite, O(Δt², Δx²), inconditionnellement stable.
    """
    r = alpha * dt / dx**2
    nx = len(u0)
    u = np.zeros((n_steps + 1, nx))
    u[0] = u0.copy()
    n_int = nx - 2

    # Matrices (intérieur seulement)
    A = np.zeros((n_int, n_int))
    B = np.zeros((n_int, n_int))
    for i in range(n_int):
        A[i, i] = 1 + r
        B[i, i] = 1 - r
        if i > 0:
            A[i, i-1] = -r/2
            B[i, i-1] = r/2
        if i < n_int - 1:
            A[i, i+1] = -r/2
            B[i, i+1] = r/2

    for n in range(n_steps):
        rhs = B @ u[n, 1:-1]
        rhs[0] += r/2 * (bc_left + bc_left)  # terme BC
        rhs[-1] += r/2 * (bc_right + bc_right)
        u[n+1, 1:-1] = np.linalg.solve(A, rhs)
        u[n+1, 0] = bc_left
        u[n+1, -1] = bc_right

    return u


# ======================================================================
#  4. Analyse de Von Neumann
# ======================================================================

def facteur_amplification_ftcs(r: float, theta: np.ndarray) -> np.ndarray:
    """
    Facteur d'amplification du schéma FTCS (chaleur) :
        g(θ) = 1 - 4r sin²(θ/2).
    Stable ssi |g| ≤ 1 pour tout θ → r ≤ 0.5.
    """
    return 1 - 4*r*np.sin(theta/2)**2


def facteur_amplification_cn(r: float, theta: np.ndarray) -> np.ndarray:
    """
    Crank-Nicolson :
        g(θ) = (1 - 2r sin²(θ/2)) / (1 + 2r sin²(θ/2)).
    |g| ≤ 1 pour tout r → inconditionnellement stable.
    """
    s = 2*r*np.sin(theta/2)**2
    return (1 - s) / (1 + s)


# ======================================================================
#  5. Comparaison des schémas
# ======================================================================

def comparer_schemas(ax: plt.Axes | None = None) -> plt.Axes:
    """Compare FTCS, BTCS, Crank-Nicolson sur l'eq. de la chaleur."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    L, nx, alpha = 1.0, 51, 0.01
    x = np.linspace(0, L, nx)
    dx = L / (nx - 1)
    u0 = np.sin(np.pi * x / L)
    t_final = 1.0

    # Solution exacte
    u_exact = lambda t: np.sin(np.pi*x/L) * np.exp(-alpha*(np.pi/L)**2 * t)

    hs_dt = np.logspace(-4, -1, 15)
    err_ftcs, err_cn = [], []

    for dt in hs_dt:
        r = alpha * dt / dx**2
        n_steps = int(t_final / dt)
        if n_steps < 1:
            continue

        # FTCS (seulement si stable)
        if r <= 0.5:
            from heat_equation import heat_explicite
            u_f = heat_explicite(u0, alpha, dx, dt, n_steps)
            err_ftcs.append((dt, np.max(np.abs(u_f[-1] - u_exact(n_steps*dt)))))
        
        # Crank-Nicolson
        u_cn = crank_nicolson_heat(u0, alpha, dx, dt, n_steps)
        err_cn.append((dt, np.max(np.abs(u_cn[-1] - u_exact(n_steps*dt)))))

    if err_ftcs:
        dts, errs = zip(*err_ftcs)
        ax.loglog(dts, errs, "rs-", markersize=5, label="FTCS $O(\\Delta t)$")
    dts, errs = zip(*err_cn)
    ax.loglog(dts, errs, "bo-", markersize=5, label="Crank-Nicolson $O(\\Delta t^2)$")

    ax.set_xlabel("$\\Delta t$"); ax.set_ylabel("erreur max")
    ax.set_title("Comparaison des schémas (eq. chaleur)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    return ax


# ======================================================================
#  6. Tracés
# ======================================================================

def tracer_classification(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    edps = [
        ("$u_{xx} + u_{yy} = 0$ (Laplace)", 1, 0, 1),
        ("$u_t = u_{xx}$ (chaleur)", 1, 0, 0),
        ("$u_{tt} = u_{xx}$ (ondes)", 1, 0, -1),
        ("$u_{xx} + u_{xy} + u_{yy} = 0$", 1, 1, 1),
        ("$u_{xx} - u_{yy} = 0$", 1, 0, -1),
    ]

    types_colors = {"elliptique": "blue", "parabolique": "green", "hyperbolique": "red"}

    for i, (nom, A, B, C) in enumerate(edps):
        typ = classifier_edp(A, B, C).split(" ")[0]
        color = types_colors.get(typ, "grey")
        ax.barh(i, 1, color=color, alpha=0.6)
        ax.text(0.05, i, f"{nom} → {typ}", va="center", fontsize=10)

    ax.set_yticks([]); ax.set_xticks([])
    ax.set_title("Classification des EDP du 2e ordre ($D = B^2 - 4AC$)")
    # Légende manuelle
    for typ, c in types_colors.items():
        ax.barh([], [], color=c, alpha=0.6, label=typ)
    ax.legend(loc="lower right")
    return ax


def tracer_von_neumann(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    theta = np.linspace(0, np.pi, 200)
    for r in [0.1, 0.25, 0.5, 0.6]:
        g = facteur_amplification_ftcs(r, theta)
        stable = np.all(np.abs(g) <= 1 + 1e-10)
        ax.plot(theta, g, linewidth=2,
                label=f"FTCS $r={r}$ ({'✓' if stable else '✗'})")

    g_cn = facteur_amplification_cn(1.0, theta)
    ax.plot(theta, g_cn, "k--", linewidth=2, label="CN $r=1$ (toujours stable)")

    ax.axhline(1, color="grey", linestyle=":", alpha=0.3)
    ax.axhline(-1, color="grey", linestyle=":", alpha=0.3)
    ax.set_xlabel("$\\theta$ (nombre d'onde)"); ax.set_ylabel("$g(\\theta)$")
    ax.set_title("Analyse de Von Neumann : facteur d'amplification")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Classification des EDP ===\n")
    for nom, A, B, C in [("Laplace", 1, 0, 1), ("Chaleur", 1, 0, 0),
                           ("Ondes", 1, 0, -1), ("Advection", 0, 0, 0)]:
        print(f"  {nom:10s} : {classifier_edp(A, B, C)}")

    print(f"\n=== Crank-Nicolson ===\n")
    L, nx, alpha = 1.0, 51, 0.01
    x = np.linspace(0, L, nx)
    dx = L / (nx - 1)
    u0 = np.sin(np.pi * x / L)

    for dt in [0.01, 0.1, 0.5]:
        r = alpha * dt / dx**2
        n_steps = int(1.0 / dt)
        u_cn = crank_nicolson_heat(u0, alpha, dx, dt, n_steps)
        u_exact = np.sin(np.pi*x/L) * np.exp(-alpha*(np.pi/L)**2 * n_steps*dt)
        err = np.max(np.abs(u_cn[-1] - u_exact))
        print(f"  dt = {dt:<4}, r = {r:>6.2f} : err = {err:.2e} (CN stable même si r >> 0.5)")

    print(f"\n=== Ordre de précision ===\n")
    print(f"  FTCS : O(Δt) en temps, O(Δx²) en espace")
    print(f"  Crank-Nicolson : O(Δt²) en temps, O(Δx²) en espace")
    print(f"  → CN est plus précis pour le même coût")

    print(f"\n=== Von Neumann (FTCS) ===\n")
    for r in [0.1, 0.25, 0.5, 0.6]:
        g_max = np.max(np.abs(facteur_amplification_ftcs(r, np.linspace(0, np.pi, 100))))
        print(f"  r = {r} : max|g| = {g_max:.4f} → {'stable ✓' if g_max <= 1+1e-10 else 'INSTABLE ✗'}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_classification(ax=axes[0])
    tracer_von_neumann(ax=axes[1])
    plt.tight_layout()
    plt.savefig("finite_differences_pde_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
