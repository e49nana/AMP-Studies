"""
laplace_equation.py
===================

Équation de Laplace/Poisson 2D : Δu = f.

Couvre :
    - Laplace (Δu = 0) : problème stationnaire, pas de source
    - Poisson (Δu = f) : avec terme source
    - Méthode de Jacobi itérative
    - Méthode de Gauss-Seidel (convergence plus rapide)
    - Conditions aux limites Dirichlet
    - Propriété de la valeur moyenne (solutions harmoniques)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Jacobi
# ======================================================================

def laplace_jacobi(
    nx: int, ny: int, bc: dict,
    f: np.ndarray | None = None,
    tol: float = 1e-6, n_max: int = 50000,
) -> tuple[np.ndarray, int]:
    """
    Résout Δu = f sur [0,1]² par itération de Jacobi.
    u_{i,j}^{new} = (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - h²f_{ij}) / 4.
    bc = {"top": val, "bottom": val, "left": val, "right": val} ou arrays.
    """
    h = 1.0 / (nx - 1)
    u = np.zeros((ny, nx))

    # Conditions aux limites
    _apply_bc(u, bc, nx, ny)

    if f is None:
        f = np.zeros((ny, nx))

    for k in range(1, n_max + 1):
        u_old = u.copy()
        u[1:-1, 1:-1] = 0.25 * (
            u_old[2:, 1:-1] + u_old[:-2, 1:-1] +
            u_old[1:-1, 2:] + u_old[1:-1, :-2] - h**2 * f[1:-1, 1:-1]
        )
        _apply_bc(u, bc, nx, ny)

        if np.max(np.abs(u - u_old)) < tol:
            break

    return u, k


# ======================================================================
#  2. Gauss-Seidel
# ======================================================================

def laplace_gauss_seidel(
    nx: int, ny: int, bc: dict,
    f: np.ndarray | None = None,
    tol: float = 1e-6, n_max: int = 50000,
) -> tuple[np.ndarray, int]:
    """Gauss-Seidel : utilise les valeurs déjà mises à jour (convergence ~2× plus rapide)."""
    h = 1.0 / (nx - 1)
    u = np.zeros((ny, nx))
    _apply_bc(u, bc, nx, ny)

    if f is None:
        f = np.zeros((ny, nx))

    for k in range(1, n_max + 1):
        max_diff = 0
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                u_new = 0.25 * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - h**2*f[i,j])
                max_diff = max(max_diff, abs(u_new - u[i,j]))
                u[i,j] = u_new

        if max_diff < tol:
            break

    return u, k


def _apply_bc(u: np.ndarray, bc: dict, nx: int, ny: int) -> None:
    """Applique les conditions aux limites Dirichlet."""
    if isinstance(bc.get("top"), (int, float)):
        u[-1, :] = bc["top"]
    elif bc.get("top") is not None:
        u[-1, :] = bc["top"]

    if isinstance(bc.get("bottom"), (int, float)):
        u[0, :] = bc["bottom"]
    elif bc.get("bottom") is not None:
        u[0, :] = bc["bottom"]

    if isinstance(bc.get("left"), (int, float)):
        u[:, 0] = bc["left"]
    elif bc.get("left") is not None:
        u[:, 0] = bc["left"]

    if isinstance(bc.get("right"), (int, float)):
        u[:, -1] = bc["right"]
    elif bc.get("right") is not None:
        u[:, -1] = bc["right"]


# ======================================================================
#  3. Propriété de la valeur moyenne
# ======================================================================

def verifier_valeur_moyenne(u: np.ndarray, i: int, j: int) -> dict:
    """Pour Δu = 0 : u(P) = moyenne des 4 voisins."""
    moyenne = 0.25 * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1])
    return {"u(P)": u[i,j], "moyenne_voisins": moyenne,
            "erreur": abs(u[i,j] - moyenne)}


# ======================================================================
#  4. Tracés
# ======================================================================

def tracer_solution(u: np.ndarray, titre: str = "",
                     ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(u, origin="lower", cmap="hot", extent=[0, 1, 0, 1])
    ax.contour(np.linspace(0, 1, u.shape[1]), np.linspace(0, 1, u.shape[0]),
               u, levels=15, colors="white", linewidths=0.5, alpha=0.5)
    plt.colorbar(im, ax=ax, label="$u(x, y)$")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(titre if titre else "Solution de Laplace/Poisson")
    return ax


def tracer_convergence_methodes(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    nx = ny = 30
    bc = {"top": 100, "bottom": 0, "left": 0, "right": 0}

    # Mesurer la convergence
    residus_j, residus_gs = [], []
    u_j = np.zeros((ny, nx)); _apply_bc(u_j, bc, nx, ny)
    u_gs = np.zeros((ny, nx)); _apply_bc(u_gs, bc, nx, ny)

    for k in range(500):
        u_j_old = u_j.copy()
        u_j[1:-1, 1:-1] = 0.25 * (u_j_old[2:,1:-1] + u_j_old[:-2,1:-1] +
                                     u_j_old[1:-1,2:] + u_j_old[1:-1,:-2])
        _apply_bc(u_j, bc, nx, ny)
        residus_j.append(np.max(np.abs(u_j - u_j_old)))

        max_diff = 0
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                old = u_gs[i,j]
                u_gs[i,j] = 0.25*(u_gs[i+1,j]+u_gs[i-1,j]+u_gs[i,j+1]+u_gs[i,j-1])
                max_diff = max(max_diff, abs(u_gs[i,j] - old))
        _apply_bc(u_gs, bc, nx, ny)
        residus_gs.append(max_diff)

    ax.semilogy(residus_j, "b-", linewidth=1.5, label="Jacobi")
    ax.semilogy(residus_gs, "r-", linewidth=1.5, label="Gauss-Seidel")
    ax.set_xlabel("itération"); ax.set_ylabel("max $|u^{new} - u^{old}|$")
    ax.set_title("Convergence : GS ~ 2× plus rapide que Jacobi")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Laplace Δu = 0 ===\n")
    nx = ny = 41
    bc = {"top": 100, "bottom": 0, "left": 0, "right": 0}

    u_j, k_j = laplace_jacobi(nx, ny, bc)
    u_gs, k_gs = laplace_gauss_seidel(nx, ny, bc, tol=1e-6)
    print(f"  Grille {nx}×{ny}, top=100, reste=0")
    print(f"  Jacobi       : {k_j} itérations")
    print(f"  Gauss-Seidel : {k_gs} itérations")
    print(f"  ||u_J - u_GS||_∞ = {np.max(np.abs(u_j - u_gs)):.2e}")

    print(f"\n=== Valeur moyenne ===\n")
    mid = nx // 2
    vm = verifier_valeur_moyenne(u_j, mid, mid)
    print(f"  u({mid},{mid}) = {vm['u(P)']:.4f}")
    print(f"  Moyenne voisins = {vm['moyenne_voisins']:.4f}")
    print(f"  Erreur = {vm['erreur']:.2e} ✓")

    print(f"\n=== Poisson Δu = f ===\n")
    f = np.zeros((ny, nx))
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    f = -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    bc_zero = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    u_p, k_p = laplace_jacobi(nx, ny, bc_zero, f)
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    print(f"  f = -2π² sin(πx)sin(πy) → u = sin(πx)sin(πy)")
    print(f"  ||u_num - u_exact||_∞ = {np.max(np.abs(u_p - u_exact)):.4e}")
    print(f"  {k_p} itérations")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    tracer_solution(u_j, "Laplace (top=100)", ax=axes[0, 0])
    tracer_solution(u_p, "Poisson ($u = \\sin\\pi x \\sin\\pi y$)", ax=axes[0, 1])
    tracer_convergence_methodes(ax=axes[1, 0])
    tracer_solution(np.abs(u_p - u_exact), "Erreur Poisson", ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig("laplace_equation_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
