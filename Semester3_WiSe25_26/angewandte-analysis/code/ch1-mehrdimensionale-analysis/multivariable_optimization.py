"""
multivariable_optimization.py
=============================

Optimisation en plusieurs variables.

Couvre :
    - Gradient descent en nD
    - Méthode de Newton multivariable
    - Multiplicateurs de Lagrange (contraintes)
    - Conditions nécessaires et suffisantes (Hessienne)
    - Visualisation des trajectoires d'optimisation

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def gradient_num(f: Callable, x: np.ndarray, h: float = 1e-7) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def hessienne_num(f: Callable, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i] = h; ej[j] = h
            H[i,j] = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4*h*h)
    return H


@dataclass
class OptResult:
    x: np.ndarray
    fx: float
    iterations: int
    methode: str
    trajectoire: list[np.ndarray] = field(default_factory=list)


# ======================================================================
#  1. Gradient Descent
# ======================================================================

def gradient_descent(
    f: Callable, x0: np.ndarray, alpha: float = 0.01,
    tol: float = 1e-10, n_max: int = 10000,
) -> OptResult:
    """x⁺ = x - α∇f(x). Convergence linéaire."""
    x = np.asarray(x0, dtype=float)
    traj = [x.copy()]
    for k in range(1, n_max + 1):
        g = gradient_num(f, x)
        x = x - alpha * g
        traj.append(x.copy())
        if np.linalg.norm(g) < tol:
            break
    return OptResult(x, f(x), k, f"GD (α={alpha})", traj)


def gradient_descent_backtracking(
    f: Callable, x0: np.ndarray, tol: float = 1e-10, n_max: int = 5000,
) -> OptResult:
    """GD avec line search (Armijo backtracking)."""
    x = np.asarray(x0, dtype=float)
    traj = [x.copy()]
    for k in range(1, n_max + 1):
        g = gradient_num(f, x)
        if np.linalg.norm(g) < tol:
            break
        # Backtracking
        alpha = 1.0
        while f(x - alpha * g) > f(x) - 0.5 * alpha * np.dot(g, g):
            alpha *= 0.5
            if alpha < 1e-15:
                break
        x = x - alpha * g
        traj.append(x.copy())
    return OptResult(x, f(x), k, "GD (backtracking)", traj)


# ======================================================================
#  2. Newton multivariable
# ======================================================================

def newton_multivariable(
    f: Callable, x0: np.ndarray, tol: float = 1e-12, n_max: int = 100,
) -> OptResult:
    """x⁺ = x - H⁻¹ ∇f. Convergence quadratique près du minimum."""
    x = np.asarray(x0, dtype=float)
    traj = [x.copy()]
    for k in range(1, n_max + 1):
        g = gradient_num(f, x)
        H = hessienne_num(f, x)
        if np.linalg.norm(g) < tol:
            break
        try:
            dx = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
        x = x + dx
        traj.append(x.copy())
    return OptResult(x, f(x), k, "Newton", traj)


# ======================================================================
#  3. Lagrange
# ======================================================================

def lagrange_2d(
    f: Callable, g: Callable, x0: np.ndarray, lam0: float = 0,
    tol: float = 1e-10, n_max: int = 200,
) -> dict:
    """
    min f(x,y) sous g(x,y) = 0.
    Résout ∇f = λ∇g, g = 0 par Newton sur le système augmenté.
    """
    def F(z):
        x, y, lam = z
        gf = gradient_num(lambda v: f(v), np.array([x, y]))
        gg = gradient_num(lambda v: g(v), np.array([x, y]))
        return np.array([gf[0] - lam*gg[0], gf[1] - lam*gg[1], g(np.array([x, y]))])

    z = np.array([x0[0], x0[1], lam0], dtype=float)
    for k in range(n_max):
        Fz = F(z)
        if np.linalg.norm(Fz) < tol:
            break
        J = np.zeros((3, 3))
        for j in range(3):
            e = np.zeros(3); e[j] = 1e-6
            J[:, j] = (F(z + e) - F(z - e)) / 2e-6
        try:
            dz = np.linalg.solve(J, -Fz)
        except np.linalg.LinAlgError:
            break
        z = z + dz

    return {
        "x": z[:2], "lambda": z[2],
        "f(x)": f(z[:2]), "g(x)": g(z[:2]),
        "iterations": k,
    }


# ======================================================================
#  4. Tracés
# ======================================================================

def tracer_trajectoire_2d(
    f: Callable, results: list[OptResult],
    x_range: tuple, y_range: tuple, nom: str = "f",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(lambda xi, yi: f(np.array([xi, yi])))(X, Y)

    ax.contour(X, Y, Z, levels=30, cmap="viridis", alpha=0.5)

    colors = ["red", "blue", "green", "orange"]
    for r, c in zip(results, colors):
        traj = np.array(r.trajectoire)
        ax.plot(traj[:, 0], traj[:, 1], "o-", color=c, markersize=3,
                linewidth=1.5, label=f"{r.methode} ({r.iterations} it.)")
        ax.plot(traj[-1, 0], traj[-1, 1], "*", color=c, markersize=15)

    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(f"Optimisation de ${nom}$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax


def tracer_lagrange(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    f = lambda v: v[0]**2 + v[1]**2
    g = lambda v: v[0] + v[1] - 1

    x = np.linspace(-0.5, 2, 100)
    y = np.linspace(-0.5, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    ax.contour(X, Y, Z, levels=15, cmap="viridis", alpha=0.5)
    ax.plot(x, 1 - x, "r-", linewidth=2, label="$g : x + y = 1$")

    r = lagrange_2d(f, g, np.array([1.0, 0.0]))
    ax.plot(r["x"][0], r["x"][1], "r*", markersize=15,
            label=f"min = ({r['x'][0]:.2f}, {r['x'][1]:.2f}), f = {r['f(x)']:.3f}")

    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("Lagrange : min $x^2+y^2$ sous $x+y=1$")
    ax.legend(); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    # Rosenbrock : f(x,y) = (1-x)² + 100(y-x²)²
    rosenbrock = lambda v: (1-v[0])**2 + 100*(v[1]-v[0]**2)**2
    x0 = np.array([-1.0, 1.0])

    print("=== Rosenbrock : min en (1, 1) ===\n")
    r_gd = gradient_descent(rosenbrock, x0, alpha=0.001, n_max=50000)
    r_bt = gradient_descent_backtracking(rosenbrock, x0)
    r_nw = newton_multivariable(rosenbrock, x0)

    for r in [r_gd, r_bt, r_nw]:
        print(f"  {r.methode:25s} : x* = {np.round(r.x, 6)}, f = {r.fx:.2e}, {r.iterations} it.")

    print(f"\n=== Quadratique : f = x² + 4y² ===\n")
    quad = lambda v: v[0]**2 + 4*v[1]**2
    r = newton_multivariable(quad, np.array([3.0, 2.0]))
    print(f"  Newton : x* = {np.round(r.x, 8)}, {r.iterations} itérations")

    print(f"\n=== Lagrange : min x²+y² sous x+y=1 ===\n")
    f = lambda v: v[0]**2 + v[1]**2
    g = lambda v: v[0] + v[1] - 1
    r = lagrange_2d(f, g, np.array([1.0, 0.0]))
    print(f"  x* = {np.round(r['x'], 6)}, λ = {r['lambda']:.4f}")
    print(f"  f(x*) = {r['f(x)']:.6f} (exact: 0.5)")
    print(f"  g(x*) = {r['g(x)']:.2e} ≈ 0 ✓")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tracer_trajectoire_2d(rosenbrock, [r_gd, r_bt, r_nw], (-2, 2), (-1, 3), "Rosenbrock", axes[0])
    tracer_lagrange(ax=axes[1])
    plt.tight_layout()
    plt.savefig("multivariable_optimization_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
