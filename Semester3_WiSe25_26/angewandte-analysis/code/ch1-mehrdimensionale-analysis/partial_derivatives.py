"""
partial_derivatives.py
======================

Dérivées partielles, gradient et Hessienne.

Couvre :
    - Dérivées partielles ∂f/∂xᵢ (différences finies)
    - Gradient ∇f : direction de plus forte croissance
    - Hessienne H : matrice des dérivées secondes
    - Différentielle totale : df = ∇f · dx
    - Jacobienne pour les fonctions vectorielles
    - Visualisation : surface, gradient, courbes de niveau

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Dérivées partielles
# ======================================================================

def derivee_partielle(
    f: Callable[..., float], x: np.ndarray, i: int, h: float = 1e-6,
) -> float:
    """∂f/∂xᵢ ≈ (f(x + hεᵢ) - f(x - hεᵢ)) / (2h)."""
    x = np.asarray(x, dtype=float)
    e = np.zeros_like(x)
    e[i] = h
    return (f(*(x + e)) - f(*(x - e))) / (2 * h)


def gradient(f: Callable, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """∇f(x) = (∂f/∂x₁, ..., ∂f/∂xₙ)."""
    x = np.asarray(x, dtype=float)
    return np.array([derivee_partielle(f, x, i, h) for i in range(len(x))])


def hessienne(f: Callable, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """H_ij = ∂²f / (∂xᵢ ∂xⱼ)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i] = h; ej[j] = h
            H[i, j] = (f(*(x+ei+ej)) - f(*(x+ei-ej)) - f(*(x-ei+ej)) + f(*(x-ei-ej))) / (4*h*h)
    return H


def jacobienne(
    F: Callable[..., np.ndarray], x: np.ndarray, h: float = 1e-6,
) -> np.ndarray:
    """J_ij = ∂Fᵢ/∂xⱼ pour F : Rⁿ → Rᵐ."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    F0 = np.asarray(F(*x))
    m = len(F0)
    J = np.zeros((m, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        J[:, j] = (np.asarray(F(*(x+e))) - np.asarray(F(*(x-e)))) / (2*h)
    return J


def differentielle_totale(f: Callable, x: np.ndarray, dx: np.ndarray) -> float:
    """df = ∇f · dx (approximation linéaire du changement)."""
    grad = gradient(f, x)
    return float(np.dot(grad, dx))


# ======================================================================
#  2. Classification des points critiques
# ======================================================================

def classifier_point_critique(f: Callable, x: np.ndarray) -> str:
    """Classifie un point critique par le test de la Hessienne (2D)."""
    H = hessienne(f, x)
    if len(x) != 2:
        eigvals = np.linalg.eigvalsh(H)
        if np.all(eigvals > 0):
            return "minimum local"
        elif np.all(eigvals < 0):
            return "maximum local"
        else:
            return "point selle"

    fxx, fxy, fyy = H[0, 0], H[0, 1], H[1, 1]
    D = fxx * fyy - fxy**2
    if D > 0 and fxx > 0:
        return "minimum local"
    elif D > 0 and fxx < 0:
        return "maximum local"
    elif D < 0:
        return "point selle"
    return "indéterminé"


# ======================================================================
#  3. Tracés
# ======================================================================

def tracer_gradient_2d(f: Callable, x_range: tuple, y_range: tuple,
                        nom: str = "f", ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    x = np.linspace(*x_range, 50)
    y = np.linspace(*y_range, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(f)(X, Y)

    ax.contour(X, Y, Z, levels=20, cmap="viridis", alpha=0.7)
    cs = ax.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.3)
    plt.colorbar(cs, ax=ax, label="$f(x,y)$")

    # Gradient sur grille lâche
    xg = np.linspace(*x_range, 12)
    yg = np.linspace(*y_range, 12)
    Xg, Yg = np.meshgrid(xg, yg)
    U, V = np.zeros_like(Xg), np.zeros_like(Yg)
    for i in range(Xg.shape[0]):
        for j in range(Xg.shape[1]):
            g = gradient(f, np.array([Xg[i,j], Yg[i,j]]))
            U[i,j], V[i,j] = g[0], g[1]

    ax.quiver(Xg, Yg, U, V, color="red", alpha=0.6, scale=None)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(f"Courbes de niveau + $\\nabla {nom}$ (rouge)")
    ax.set_aspect("equal")
    return ax


def tracer_surface_3d(f: Callable, x_range: tuple, y_range: tuple,
                       nom: str = "f", ax=None) -> None:
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    x = np.linspace(*x_range, 60)
    y = np.linspace(*y_range, 60)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(f)(X, Y)

    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7, edgecolor="none")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$f$")
    ax.set_title(f"${nom}(x,y)$")


def tracer_classification(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre min, max, selle sur la même figure."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    f = lambda x, y: x**2 - y**2  # selle en (0,0)
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(f)(X, Y)

    ax.contour(X, Y, Z, levels=20, cmap="RdBu_r")
    ax.plot(0, 0, "ko", markersize=10, label=f"selle (0,0)")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("$f = x^2 - y^2$ (point selle)")
    ax.legend(); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Gradient et Hessienne ===\n")
    f = lambda x, y: x**2 + 2*y**2 - x*y + 3*x - 2*y + 5
    pt = np.array([1.0, 1.0])
    g = gradient(f, pt)
    H = hessienne(f, pt)
    print(f"  f(x,y) = x² + 2y² - xy + 3x - 2y + 5")
    print(f"  ∇f(1,1) = {np.round(g, 4)}")
    print(f"  H = {np.round(H, 4).tolist()}")
    print(f"  det(H) = {np.linalg.det(H):.4f}")

    print(f"\n=== Points critiques ===\n")
    # ∇f = 0 → 2x - y + 3 = 0, 4y - x - 2 = 0 → x = -10/7, y = 1/7
    x_crit = np.array([-10/7, 1/7])
    g_crit = gradient(f, x_crit)
    cls = classifier_point_critique(f, x_crit)
    print(f"  Point critique : {np.round(x_crit, 4)}")
    print(f"  ∇f = {np.round(g_crit, 6)} ≈ 0 ✓")
    print(f"  Classification : {cls}")

    print(f"\n=== Point selle ===\n")
    g_selle = lambda x, y: x**2 - y**2
    cls_s = classifier_point_critique(g_selle, np.array([0.0, 0.0]))
    print(f"  f = x² - y² en (0,0) : {cls_s}")

    print(f"\n=== Jacobienne ===\n")
    F = lambda x, y: np.array([x**2 + y, x*y - 1])
    J = jacobienne(F, np.array([1.0, 2.0]))
    print(f"  F(x,y) = (x²+y, xy-1)")
    print(f"  J(1,2) = {np.round(J, 4).tolist()}")
    print(f"  Exact : [[2, 1], [2, 1]] ✓")

    print(f"\n=== Différentielle totale ===\n")
    dx = np.array([0.01, -0.02])
    df = differentielle_totale(f, pt, dx)
    df_exact = f(*(pt + dx)) - f(*pt)
    print(f"  df ≈ ∇f·dx = {df:.6f}")
    print(f"  Δf exact   = {df_exact:.6f}")
    print(f"  Erreur = {abs(df - df_exact):.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tracer_gradient_2d(f, (-3, 2), (-1, 3), "f", ax=axes[0])
    tracer_classification(ax=axes[1])
    plt.tight_layout()
    plt.savefig("partial_derivatives_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
