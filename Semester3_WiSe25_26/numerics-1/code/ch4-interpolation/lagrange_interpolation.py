"""
lagrange_interpolation.py
=========================

Interpolation de Lagrange — module standalone avec focus pédagogique.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", section 4.2.2.

Couvre :
    - Base de Lagrange L_i^n(x) (formule 4.2)
    - Polynôme interpolant p(x) = Σ y_i L_i(x) (formule 4.3)
    - Forme barycentrique (plus stable numériquement)
    - Visualisation des fonctions de base
    - Comparaison from-scratch vs numpy.polyfit

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def lagrange_basis(xs: np.ndarray, i: int, x: np.ndarray) -> np.ndarray:
    """L_i^n(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)."""
    x = np.asarray(x, dtype=float)
    val = np.ones_like(x)
    for j in range(len(xs)):
        if j != i:
            val *= (x - xs[j]) / (xs[i] - xs[j])
    return val


def lagrange_interpolation(xs: np.ndarray, ys: np.ndarray, x: np.ndarray) -> np.ndarray:
    """p(x) = Σ y_i L_i^n(x)."""
    x = np.asarray(x, dtype=float)
    return sum(ys[i] * lagrange_basis(xs, i, x) for i in range(len(xs)))


def lagrange_barycentrique(xs: np.ndarray, ys: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Forme barycentrique (numériquement plus stable, O(n) par point) :
        p(x) = Σ (w_i y_i / (x - x_i)) / Σ (w_i / (x - x_i))
    avec w_i = 1 / Π_{j≠i} (x_i - x_j).
    """
    x = np.asarray(x, dtype=float)
    n = len(xs)
    w = np.ones(n)
    for i in range(n):
        for j in range(n):
            if j != i:
                w[i] /= (xs[i] - xs[j])

    out = np.empty_like(x)
    for k, xk in enumerate(x):
        # Vérifier si xk coïncide avec un nœud
        exact = np.where(np.abs(xk - xs) < 1e-15)[0]
        if len(exact) > 0:
            out[k] = ys[exact[0]]
        else:
            terms = w / (xk - xs)
            out[k] = np.dot(terms, ys) / np.sum(terms)
    return out


def tracer_base_lagrange(xs: np.ndarray, ax: plt.Axes | None = None) -> plt.Axes:
    """Trace les n+1 fonctions de base L_i(x)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    x_fine = np.linspace(xs[0] - 0.5, xs[-1] + 0.5, 300)
    for i in range(len(xs)):
        ax.plot(x_fine, lagrange_basis(xs, i, x_fine), linewidth=1.5, label=f"$L_{i}$")
    ax.plot(xs, np.zeros_like(xs), "ko", markersize=6)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axhline(1, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlabel("$x$"); ax.set_ylabel("$L_i(x)$")
    ax.set_title("Fonctions de base de Lagrange (formule 4.2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    xs = np.array([0, 1, 3, 4], dtype=float)
    ys = np.array([1, 2, 0, 5], dtype=float)
    x_test = np.linspace(-0.5, 4.5, 200)

    p_std = lagrange_interpolation(xs, ys, x_test)
    p_bary = lagrange_barycentrique(xs, ys, x_test)
    print(f"||standard - barycentrique||_∞ = {np.max(np.abs(p_std - p_bary)):.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_base_lagrange(xs, ax=axes[0])
    axes[1].plot(x_test, p_std, "b-", linewidth=2, label="$p(x)$")
    axes[1].plot(xs, ys, "ro", markersize=8, label="données")
    axes[1].set_title("Interpolation de Lagrange")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lagrange_demo.png", dpi=120)
    print("Figure sauvegardée.")
