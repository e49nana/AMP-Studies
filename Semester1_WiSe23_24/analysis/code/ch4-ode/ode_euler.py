"""
ode_euler.py
============

Méthode d'Euler pour les EDO : y' = f(t, y), y(t₀) = y₀.

Couvre :
    - Euler explicite (forward) : y_{k+1} = y_k + h·f(t_k, y_k)
    - Euler implicite (backward) : y_{k+1} = y_k + h·f(t_{k+1}, y_{k+1})
    - Erreur locale O(h²), erreur globale O(h)
    - Stabilité : Euler explicite instable pour h trop grand
    - Comparaison avec solution exacte

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def euler_explicite(
    f: Callable[[float, float], float],
    t0: float, y0: float, t_end: float, h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Euler explicite : y_{k+1} = y_k + h·f(t_k, y_k).
    Renvoie (t_array, y_array).
    """
    n = int((t_end - t0) / h)
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    t[0], y[0] = t0, y0

    for k in range(n):
        y[k + 1] = y[k] + h * f(t[k], y[k])
        t[k + 1] = t[k] + h

    return t, y


def euler_implicite(
    f: Callable[[float, float], float],
    t0: float, y0: float, t_end: float, h: float,
    n_newton: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Euler implicite : y_{k+1} = y_k + h·f(t_{k+1}, y_{k+1}).
    Résolu par itération de point fixe (Newton simplifié).
    Plus stable que l'explicite pour les problèmes raides.
    """
    n = int((t_end - t0) / h)
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    t[0], y[0] = t0, y0

    for k in range(n):
        t[k + 1] = t[k] + h
        # Point fixe : y = y_k + h·f(t_{k+1}, y)
        y_guess = y[k] + h * f(t[k], y[k])  # prédicteur explicite
        for _ in range(n_newton):
            y_guess = y[k] + h * f(t[k + 1], y_guess)
        y[k + 1] = y_guess

    return t, y


def erreur_globale(y_num: np.ndarray, y_exact: np.ndarray) -> float:
    """Erreur globale max |y_num - y_exact|."""
    return float(np.max(np.abs(y_num - y_exact)))


def tracer_euler(
    f: Callable, t0: float, y0: float, t_end: float,
    y_exact: Callable | None = None,
    hs: tuple[float, ...] = (0.5, 0.2, 0.1, 0.01),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if y_exact is not None:
        t_fine = np.linspace(t0, t_end, 500)
        ax.plot(t_fine, y_exact(t_fine), "k-", linewidth=2.5, label="exacte")

    for h in hs:
        t, y = euler_explicite(f, t0, y0, t_end, h)
        ax.plot(t, y, "o-", markersize=3, linewidth=1, label=f"h = {h}")

    ax.set_xlabel("$t$"); ax.set_ylabel("$y$")
    ax.set_title("Euler explicite")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_convergence_ordre(
    f: Callable, t0: float, y0: float, t_end: float,
    y_exact: Callable,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    hs = np.logspace(-3, -0.3, 20)
    errs_exp, errs_imp = [], []

    for h in hs:
        t, y = euler_explicite(f, t0, y0, t_end, h)
        errs_exp.append(erreur_globale(y, y_exact(t)))
        t, y = euler_implicite(f, t0, y0, t_end, h)
        errs_imp.append(erreur_globale(y, y_exact(t)))

    ax.loglog(hs, errs_exp, "rs-", markersize=4, label="Euler explicite")
    ax.loglog(hs, errs_imp, "bo-", markersize=4, label="Euler implicite")
    ax.loglog(hs, hs, "k:", alpha=0.3, label="$O(h)$")
    ax.set_xlabel("$h$"); ax.set_ylabel("erreur globale")
    ax.set_title("Ordre de convergence : $O(h)$ pour les deux")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    return ax


def demo_stabilite(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre l'instabilité d'Euler explicite pour h trop grand."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # y' = -10y, y(0) = 1 → y = e^{-10t}
    f = lambda t, y: -10 * y
    exact = lambda t: np.exp(-10 * t)
    t_fine = np.linspace(0, 2, 300)
    ax.plot(t_fine, exact(t_fine), "k-", linewidth=2, label="exacte")

    for h in [0.05, 0.15, 0.19, 0.21]:
        t, y = euler_explicite(f, 0, 1, 2, h)
        style = "-" if max(np.abs(y)) < 10 else "--"
        ax.plot(t, y, f"o{style}", markersize=3, linewidth=1,
                label=f"h={h} {'✓' if h < 0.2 else '✗ instable'}")

    ax.set_ylim(-2, 3)
    ax.set_xlabel("$t$"); ax.set_ylabel("$y$")
    ax.set_title("Stabilité : $y' = -10y$, instable si $h > 2/|\\lambda| = 0.2$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    # y' = -y, y(0) = 1 → y = e^{-t}
    f = lambda t, y: -y
    exact = lambda t: np.exp(-t)

    print("=== y' = -y, y(0) = 1 ===\n")
    for h in [0.5, 0.1, 0.01]:
        t, y = euler_explicite(f, 0, 1, 3, h)
        err = erreur_globale(y, exact(t))
        print(f"  h = {h:<5} : y(3) = {y[-1]:.8f} (exact: {exact(3):.8f}), "
              f"err_glob = {err:.2e}")

    print(f"\n=== Euler implicite ===\n")
    for h in [0.5, 0.1, 0.01]:
        t, y = euler_implicite(f, 0, 1, 3, h)
        err = erreur_globale(y, exact(t))
        print(f"  h = {h:<5} : y(3) = {y[-1]:.8f}, err_glob = {err:.2e}")

    print(f"\n=== Stabilité : y' = -10y ===")
    print(f"  Condition de stabilité Euler explicite : h < 2/|λ| = 0.2")
    for h in [0.1, 0.19, 0.21, 0.3]:
        t, y = euler_explicite(lambda t, y: -10*y, 0, 1, 1, h)
        print(f"  h = {h} : y(1) = {y[-1]:.6f} "
              f"{'✓' if abs(y[-1]) < 10 else '✗ instable'}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_euler(f, 0, 1, 3, exact, ax=axes[0])
    tracer_convergence_ordre(f, 0, 1, 3, exact, ax=axes[1])
    demo_stabilite(ax=axes[2])
    plt.tight_layout()
    plt.savefig("ode_euler_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
