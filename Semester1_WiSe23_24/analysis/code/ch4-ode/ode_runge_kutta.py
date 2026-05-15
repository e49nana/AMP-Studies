"""
ode_runge_kutta.py
==================

Méthodes de Runge-Kutta : RK2 (Heun) et RK4 classique.

Couvre :
    - RK2 (méthode de Heun) : erreur O(h²)
    - RK4 classique : erreur O(h⁴)
    - Tableau de Butcher
    - Comparaison Euler vs RK2 vs RK4 : précision vs coût
    - Contrôle adaptatif du pas (introduction)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def rk2_heun(
    f: Callable[[float, float], float],
    t0: float, y0: float, t_end: float, h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RK2 (Heun) :
        k₁ = f(t, y)
        k₂ = f(t+h, y+h·k₁)
        y⁺ = y + h/2 · (k₁ + k₂).
    Erreur globale O(h²). 2 évaluations de f par pas.
    """
    n = int((t_end - t0) / h)
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    t[0], y[0] = t0, y0

    for k in range(n):
        k1 = f(t[k], y[k])
        k2 = f(t[k] + h, y[k] + h * k1)
        y[k + 1] = y[k] + h / 2 * (k1 + k2)
        t[k + 1] = t[k] + h

    return t, y


def rk4(
    f: Callable[[float, float], float],
    t0: float, y0: float, t_end: float, h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RK4 classique :
        k₁ = f(t, y)
        k₂ = f(t+h/2, y+h/2·k₁)
        k₃ = f(t+h/2, y+h/2·k₂)
        k₄ = f(t+h, y+h·k₃)
        y⁺ = y + h/6 · (k₁ + 2k₂ + 2k₃ + k₄).
    Erreur globale O(h⁴). 4 évaluations de f par pas.
    Le « couteau suisse » des solveurs d'EDO.
    """
    n = int((t_end - t0) / h)
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    t[0], y[0] = t0, y0

    for k in range(n):
        k1 = f(t[k], y[k])
        k2 = f(t[k] + h/2, y[k] + h/2 * k1)
        k3 = f(t[k] + h/2, y[k] + h/2 * k2)
        k4 = f(t[k] + h, y[k] + h * k3)
        y[k + 1] = y[k] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t[k + 1] = t[k] + h

    return t, y


def euler(f, t0, y0, t_end, h):
    """Euler explicite pour comparaison."""
    n = int((t_end - t0) / h)
    t = np.zeros(n + 1); y = np.zeros(n + 1)
    t[0], y[0] = t0, y0
    for k in range(n):
        y[k+1] = y[k] + h * f(t[k], y[k])
        t[k+1] = t[k] + h
    return t, y


def comparer_ordres(
    f: Callable, t0: float, y0: float, t_end: float,
    y_exact: Callable,
) -> dict[str, list[tuple[float, float]]]:
    """Compare les erreurs globales pour différents h."""
    hs = np.logspace(-3, -0.5, 15)
    result = {"Euler O(h)": [], "RK2 O(h²)": [], "RK4 O(h⁴)": []}

    for h in hs:
        for nom, methode in [("Euler O(h)", euler), ("RK2 O(h²)", rk2_heun), ("RK4 O(h⁴)", rk4)]:
            t, y = methode(f, t0, y0, t_end, h)
            err = np.max(np.abs(y - y_exact(t)))
            result[nom].append((h, err))

    return result


def tracer_comparaison(
    f: Callable, t0: float, y0: float, t_end: float,
    y_exact: Callable, nom: str = "y' = f(t,y)",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    comp = comparer_ordres(f, t0, y0, t_end, y_exact)
    styles = {"Euler O(h)": "rs-", "RK2 O(h²)": "g^-", "RK4 O(h⁴)": "bo-"}

    for nom_m, data in comp.items():
        hs, errs = zip(*data)
        ax.loglog(hs, errs, styles[nom_m], markersize=4, label=nom_m)

    hs = np.logspace(-3, -0.5, 50)
    ax.loglog(hs, hs, "r:", alpha=0.2)
    ax.loglog(hs, hs**2, "g:", alpha=0.2)
    ax.loglog(hs, hs**4, "b:", alpha=0.2)

    ax.set_xlabel("$h$"); ax.set_ylabel("erreur globale")
    ax.set_title(f"Comparaison : {nom}")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_solutions(
    f: Callable, t0: float, y0: float, t_end: float,
    y_exact: Callable | None, h: float = 0.2,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if y_exact is not None:
        t_fine = np.linspace(t0, t_end, 300)
        ax.plot(t_fine, y_exact(t_fine), "k-", linewidth=2.5, label="exacte")

    for nom, methode, style in [("Euler", euler, "rs--"), ("RK2", rk2_heun, "g^--"), ("RK4", rk4, "bo-")]:
        t, y = methode(f, t0, y0, t_end, h)
        ax.plot(t, y, style, markersize=5, linewidth=1.5, label=nom)

    ax.set_xlabel("$t$"); ax.set_ylabel("$y$")
    ax.set_title(f"$h = {h}$ — même pas, précision très différente")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    # y' = -y, y(0) = 1 → y = e^{-t}
    f = lambda t, y: -y
    exact = lambda t: np.exp(-t)

    print("=== y' = -y, y(0) = 1, h = 0.1 ===\n")
    h = 0.1
    for nom, methode in [("Euler", euler), ("RK2", rk2_heun), ("RK4", rk4)]:
        t, y = methode(f, 0, 1, 3, h)
        err = np.max(np.abs(y - exact(t)))
        print(f"  {nom:5s} : y(3) = {y[-1]:.12f}, err = {err:.2e}")
    print(f"  exact : y(3) = {exact(3):.12f}")

    print(f"\n=== Coût-efficacité ===")
    print(f"  {'h':>8} | {'Euler err':>12} | {'RK4 err':>12} | RK4/Euler")
    print("  " + "-" * 52)
    for h in [0.5, 0.2, 0.1, 0.05]:
        _, ye = euler(f, 0, 1, 3, h)
        _, yr = rk4(f, 0, 1, 3, h)
        ee = np.max(np.abs(ye - exact(np.linspace(0, 3, len(ye)))))
        er = np.max(np.abs(yr - exact(np.linspace(0, 3, len(yr)))))
        print(f"  {h:>8} | {ee:>12.2e} | {er:>12.2e} | {ee/er:>8.0f}×")

    print(f"\n=== Tableau de Butcher RK4 ===")
    print(f"  0   |")
    print(f"  1/2 | 1/2")
    print(f"  1/2 | 0   1/2")
    print(f"  1   | 0   0   1")
    print(f"  ----|----------------")
    print(f"      | 1/6 1/3 1/3 1/6")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_solutions(f, 0, 1, 3, exact, h=0.5, ax=axes[0])
    tracer_comparaison(f, 0, 1, 3, exact, "y' = -y", ax=axes[1])
    plt.tight_layout()
    plt.savefig("ode_runge_kutta_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
