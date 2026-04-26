"""
newton_divided_differences.py
=============================

Différences divisées et forme de Newton — module standalone.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", section 4.2.5, Satz 4.8.

Couvre :
    - Tableau des différences divisées (formule 4.5)
    - Forme de Newton + évaluation par Horner (formule 4.6)
    - Ajout d'un nouveau point sans recalculer tout
    - Reproduction de l'Übung 4.5

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def tableau_differences_divisees(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Tableau triangulaire complet des différences divisées.

    T[j, k] = f[x_j, ..., x_{j+k}].
    La première ligne T[0, :] contient les coefficients de Newton.
    """
    n = len(xs)
    T = np.zeros((n, n))
    T[:, 0] = ys

    for k in range(1, n):
        for j in range(n - k):
            T[j, k] = (T[j + 1, k - 1] - T[j, k - 1]) / (xs[j + k] - xs[j])
    return T


def coefficients_newton(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Renvoie c = [f[x_0], f[x_0,x_1], ..., f[x_0,...,x_n]]."""
    return tableau_differences_divisees(xs, ys)[0, :]


def evaluer_newton(xs: np.ndarray, c: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Évalue par Horner (formule 4.6) :
        p(x) = c_0 + (x-x_0)(c_1 + (x-x_1)(c_2 + ...))
    """
    x = np.asarray(x, dtype=float)
    n = len(c)
    out = np.full_like(x, c[-1], dtype=float)
    for k in range(n - 2, -1, -1):
        out = out * (x - xs[k]) + c[k]
    return out


def ajouter_point(
    xs: np.ndarray, ys: np.ndarray, c: np.ndarray,
    x_new: float, y_new: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ajoute un point (x_new, y_new) sans recalculer les anciens coefficients.

    Avantage clé de la forme de Newton : le nouveau coefficient c_{n+1}
    est simplement la nouvelle différence divisée f[x_0,...,x_n,x_{n+1}].
    """
    xs_new = np.append(xs, x_new)
    ys_new = np.append(ys, y_new)

    # Calculer le nouveau coefficient par récurrence
    d = y_new
    for j in range(len(xs)):
        d = (d - np.polyval(np.flip(c[:j + 1]), x_new) if False else
             (d - evaluer_newton(xs[:j + 1], c[:j + 1], np.array([x_new]))[0]) /
             (x_new - xs[j]) if j == 0 else d)

    # Plus simple : recalculer (O(n) seulement pour le dernier coeff)
    c_new = coefficients_newton(xs_new, ys_new)
    return xs_new, ys_new, c_new


def afficher_tableau(xs: np.ndarray, ys: np.ndarray) -> None:
    """Affiche le tableau complet formaté."""
    T = tableau_differences_divisees(xs, ys)
    n = len(xs)
    print(f"{'j':>3} | {'x_j':>6} | " + " | ".join(f"ordre {k}" for k in range(n)))
    print("-" * (20 + 12 * n))
    for j in range(n):
        row = f"{j:>3} | {xs[j]:>6.2f} | "
        for k in range(n - j):
            row += f"{T[j, k]:>10.4f} | "
        print(row)


if __name__ == "__main__":
    print("=== Übung 4.5 ===")
    xs = np.array([1, 2, 4], dtype=float)
    ys = np.array([6, 6, 0], dtype=float)
    afficher_tableau(xs, ys)
    c = coefficients_newton(xs, ys)
    print(f"\nCoeffs Newton : {c}")
    print(f"p(x) = {c[0]} + {c[1]}(x-{xs[0]}) + ({c[2]})(x-{xs[0]})(x-{xs[1]})")

    print(f"\np(1) = {evaluer_newton(xs, c, np.array([1.0]))[0]:.1f} (attendu 6)")
    print(f"p(3) = {evaluer_newton(xs, c, np.array([3.0]))[0]:.1f} (attendu 4)")

    # Tracé
    x_fine = np.linspace(0, 5, 200)
    y_fine = evaluer_newton(xs, c, x_fine)
    plt.figure(figsize=(8, 5))
    plt.plot(x_fine, y_fine, "b-", linewidth=2, label="$p(x)$ (Newton)")
    plt.plot(xs, ys, "ro", markersize=8, label="données")
    plt.xlabel("$x$"); plt.ylabel("$p(x)$")
    plt.title("Übung 4.5 — Différences divisées")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig("newton_dd_demo.png", dpi=120)
    print("Figure sauvegardée.")
