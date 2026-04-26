"""
neville_aitken.py
=================

Schéma de Neville-Aitken — module standalone.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", Lemma 4.6.

Couvre :
    - Schéma de Neville-Aitken (formule récursive)
    - Tableau complet des p_{j,k}(x)
    - Reproduction de l'Übung 4.7
    - Estimation d'erreur par différence des dernières colonnes

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def neville_tableau(xs: np.ndarray, ys: np.ndarray, x: float) -> np.ndarray:
    """
    Construit le tableau complet de Neville-Aitken.

    T[j][k] = p_{j, j+k}(x) pour k = 0, 1, ..., n-j.

    Renvoie la matrice triangulaire T (n×n) où T[j, k] = p_{j, j+k}(x).
    """
    n = len(xs)
    T = np.zeros((n, n))
    T[:, 0] = ys  # colonne 0 : p_{jj} = y_j

    for k in range(1, n):
        for j in range(n - k):
            T[j, k] = ((xs[j + k] - x) * T[j, k - 1] +
                        (x - xs[j]) * T[j + 1, k - 1]) / (xs[j + k] - xs[j])
    return T


def neville_valeur(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    """Évalue p(x) par Neville-Aitken. Renvoie T[0, n-1]."""
    T = neville_tableau(xs, ys, x)
    return float(T[0, -1])


def neville_erreur_estimation(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    """
    Estimation de l'erreur par |T[0,n-1] - T[0,n-2]|.
    Heuristique : la différence entre les deux dernières colonnes.
    """
    T = neville_tableau(xs, ys, x)
    n = len(xs)
    if n < 2:
        return 0.0
    return abs(T[0, n - 1] - T[0, n - 2])


def afficher_tableau(xs: np.ndarray, ys: np.ndarray, x: float) -> None:
    """Affiche le tableau de Neville formaté."""
    T = neville_tableau(xs, ys, x)
    n = len(xs)
    print(f"Tableau de Neville pour x = {x}")
    print(f"{'j':>3} | {'x_j':>6} | " + " | ".join(f"k={k}" for k in range(n)))
    print("-" * (20 + 12 * n))
    for j in range(n):
        row = f"{j:>3} | {xs[j]:>6.2f} | "
        for k in range(n):
            if k <= n - 1 - j:
                row += f"{T[j, k]:>10.4f} | "
            else:
                row += "           | "
        print(row)
    print(f"\np({x}) = {T[0, n-1]:.10f}")


if __name__ == "__main__":
    print("=== Übung 4.7 ===")
    xs = np.array([0, 1, 3, 4], dtype=float)
    ys = np.array([12, 3, -3, 12], dtype=float)
    afficher_tableau(xs, ys, 2.0)

    print(f"\n=== Estimation d'erreur ===")
    err = neville_erreur_estimation(xs, ys, 2.0)
    print(f"  |T[0,3] - T[0,2]| = {err:.6f}")

    # Tracé
    x_fine = np.linspace(-0.5, 4.5, 200)
    y_fine = [neville_valeur(xs, ys, xi) for xi in x_fine]
    plt.figure(figsize=(8, 5))
    plt.plot(x_fine, y_fine, "b-", linewidth=2, label="$p(x)$ (Neville)")
    plt.plot(xs, ys, "ro", markersize=8, label="données")
    plt.axhline(0, color="grey", linewidth=0.5)
    plt.xlabel("$x$"); plt.ylabel("$p(x)$")
    plt.title("Übung 4.7 — Neville-Aitken"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig("neville_demo.png", dpi=120)
    print("Figure sauvegardée.")
