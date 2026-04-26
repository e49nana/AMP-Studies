"""
error_analysis_linsys.py
========================

Analyse d'erreur pour les systèmes linéaires Ax = b.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", section 2.3.6.

Couvre :
    - Erreur directe et rétrograde (Satz 2.21)
    - Borne d'erreur : ||Δx||/||x|| ≤ κ(A) · ||Δb||/||b|| (Satz 2.17)
    - Expérience : erreur vs κ(A) sur matrices aléatoires
    - Nombre de chiffres perdus ≈ log₁₀(κ(A))

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def erreur_directe(x_calc: np.ndarray, x_exact: np.ndarray) -> float:
    """||x_calc - x_exact||_∞ / ||x_exact||_∞."""
    return float(np.linalg.norm(x_calc - x_exact, np.inf) /
                 np.linalg.norm(x_exact, np.inf))


def erreur_retrograde(A: np.ndarray, x_calc: np.ndarray, b: np.ndarray) -> float:
    """||Ax - b||_∞ / ||b||_∞ (résidu relatif)."""
    return float(np.linalg.norm(A @ x_calc - b, np.inf) /
                 np.linalg.norm(b, np.inf))


def chiffres_perdus(kappa: float) -> int:
    """Nombre de chiffres décimaux perdus ≈ log₁₀(κ(A))."""
    if kappa <= 0:
        return 0
    return int(np.log10(kappa))


def experience_erreur_vs_cond(
    tailles: range = range(20, 200, 20),
    n_essais: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère des matrices de conditionnement varié et mesure l'erreur.
    Renvoie (conds, erreurs).
    """
    rng = np.random.default_rng(seed)
    conds, erreurs = [], []
    for n in tailles:
        for _ in range(n_essais):
            # Matrice avec conditionnement contrôlé via SVD
            U, _, Vt = np.linalg.svd(rng.standard_normal((n, n)))
            kappa = 10 ** rng.uniform(0, 14)
            sigmas = np.logspace(0, -np.log10(kappa), n)
            A = U[:, :n] @ np.diag(sigmas) @ Vt[:n, :]
            x_exact = rng.standard_normal(n)
            b = A @ x_exact
            x_calc = np.linalg.solve(A, b)
            conds.append(np.linalg.cond(A, np.inf))
            erreurs.append(erreur_directe(x_calc, x_exact))
    return np.array(conds), np.array(erreurs)


def tracer_erreur_vs_cond(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    conds, erreurs = experience_erreur_vs_cond(tailles=range(30, 60, 10), n_essais=80)
    eps = np.finfo(float).eps
    ax.loglog(conds, erreurs, "b.", alpha=0.3, markersize=3)
    cs = np.logspace(0, 16, 100)
    ax.loglog(cs, cs * eps, "r--", linewidth=2, label=r"$\kappa(A) \cdot \varepsilon_{mach}$")
    ax.set_xlabel(r"$\kappa_\infty(A)$")
    ax.set_ylabel("erreur relative directe")
    ax.set_title("Satz 2.17 — erreur bornée par $\\kappa \\cdot \\varepsilon_{mach}$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_chiffres_perdus(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    kappas = np.logspace(0, 16, 17)
    perdus = [chiffres_perdus(k) for k in kappas]
    ax.bar(range(len(kappas)), perdus, color="tab:red", alpha=0.7)
    ax.set_xticks(range(len(kappas)))
    ax.set_xticklabels([f"$10^{{{int(np.log10(k))}}}$" for k in kappas], rotation=45)
    ax.set_xlabel("$\\kappa(A)$")
    ax.set_ylabel("chiffres décimaux perdus")
    ax.set_title("Règle pratique : $\\kappa = 10^k$ → perte de $k$ chiffres")
    ax.grid(True, alpha=0.3, axis="y")
    return ax


if __name__ == "__main__":
    print("=== Chiffres perdus ===")
    for k in [1, 1e4, 1e8, 1e12, 1e16]:
        print(f"  κ = {k:.0e} → perte de ~{chiffres_perdus(k)} chiffres")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_erreur_vs_cond(ax=axes[0])
    tracer_chiffres_perdus(ax=axes[1])
    plt.tight_layout()
    plt.savefig("error_analysis_demo.png", dpi=120)
    print("Figure sauvegardée.")
