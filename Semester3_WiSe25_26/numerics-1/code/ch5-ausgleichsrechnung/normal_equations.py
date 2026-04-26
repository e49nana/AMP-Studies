"""
normal_equations.py
===================

Moindres carrés par équations normales (méthode "naïve").

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", section 5.2.

Couvre :
    - Dérivation des équations normales AᵀA x = Aᵀb (Satz 5.2)
    - Résolution par Cholesky (AᵀA est SPD si A a rang plein)
    - Instabilité numérique : κ(AᵀA) = κ(A)² (section 5.4)
    - Comparaison directe avec QR

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def equations_normales(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Résout ||Ax - b|| → min par AᵀA x = Aᵀb."""
    C = A.T @ A
    d = A.T @ b
    return np.linalg.solve(C, d)


def equations_normales_cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Résout via Cholesky (AᵀA est SPD)."""
    C = A.T @ A
    d = A.T @ b
    L = np.linalg.cholesky(C)
    # Ly = d
    y = np.linalg.solve(L, d)
    # Lᵀx = y
    return np.linalg.solve(L.T, y)


def comparer_stabilite(max_degre: int = 18) -> tuple[list[int], list[float], list[float]]:
    """Compare erreur éq. normales vs QR sur Vandermonde."""
    degres = list(range(3, max_degre))
    err_ne, err_qr = [], []
    for n in degres:
        x = np.linspace(0, 1, 50)
        A = np.column_stack([x**k for k in range(n)])
        x_true = np.ones(n)
        b = A @ x_true
        try:
            x_ne = equations_normales(A, b)
            err_ne.append(np.linalg.norm(x_ne - x_true, np.inf))
        except Exception:
            err_ne.append(np.nan)
        Q, R = np.linalg.qr(A)
        x_qr = np.linalg.solve(R[:n], (Q.T @ b)[:n])
        err_qr.append(np.linalg.norm(x_qr - x_true, np.inf))
    return degres, err_ne, err_qr


def tracer_stabilite(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    degres, err_ne, err_qr = comparer_stabilite()
    ax.semilogy(degres, err_ne, "rs-", label="Éq. normales ($\\kappa^2$)", markersize=4)
    ax.semilogy(degres, err_qr, "bo-", label="QR ($\\kappa$)", markersize=4)
    ax.set_xlabel("degré du polynôme")
    ax.set_ylabel("erreur")
    ax.set_title("§5.4 — $\\kappa(A^T A) = \\kappa(A)^2$")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_condition_squared(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre κ(AᵀA) ≈ κ(A)² empiriquement."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(42)
    kA, kATA = [], []
    for _ in range(200):
        m, n = 30, 10
        A = rng.standard_normal((m, n))
        kA.append(np.linalg.cond(A))
        kATA.append(np.linalg.cond(A.T @ A))
    ax.loglog(kA, kATA, "b.", alpha=0.4)
    xs = np.logspace(0, 3, 50)
    ax.loglog(xs, xs**2, "r--", label="$\\kappa(A)^2$")
    ax.set_xlabel("$\\kappa(A)$"); ax.set_ylabel("$\\kappa(A^T A)$")
    ax.set_title("Vérification $\\kappa(A^T A) = \\kappa(A)^2$")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Régression linéaire ===")
    rng = np.random.default_rng(42)
    x = np.linspace(0, 5, 30)
    y = 2.5 * x + 1 + rng.normal(0, 0.5, 30)
    A = np.column_stack([np.ones_like(x), x])
    c = equations_normales(A, y)
    print(f"  y ≈ {c[0]:.4f} + {c[1]:.4f} x  (attendu : 1 + 2.5 x)")

    print("\n=== Stabilité ===")
    degres, err_ne, err_qr = comparer_stabilite(15)
    for d, e_ne, e_qr in zip(degres, err_ne, err_qr):
        print(f"  deg {d:>2} : éq. norm. = {e_ne:.2e}, QR = {e_qr:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_stabilite(ax=axes[0])
    tracer_condition_squared(ax=axes[1])
    plt.tight_layout()
    plt.savefig("normal_equations_demo.png", dpi=120)
    print("Figure sauvegardée.")
