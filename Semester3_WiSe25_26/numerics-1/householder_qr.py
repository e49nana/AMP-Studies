"""
householder_qr.py
=================

Décomposition QR par réflexions de Householder et résolution de
problèmes de moindres carrés.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", chapitre 5.

Couvre :
    - Réflexion de Householder (Satz 5.6)
    - Décomposition QR (section 5.3.4)
    - Moindres carrés par QR (section 5.3.2)
    - Moindres carrés par équations normales (section 5.2)
    - Applications : droites, polynômes, modèles non-linéaires linéarisés
      (sections 5.5.1 - 5.5.3)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Réflexion de Householder (Satz 5.6)
# ======================================================================

def householder_reflector(a: np.ndarray) -> np.ndarray:
    """
    Construit le vecteur w de la réflexion de Householder qui envoie
    a sur ±||a||_2 · e_1 (section 5.3.3).

    Choix du signe : w = a + sign(a_1)||a||_2 e_1 pour éviter
    l'Auslöschung dans la première composante.
    """
    n = len(a)
    norm_a = np.linalg.norm(a, 2)
    if norm_a == 0:
        return np.zeros(n)
    sign = 1.0 if a[0] >= 0 else -1.0
    w = a.copy()
    w[0] += sign * norm_a
    return w


def appliquer_householder(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Applique la réflexion Q = I - 2 wwᵀ/||w||² à la matrice X :
        QX = X - 2 w (wᵀ X) / ||w||²

    Coût : O(m·n) au lieu de O(m²·n) pour la multiplication matricielle.
    """
    ww = np.dot(w, w)
    if ww == 0:
        return X.copy()
    factor = 2.0 / ww
    return X - factor * np.outer(w, w @ X)


# ======================================================================
#  2. Décomposition QR (section 5.3.4)
# ======================================================================

@dataclass
class DecompositionQR:
    """Résultat Q A = R (Q orthogonale, R triangulaire supérieure)."""
    R: np.ndarray
    Q: np.ndarray
    reflectors: list[tuple[int, np.ndarray]]  # (offset, w) pour chaque étape

    @property
    def rang(self) -> int:
        """Rang numérique estimé."""
        return int(np.sum(np.abs(np.diag(self.R[:self.R.shape[1], :])) > 1e-12))


def decomposition_qr(A: np.ndarray) -> DecompositionQR:
    """
    QR par réflexions de Householder (section 5.3.4).

    À chaque étape k, on annule les éléments sous-diagonaux de la
    colonne k en appliquant une réflexion de Householder à la
    sous-matrice A[k:, k:].

    Renvoie Q (N×N orthogonale) et R (N×n triangulaire supérieure)
    tels que Q @ A = R.
    """
    A = np.asarray(A, dtype=float).copy()
    N, n = A.shape
    Q = np.eye(N)
    reflectors = []

    for k in range(min(N - 1, n)):
        # Sous-colonne à éliminer
        a = A[k:, k].copy()
        w = householder_reflector(a)
        reflectors.append((k, w))

        # Appliquer la réflexion à la sous-matrice
        A[k:, k:] = appliquer_householder(w, A[k:, k:])
        # Appliquer aussi à Q (pour construire Q explicitement)
        Q[k:, :] = appliquer_householder(w, Q[k:, :])

    return DecompositionQR(R=A, Q=Q, reflectors=reflectors)


# ======================================================================
#  3. Moindres carrés
# ======================================================================

def moindres_carres_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Résout ||Ax - b||_2 → min par décomposition QR (section 5.3.2).

    1. QA = R  (QR-Zerlegung)
    2. c = Qb  (rotation du second membre)
    3. R₁ x = c₁  (substitution arrière sur les n premières lignes)

    Renvoie x (vecteur de taille n).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    N, n = A.shape

    qr = decomposition_qr(A)
    c = qr.Q @ b  # = Qb

    # Résoudre R₁ x = c₁ par substitution arrière
    R1 = qr.R[:n, :n]
    c1 = c[:n]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c1[i] - R1[i, i + 1:] @ x[i + 1:]) / R1[i, i]

    return x


def moindres_carres_normales(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Résout ||Ax - b||_2 → min par équations normales (section 5.2) :
        AᵀA x = Aᵀb.

    Plus simple mais moins stable (κ(AᵀA) = κ(A)²).
    """
    C = A.T @ A
    d = A.T @ b
    return np.linalg.solve(C, d)


def residu_moindres_carres(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """||Ax - b||_2."""
    return float(np.linalg.norm(A @ x - b, 2))


# ======================================================================
#  4. Applications (sections 5.5.1 - 5.5.3)
# ======================================================================

def ajustement_polynomial(
    x_data: np.ndarray, y_data: np.ndarray, degre: int,
) -> np.ndarray:
    """
    Ajustement polynomial de degré `degre` au sens des moindres carrés.

    Construit la matrice de Vandermonde et résout par QR.
    Renvoie les coefficients [a₀, a₁, ..., a_d] tels que
    p(x) = a₀ + a₁x + a₂x² + ... + a_d x^d.
    """
    N = len(x_data)
    A = np.column_stack([x_data**k for k in range(degre + 1)])
    return moindres_carres_qr(A, y_data)


def ajustement_exponentiel(
    x_data: np.ndarray, y_data: np.ndarray,
) -> tuple[float, float]:
    """
    Ajuste y = a · e^(bx) par linéarisation (section 5.5.3) :
        ln(y) = ln(a) + b x  →  problème linéaire en (ln(a), b).

    Renvoie (a, b).
    """
    log_y = np.log(y_data)
    A = np.column_stack([np.ones_like(x_data), x_data])
    coeffs = moindres_carres_qr(A, log_y)
    return float(np.exp(coeffs[0])), float(coeffs[1])


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_ajustement(
    x_data: np.ndarray, y_data: np.ndarray,
    degres: tuple[int, ...] = (1, 2, 4),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Trace les données et les courbes d'ajustement polynomial."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x_data, y_data, "ko", markersize=5, label="données")
    x_fine = np.linspace(x_data.min(), x_data.max(), 200)

    for d in degres:
        coeffs = ajustement_polynomial(x_data, y_data, d)
        y_fit = sum(coeffs[k] * x_fine**k for k in range(d + 1))
        res = residu_moindres_carres(
            np.column_stack([x_data**k for k in range(d + 1)]),
            coeffs, y_data,
        )
        ax.plot(x_fine, y_fit, "-", linewidth=1.5,
                label=f"degré {d} ($\\|r\\|_2 = {res:.3f}$)")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Ajustement polynomial (section 5.5.2)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def tracer_stabilite_comparison(ax: plt.Axes | None = None) -> plt.Axes:
    """
    Compare la stabilité de QR vs équations normales
    sur un système de Vandermonde mal conditionné.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ns = range(3, 18)
    err_qr, err_norm = [], []
    for n in ns:
        x = np.linspace(0, 1, 50)
        A = np.column_stack([x**k for k in range(n)])
        x_true = np.ones(n)
        b = A @ x_true

        try:
            x_qr = moindres_carres_qr(A, b)
            err_qr.append(np.linalg.norm(x_qr - x_true, np.inf))
        except Exception:
            err_qr.append(np.nan)
        try:
            x_ne = moindres_carres_normales(A, b)
            err_norm.append(np.linalg.norm(x_ne - x_true, np.inf))
        except Exception:
            err_norm.append(np.nan)

    ax.semilogy(list(ns), err_qr, "bo-", label="QR (Householder)")
    ax.semilogy(list(ns), err_norm, "rs-", label="Éq. normales ($A^T A$)")
    ax.set_xlabel("degré du polynôme")
    ax.set_ylabel("erreur $\\|x_{calc} - x_{exact}\\|_\\infty$")
    ax.set_title("Section 5.4 — QR est plus stable que les éq. normales")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


# ======================================================================
#  Démo
# ======================================================================

if __name__ == "__main__":
    print("=== QR d'une matrice 4×3 ===")
    A = np.array([[1, 1, 1], [1, 2, 4], [1, 3, 9], [1, 4, 16]], dtype=float)
    qr = decomposition_qr(A)
    print(f"R =\n{np.round(qr.R, 6)}")
    print(f"||QᵀQ - I||_∞ = {np.linalg.norm(qr.Q.T @ qr.Q - np.eye(4), np.inf):.2e}")
    print(f"||QA - R||_∞ = {np.linalg.norm(qr.Q @ A - qr.R, np.inf):.2e}")

    print("\n=== Moindres carrés : droite de régression ===")
    rng = np.random.default_rng(42)
    x_data = np.linspace(0, 5, 30)
    y_data = 2.5 * x_data + 1.0 + rng.normal(0, 0.5, 30)

    coeffs = ajustement_polynomial(x_data, y_data, 1)
    print(f"y ≈ {coeffs[0]:.4f} + {coeffs[1]:.4f} x")
    print(f"  (attendu : 1.0 + 2.5 x)")

    coeffs_np = np.polyfit(x_data, y_data, 1)
    print(f"  numpy polyfit : {coeffs_np[1]:.4f} + {coeffs_np[0]:.4f} x")

    print("\n=== Comparaison QR vs éq. normales ===")
    x = np.linspace(0, 1, 30)
    A = np.column_stack([x**k for k in range(10)])
    x_true = np.ones(10)
    b = A @ x_true
    x_qr = moindres_carres_qr(A, b)
    x_ne = moindres_carres_normales(A, b)
    print(f"Erreur QR           : {np.linalg.norm(x_qr - x_true, np.inf):.2e}")
    print(f"Erreur éq. normales : {np.linalg.norm(x_ne - x_true, np.inf):.2e}")

    print("\n=== Tracés ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_ajustement(x_data, y_data, degres=(1, 3, 8), ax=axes[0])
    tracer_stabilite_comparison(ax=axes[1])
    plt.tight_layout()
    plt.savefig("householder_qr_demo.png", dpi=120)
    print("Figure sauvegardée : householder_qr_demo.png")
