"""
exercises_ch5.py — Übungen résolues du chapitre 5
==================================================

Référence : Kröger, Numerische Mathematik 1, §5.1–5.5.

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

import numpy as np


def uebung_5_2():
    """
    Übung 5.2 — Pourquoi C = AᵀA est symétrique et définie positive ?
    """
    print("=== Übung 5.2 : AᵀA est SPD ===")
    print("  1. Symétrie : (AᵀA)ᵀ = Aᵀ(Aᵀ)ᵀ = AᵀA  ✓")
    print("  2. Définie positive (si A a rang plein) :")
    print("     xᵀ(AᵀA)x = (Ax)ᵀ(Ax) = ||Ax||₂² ≥ 0")
    print("     = 0 ssi Ax = 0 ssi x = 0 (car rang plein)  ✓")

    print("\n  Vérification numérique :")
    rng = np.random.default_rng(42)
    A = rng.standard_normal((10, 5))
    C = A.T @ A
    eigvals = np.linalg.eigvalsh(C)
    print(f"    A : {A.shape}, rang = {np.linalg.matrix_rank(A)}")
    print(f"    Eigenvalues(AᵀA) = {eigvals}")
    print(f"    Toutes > 0 : {np.all(eigvals > 0)} ✓\n")


def uebung_5_8():
    """
    Übung 5.8 — Appliquer Householder à un vecteur concret.
    """
    print("=== Übung 5.8 : Réflexion de Householder ===")
    a = np.array([3.0, 4.0])
    norm_a = np.linalg.norm(a)
    print(f"  a = {a}, ||a||₂ = {norm_a}")

    # w = a + sign(a₁)||a||₂ e₁
    sign = 1.0 if a[0] >= 0 else -1.0
    w = a.copy()
    w[0] += sign * norm_a
    print(f"  w = a + sign(a₁)||a||e₁ = {w}")

    # Q = I - 2wwᵀ/||w||²
    Q = np.eye(2) - 2 * np.outer(w, w) / np.dot(w, w)
    print(f"  Q = I - 2wwᵀ/||w||² =")
    print(f"    {Q}")
    print(f"  Qa = {Q @ a}")
    print(f"  → Qa = [-5, 0]ᵀ = -||a||₂ e₁  ✓")
    print(f"  QᵀQ = {Q.T @ Q} (orthogonal) ✓\n")


def uebung_5_10():
    """
    Übung 5.10 — Ajustement de courbe : données de croissance.
    """
    print("=== Übung 5.10 : Ajustement exponentiel ===")
    # Données fictives de croissance bactérienne
    t = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    N = np.array([100, 180, 330, 600, 1100, 2000], dtype=float)

    # Modèle : N = a·e^{bt} → ln(N) = ln(a) + bt
    ln_N = np.log(N)
    A = np.column_stack([np.ones_like(t), t])
    c = np.linalg.lstsq(A, ln_N, rcond=None)[0]
    a, b = np.exp(c[0]), c[1]

    print(f"  Données : t = {t.tolist()}, N = {N.tolist()}")
    print(f"  Modèle : N = a·exp(bt)")
    print(f"  Linéarisation : ln(N) = ln(a) + bt")
    print(f"  Résultat : a = {a:.2f}, b = {b:.4f}")
    print(f"  → N(t) ≈ {a:.1f}·exp({b:.3f}t)")
    print(f"  → Temps de doublement = ln(2)/b = {np.log(2)/b:.2f} heures")

    # R²
    N_pred = a * np.exp(b * t)
    ss_res = np.sum((N - N_pred)**2)
    ss_tot = np.sum((N - np.mean(N))**2)
    r2 = 1 - ss_res / ss_tot
    print(f"  R² = {r2:.6f}\n")


if __name__ == "__main__":
    uebung_5_2()
    uebung_5_8()
    uebung_5_10()
