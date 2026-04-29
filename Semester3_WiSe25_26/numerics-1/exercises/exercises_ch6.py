"""
exercises_ch6.py — Übungen résolues du chapitre 6
==================================================

Référence : Kröger, Numerische Mathematik 1, §6.1–6.7.

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def uebung_6_2():
    """
    Übung 6.2 — Condition absolue du problème de valeurs propres.
    """
    print("=== Übung 6.2 : Condition du problème EW ===")
    print("  Perturbation A → A + εE :")
    print("  λ(A+εE) ≈ λ(A) + ε · vᵀEv / (vᵀv)")
    print("  cond_abs ≈ ||v||₂² / |vᵀv| pour les valeurs propres.")
    print("  Pour matrices symétriques : v réels, cond_abs = 1 (bien conditionné).")
    print("  Pour matrices non-symétriques : peut être très mal conditionné.\n")

    # Exemple de Wilkinson
    A = np.array([[1, 1000], [0, 1.001]])
    print(f"  Exemple de Wilkinson : A = {A.tolist()}")
    print(f"  λ(A) = {np.linalg.eigvals(A)}")
    eps = 1e-6
    E = np.array([[0, 0], [eps, 0]])
    print(f"  λ(A + εE) = {np.linalg.eigvals(A + E)}")
    print(f"  → Perturbation de ε = {eps} change λ de O(√ε) !\n")


def uebung_6_8():
    """
    Übung 6.8 — Tester la Vektoriteration sur A = [[-1.5, 3.5], [3.5, -1.5]].
    """
    print("=== Übung 6.8 : Vektoriteration ===")
    A = np.array([[-1.5, 3.5], [3.5, -1.5]])
    eigvals = np.linalg.eigvalsh(A)
    print(f"  A = {A.tolist()}")
    print(f"  Valeurs propres exactes : {eigvals}")
    print(f"  Rate théorique |λ₂/λ₁| = {abs(eigvals[0]/eigvals[1]):.4f}")

    # Itération
    x = np.array([1.0, 3.0])
    x = x / np.linalg.norm(x)
    print(f"\n  {'k':>3} | {'µ_k':>12} | {'|µ_k - λ₁|':>14}")
    print("  " + "-" * 35)

    for k in range(10):
        y = A @ x
        i_max = np.argmax(np.abs(y))
        sign = 1.0 if x[i_max] * y[i_max] >= 0 else -1.0
        mu = sign * np.linalg.norm(y)
        x = y / (sign * np.linalg.norm(y))
        err = abs(mu - eigvals[1])  # λ₁ = -5 (dominant en module)
        print(f"  {k:>3} | {mu:>12.8f} | {err:>14.2e}")
        if err < 1e-10:
            break

    print(f"\n  Convergé vers λ = {mu:.10f} (exact : {eigvals[1] if abs(eigvals[1]) > abs(eigvals[0]) else eigvals[0]})\n")


def uebung_6_9():
    """
    Übung 6.9 — Inverse Vektoriteration sur la même matrice.
    """
    print("=== Übung 6.9 : Inverse Iteration (Wielandt) ===")
    A = np.array([[-1.5, 3.5], [3.5, -1.5]])
    eigvals = np.linalg.eigvalsh(A)

    for sigma in [-4.0, 0.0, 3.0]:
        B = A - sigma * np.eye(2)
        x = np.array([1.0, 1.0])
        x = x / np.linalg.norm(x)

        for k in range(20):
            y = np.linalg.solve(B, x)
            norm_y = np.linalg.norm(y)
            i_max = np.argmax(np.abs(y))
            sign = 1.0 if x[i_max] * y[i_max] >= 0 else -1.0
            theta = sign * norm_y
            mu = sigma + 1.0 / theta
            x = y / (sign * norm_y)
            if abs(mu - round(mu)) < 1e-10 or k > 15:
                break

        closest = eigvals[np.argmin(np.abs(eigvals - sigma))]
        print(f"  σ = {sigma:>5.1f} → λ = {mu:>12.8f} (exact: {closest:.1f}) en {k+1} it.")
    print()


def uebung_6_15():
    """
    Übung 6.15 — Cercles de Gershgorin.
    """
    print("=== Übung 6.15 : Gershgorin-Kreise ===")
    A = np.array([[2, 0.1, -0.1], [0.3, 4, -0.2], [0, 0.8, 5]], dtype=float)
    eigvals = np.linalg.eigvals(A).real

    print(f"  A = {A.tolist()}")
    print(f"  Valeurs propres : {np.sort(eigvals)}\n")

    for i in range(3):
        centre = A[i, i]
        rayon = sum(abs(A[i, j]) for j in range(3) if j != i)
        print(f"  K_{i+1} : centre = {centre:.1f}, rayon = {rayon:.1f}, "
              f"intervalle = [{centre-rayon:.1f}, {centre+rayon:.1f}]")

    # Vérification
    print("\n  Vérification :")
    for lam in np.sort(eigvals):
        dans = []
        for i in range(3):
            centre = A[i, i]
            rayon = sum(abs(A[i, j]) for j in range(3) if j != i)
            if abs(lam - centre) <= rayon + 1e-10:
                dans.append(i + 1)
        print(f"    λ = {lam:.4f} ∈ K_{dans}")

    # Tracé
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for i in range(3):
        centre = A[i, i]
        rayon = sum(abs(A[i, j]) for j in range(3) if j != i)
        circle = Circle((centre, 0), rayon, fill=True,
                         facecolor=(*plt.cm.tab10(i)[:3], 0.2),
                         edgecolor=colors[i], linewidth=2,
                         label=f"$K_{i+1}$ (c={centre:.1f}, r={rayon:.1f})")
        ax.add_patch(circle)
    ax.plot(eigvals, np.zeros_like(eigvals), "k*", markersize=15, label="λ exactes")
    ax.set_xlim(1, 6.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title("Übung 6.15 — Cercles de Gershgorin")
    plt.savefig("uebung_6_15.png", dpi=120)
    print("\n  Figure sauvegardée.\n")


if __name__ == "__main__":
    uebung_6_2()
    uebung_6_8()
    uebung_6_9()
    uebung_6_15()
