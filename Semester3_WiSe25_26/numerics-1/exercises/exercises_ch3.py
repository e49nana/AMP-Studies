"""
exercises_ch3.py — Übungen résolues du chapitre 3
==================================================

Référence : Kröger, Numerische Mathematik 1, §3.1–3.2.

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

import numpy as np


def uebung_3_1():
    """
    Übung 3.1 — Trois pas de Newton pour f(x) = x² - 3, x₀ = 4.5.
    """
    print("=== Übung 3.1 : Newton pour √3 ===")
    f = lambda x: x**2 - 3
    df = lambda x: 2*x
    x = 4.5
    x_star = np.sqrt(3)

    print(f"  x* = √3 = {x_star:.15f}")
    print(f"  {'k':>3} | {'x_k':>18} | {'f(x_k)':>14} | {'|x_k - x*|':>14}")
    print("  " + "-" * 58)
    print(f"  {0:>3} | {x:>18.15f} | {f(x):>14.6e} | {abs(x - x_star):>14.6e}")

    for k in range(1, 4):
        x = x - f(x) / df(x)
        print(f"  {k:>3} | {x:>18.15f} | {f(x):>14.6e} | {abs(x - x_star):>14.6e}")

    print(f"\n  → Le nombre de chiffres corrects double à chaque pas (α = 2).\n")


def uebung_3_6():
    """
    Übung 3.6 — Combien de pas de Newton / Sécante / Bissection
    pour obtenir 10⁻¹² de précision ?
    """
    print("=== Übung 3.6 : Nombre de pas pour 10⁻¹² ===")
    eps_target = 1e-12
    # Pour atteindre 12 chiffres de précision :

    # Newton (α=2) : on double les chiffres corrects à chaque pas
    # De ~0 chiffres à 12 : il faut k tq 2^k ≥ 12
    k_newton = int(np.ceil(np.log2(12)))
    print(f"  Newton (α=2)     : ~{k_newton} pas (doublement de chiffres)")

    # Sécante : α = φ ≈ 1.618, on gagne le facteur φ à chaque pas
    phi = (1 + np.sqrt(5)) / 2
    k_sec = int(np.ceil(np.log(12) / np.log(phi)))
    print(f"  Sécante (α≈1.618): ~{k_sec} pas")

    # Bissection : e_k = (b-a)/2^k, on veut ≤ eps
    k_bis = int(np.ceil(np.log2(1/eps_target)))
    print(f"  Bissection (α=1) : ~{k_bis} pas")
    print(f"  → Newton est ~{k_bis//k_newton}× plus efficace que la bissection.\n")


def uebung_3_9():
    """
    Übung 3.9 — Déterminer h optimal pour la dérivée numérique.
    Minimiser e(h) = erreur de troncature + erreur d'arrondi.
    """
    print("=== Übung 3.9 : h optimal pour la dérivée numérique ===")
    eps_mach = np.finfo(float).eps
    # Erreur totale ≈ h/2 · |f''| + eps_mach / h · |f|
    # Minimum à h_opt = √(2 · eps_mach · |f| / |f''|)
    # Si |f| ≈ |f''| ≈ 1 : h_opt ≈ √(2 eps_mach)

    h_opt = np.sqrt(2 * eps_mach)
    print(f"  ε_mach = {eps_mach:.3e}")
    print(f"  h_opt ≈ √(2 ε_mach) = {h_opt:.3e}")
    print()

    # Vérification sur f(x) = sin(x), x = 1
    f = np.sin
    x = 1.0
    exact = np.cos(x)
    hs = np.logspace(-16, 0, 50)
    erreurs = [abs((f(x+h) - f(x))/h - exact) for h in hs]
    i_best = np.argmin(erreurs)
    print(f"  Vérification sur sin'(1) = cos(1) :")
    print(f"    h mesuré le meilleur : {hs[i_best]:.3e}")
    print(f"    h théorique          : {h_opt:.3e}")
    print(f"    erreur minimale      : {erreurs[i_best]:.3e}\n")


def uebung_3_10():
    """
    Übung 3.10 — Un pas de Newton pour le système 2×2.

    2(x₁+x₂)² + (x₁-x₂)² - 8 = 0
    5x₁²      + (x₂-3)²   - 9 = 0

    x₀ = (2, 0)ᵀ.
    """
    print("=== Übung 3.10 : Newton pour système 2×2 ===")

    def f(x):
        return np.array([
            2*(x[0]+x[1])**2 + (x[0]-x[1])**2 - 8,
            5*x[0]**2 + (x[1]-3)**2 - 9,
        ])

    def J(x):
        return np.array([
            [4*(x[0]+x[1]) + 2*(x[0]-x[1]), 4*(x[0]+x[1]) - 2*(x[0]-x[1])],
            [10*x[0], 2*(x[1]-3)],
        ])

    x = np.array([2.0, 0.0])
    print(f"  x₀ = {x}")
    print(f"  f(x₀) = {f(x)}")
    print(f"  J(x₀) =\n    {J(x)}")

    s = np.linalg.solve(J(x), -f(x))
    x1 = x + s
    print(f"\n  s = J⁻¹(-f) = {s}")
    print(f"  x₁ = x₀ + s = {x1}")
    print(f"  f(x₁) = {f(x1)}")
    print(f"  ||f(x₁)||_∞ = {np.linalg.norm(f(x1), np.inf):.6e}")

    # Itérer jusqu'à convergence
    for k in range(2, 10):
        s = np.linalg.solve(J(x1), -f(x1))
        x1 = x1 + s
        nf = np.linalg.norm(f(x1), np.inf)
        if nf < 1e-12:
            print(f"  Convergé en {k} itérations : x* = {x1}")
            break
    print()


if __name__ == "__main__":
    uebung_3_1()
    uebung_3_6()
    uebung_3_9()
    uebung_3_10()
