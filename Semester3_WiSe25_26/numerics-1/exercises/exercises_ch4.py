"""
exercises_ch4.py — Übungen résolues du chapitre 4
==================================================

Référence : Kröger, Numerische Mathematik 1, §4.1–4.3.

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

import numpy as np
import matplotlib.pyplot as plt


def uebung_4_5():
    """
    Übung 4.5 — Interpolationspolynom par Lagrange et Newton.

    Knoten x = [1, 2, 4], Werte y = [6, 6, 0].
    """
    print("=== Übung 4.5 : Interpolation 3 points ===")
    xs = np.array([1, 2, 4], dtype=float)
    ys = np.array([6, 6, 0], dtype=float)

    # Lagrange
    print("  Lagrange :")
    print(f"    L₀(x) = (x-2)(x-4)/((1-2)(1-4)) = (x-2)(x-4)/3")
    print(f"    L₁(x) = (x-1)(x-4)/((2-1)(2-4)) = (x-1)(x-4)/(-2)")
    print(f"    L₂(x) = (x-1)(x-2)/((4-1)(4-2)) = (x-1)(x-2)/6")
    print(f"    p(x) = 6·L₀ + 6·L₁ + 0·L₂ = -x² + 3x + 4")

    # Newton (différences divisées)
    print("\n  Newton (diff. divisées) :")
    c0 = ys[0]
    c1 = (ys[1] - ys[0]) / (xs[1] - xs[0])
    c2 = ((ys[2]-ys[1])/(xs[2]-xs[1]) - c1) / (xs[2] - xs[0])
    print(f"    c₀ = f[x₀] = {c0}")
    print(f"    c₁ = f[x₀,x₁] = ({ys[1]}-{ys[0]})/({xs[1]}-{xs[0]}) = {c1}")
    print(f"    c₂ = f[x₀,x₁,x₂] = {c2}")
    print(f"    p(x) = {c0} + {c1}(x-{xs[0]}) + ({c2})(x-{xs[0]})(x-{xs[1]})")
    print(f"         = -x² + 3x + 4  ✓")

    # Vérification
    p = lambda x: -x**2 + 3*x + 4
    print(f"\n  Vérification : p(1)={p(1)}, p(2)={p(2)}, p(4)={p(4)}")
    print(f"  p(3) = {p(3)} (prédiction)\n")


def uebung_4_7():
    """
    Übung 4.7 — Neville-Aitken pour p(2).

    x = [0, 1, 3, 4], y = [12, 3, -3, 12].
    """
    print("=== Übung 4.7 : Neville-Aitken ===")
    xs = np.array([0, 1, 3, 4], dtype=float)
    ys = np.array([12, 3, -3, 12], dtype=float)
    x = 2.0

    n = len(xs)
    T = np.zeros((n, n))
    T[:, 0] = ys

    print(f"  Tableau de Neville pour x = {x} :")
    print(f"  {'j':>3} | {'x_j':>5} | {'k=0':>8} | {'k=1':>8} | {'k=2':>8} | {'k=3':>8}")
    print("  " + "-" * 55)

    for k in range(1, n):
        for j in range(n - k):
            T[j, k] = ((xs[j+k] - x)*T[j, k-1] + (x - xs[j])*T[j+1, k-1]) / (xs[j+k] - xs[j])

    for j in range(n):
        row = f"  {j:>3} | {xs[j]:>5.0f} |"
        for k in range(n - j):
            row += f" {T[j,k]:>8.2f} |"
        print(row)

    print(f"\n  p(2) = T[0,3] = {T[0, n-1]:.2f}\n")


def uebung_4_9():
    """
    Übung 4.9 — Différences divisées pour les données de l'Übung 4.7.
    """
    print("=== Übung 4.9 : Différences divisées (données Übung 4.7) ===")
    xs = np.array([0, 1, 3, 4], dtype=float)
    ys = np.array([12, 3, -3, 12], dtype=float)

    n = len(xs)
    T = np.zeros((n, n))
    T[:, 0] = ys

    for k in range(1, n):
        for j in range(n - k):
            T[j, k] = (T[j+1, k-1] - T[j, k-1]) / (xs[j+k] - xs[j])

    print(f"  {'j':>3} | {'x_j':>5} | {'[x_j]':>8} | {'[x_j,.]':>8} | {'[x_j,..]':>8} | {'[x_j,...]':>8}")
    print("  " + "-" * 60)
    for j in range(n):
        row = f"  {j:>3} | {xs[j]:>5.0f} |"
        for k in range(n - j):
            row += f" {T[j,k]:>8.2f} |"
        print(row)

    c = T[0, :]
    print(f"\n  Coefficients de Newton : c = {c}")
    print(f"  p(x) = {c[0]} + ({c[1]})(x-{xs[0]}) + ({c[2]})(x-{xs[0]})(x-{xs[1]}) + ({c[3]:.4f})(x-{xs[0]})(x-{xs[1]})(x-{xs[2]})")

    # Vérification
    def p(x):
        return c[0] + c[1]*(x-xs[0]) + c[2]*(x-xs[0])*(x-xs[1]) + c[3]*(x-xs[0])*(x-xs[1])*(x-xs[2])
    print(f"  p(2) = {p(2):.2f} (doit être = -4, cf. Übung 4.7) ✓\n")


if __name__ == "__main__":
    uebung_4_5()
    uebung_4_7()
    uebung_4_9()
