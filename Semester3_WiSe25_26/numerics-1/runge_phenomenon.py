"""
runge_phenomenon.py
===================

Interpolation polynomiale classique et phénomène de Runge.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", chapitre 4.

Couvre :
    - Lagrange (formule 4.2 / 4.3)
    - Newton — différences divisées (Satz 4.8, formule 4.5)
    - Neville-Aitken (Lemma 4.6)
    - Analyse d'erreur (Satz 4.10) : rôle du polynôme nodal Π(x-x_i)
    - Phénomène de Runge (nœuds équidistants → oscillations divergentes)
    - Nœuds de Tchebychev (section 4.2.6, minimisent le polynôme nodal)
    - Comparaison from-scratch ↔ numpy / scipy

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Lagrange (formule 4.2 / 4.3)
# ======================================================================

def base_lagrange(xs: np.ndarray, i: int, x: float) -> float:
    """
    Évalue le i-ème polynôme de base de Lagrange L_i^n(x).

    Formule 4.2 : L_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j).
    """
    n = len(xs)
    val = 1.0
    for j in range(n):
        if j != i:
            val *= (x - xs[j]) / (xs[i] - xs[j])
    return val


def interpolation_lagrange(
    xs: np.ndarray, ys: np.ndarray, x_eval: np.ndarray,
) -> np.ndarray:
    """
    Évalue le polynôme d'interpolation de Lagrange (formule 4.3) :
        p(x) = Σ y_i L_i^n(x).

    Coût : O(n²) par point d'évaluation.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    out = np.zeros_like(x_eval)
    for i in range(len(xs)):
        for k in range(len(x_eval)):
            out[k] += ys[i] * base_lagrange(xs, i, x_eval[k])
    return out


# ======================================================================
#  2. Différences divisées & Newton (Satz 4.8)
# ======================================================================

def differences_divisees(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Calcule le tableau des différences divisées (formule 4.5) :
        c_{jj} = y_j,
        c_{jk} = (c_{j+1,k} - c_{j,k-1}) / (x_k - x_j).

    Renvoie le vecteur [c_{0,0}, c_{0,1}, ..., c_{0,n}] = coefficients de Newton.
    """
    n = len(xs)
    # Tableau triangulaire, colonne par colonne
    c = np.array(ys, dtype=float)
    for j in range(1, n):
        # Mise à jour in-place de droite à gauche
        c[j:] = (c[j:] - c[j - 1:-1]) / (xs[j:] - xs[:n - j])
    return c


def interpolation_newton(
    xs: np.ndarray, ys: np.ndarray, x_eval: np.ndarray,
) -> np.ndarray:
    """
    Évalue le polynôme de Newton par schéma de Horner (formule 4.6) :
        p(x) = c_0 + (x-x_0)(c_1 + (x-x_1)(c_2 + ...)).

    Coût : O(n) par point (après O(n²) pour les diff. divisées).
    """
    xs = np.asarray(xs, dtype=float)
    c = differences_divisees(xs, ys)
    x_eval = np.asarray(x_eval, dtype=float)
    n = len(c)
    # Horner de droite à gauche
    out = np.full_like(x_eval, c[-1], dtype=float)
    for k in range(n - 2, -1, -1):
        out = out * (x_eval - xs[k]) + c[k]
    return out


# ======================================================================
#  3. Neville-Aitken (Lemma 4.6)
# ======================================================================

def neville_aitken(
    xs: np.ndarray, ys: np.ndarray, x: float,
) -> float:
    """
    Évalue p(x) par le schéma de Neville-Aitken (Lemma 4.6) :
        p_{jk}(x) = (p_{j,k-1}(x)(x_k - x) + p_{j+1,k}(x)(x - x_j)) / (x_k - x_j).

    Renvoie la valeur du polynôme d'interpolation en un seul point x.
    Coût : O(n²) mais sans construire les coefficients.
    """
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    p = np.array(ys, dtype=float)  # p[i] = p_{ii}(x) au départ

    for j in range(1, n):
        for i in range(n - j):
            p[i] = ((xs[i + j] - x) * p[i] + (x - xs[i]) * p[i + 1]) / (
                xs[i + j] - xs[i]
            )
    return float(p[0])


# ======================================================================
#  4. Polynôme nodal et borne d'erreur (Satz 4.10)
# ======================================================================

def polynome_nodal(xs: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """
    Polynôme nodal ω(x) = Π_{i=0}^n (x - x_i).

    Apparaît dans la borne d'erreur (Satz 4.10) :
        |p(x) - f(x)| ≤ |ω(x)| · max|f^{(n+1)}| / (n+1)!
    """
    x_eval = np.asarray(x_eval, dtype=float)
    out = np.ones_like(x_eval)
    for xi in xs:
        out *= (x_eval - xi)
    return out


# ======================================================================
#  5. Nœuds de Tchebychev (section 4.2.6)
# ======================================================================

def noeuds_equidistants(a: float, b: float, n: int) -> np.ndarray:
    """n+1 nœuds équidistants sur [a, b]."""
    return np.linspace(a, b, n + 1)


def noeuds_tchebychev(a: float, b: float, n: int) -> np.ndarray:
    """
    n+1 nœuds de Tchebychev sur [a, b] :
        x_k = (a+b)/2 + (b-a)/2 · cos((2k+1)π / (2(n+1))),  k = 0, ..., n.

    Minimisent ||ω||_∞ sur [a, b] parmi tous les choix de n+1 nœuds.
    Le max de |ω| sur [a, b] vaut (b-a)^{n+1} / 2^{2n+1}.
    """
    k = np.arange(n + 1)
    x_ref = np.cos((2 * k + 1) * np.pi / (2 * (n + 1)))
    return (a + b) / 2 + (b - a) / 2 * x_ref


# ======================================================================
#  6. Fonction de Runge
# ======================================================================

def runge(x: float | np.ndarray) -> float | np.ndarray:
    """
    Fonction de Runge : f(x) = 1 / (1 + 25 x²).

    Exemple canonique : l'interpolation de Lagrange sur nœuds
    équidistants diverge quand n augmente (oscillations aux bords).
    """
    return 1.0 / (1.0 + 25.0 * np.asarray(x) ** 2)


# ======================================================================
#  7. Tracés
# ======================================================================

def tracer_runge_equidistant(
    degres: tuple[int, ...] = (5, 10, 15, 20),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Trace la fonction de Runge et ses interpolations sur nœuds
    équidistants pour différents degrés — montre les oscillations.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    a, b = -1, 1
    x_fine = np.linspace(a, b, 500)
    ax.plot(x_fine, runge(x_fine), "k-", linewidth=2.5, label="$f(x) = 1/(1+25x^2)$")

    for n in degres:
        xs = noeuds_equidistants(a, b, n)
        ys = runge(xs)
        p = interpolation_newton(xs, ys, x_fine)
        ax.plot(x_fine, p, "--", linewidth=1.5, label=f"$n={n}$ (équidistant)")

    ax.set_ylim(-1.5, 2.0)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p(x)$")
    ax.set_title("Phénomène de Runge — nœuds équidistants")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def tracer_runge_tchebychev(
    degres: tuple[int, ...] = (5, 10, 15, 20),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Même chose mais avec nœuds de Tchebychev — pas d'oscillation.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    a, b = -1, 1
    x_fine = np.linspace(a, b, 500)
    ax.plot(x_fine, runge(x_fine), "k-", linewidth=2.5, label="$f(x) = 1/(1+25x^2)$")

    for n in degres:
        xs = noeuds_tchebychev(a, b, n)
        ys = runge(xs)
        p = interpolation_newton(xs, ys, x_fine)
        ax.plot(x_fine, p, "-", linewidth=1.5, label=f"$n={n}$ (Tchebychev)")

    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p(x)$")
    ax.set_title("Nœuds de Tchebychev — plus d'oscillation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def tracer_polynome_nodal(
    n: int = 10,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Compare |ω(x)| = |Π(x - x_i)| pour nœuds équidistants vs Tchebychev.

    C'est le facteur clé dans la borne d'erreur (Satz 4.10).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    a, b = -1, 1
    x_fine = np.linspace(a, b, 500)

    xs_eq = noeuds_equidistants(a, b, n)
    xs_ch = noeuds_tchebychev(a, b, n)

    omega_eq = np.abs(polynome_nodal(xs_eq, x_fine))
    omega_ch = np.abs(polynome_nodal(xs_ch, x_fine))

    ax.semilogy(x_fine, omega_eq, "r-", linewidth=2, label=f"équidistant ($n={n}$)")
    ax.semilogy(x_fine, omega_ch, "b-", linewidth=2, label=f"Tchebychev ($n={n}$)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|\\omega(x)|$")
    ax.set_title("Satz 4.10 — polynôme nodal $|\\omega(x)| = |\\prod(x - x_i)|$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_erreur_vs_degre(ax: plt.Axes | None = None) -> plt.Axes:
    """
    Erreur max de l'interpolation de Runge en fonction du degré,
    nœuds équidistants vs Tchebychev.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    a, b = -1, 1
    x_fine = np.linspace(a, b, 1000)
    f_fine = runge(x_fine)
    degres = range(2, 35)

    err_eq, err_ch = [], []
    for n in degres:
        xs_eq = noeuds_equidistants(a, b, n)
        xs_ch = noeuds_tchebychev(a, b, n)
        p_eq = interpolation_newton(xs_eq, runge(xs_eq), x_fine)
        p_ch = interpolation_newton(xs_ch, runge(xs_ch), x_fine)
        err_eq.append(np.max(np.abs(p_eq - f_fine)))
        err_ch.append(np.max(np.abs(p_ch - f_fine)))

    ax.semilogy(list(degres), err_eq, "rs-", markersize=4, label="équidistant")
    ax.semilogy(list(degres), err_ch, "bo-", markersize=4, label="Tchebychev")
    ax.set_xlabel("degré $n$")
    ax.set_ylabel("$\\max_{x \\in [-1,1]} |p(x) - f(x)|$")
    ax.set_title("Erreur maximale — Runge")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


# ======================================================================
#  Démo
# ======================================================================

if __name__ == "__main__":
    # -- Übung 4.5 --
    print("=== Übung 4.5 : interpolation 3 points ===")
    xs = np.array([1, 2, 4], dtype=float)
    ys = np.array([6, 6, 0], dtype=float)
    c = differences_divisees(xs, ys)
    print(f"Différences divisées : {c}")
    test_x = np.array([1, 2, 3, 4])
    print(f"p({test_x}) = {interpolation_newton(xs, ys, test_x)}")

    # -- Übung 4.7 (Neville-Aitken) --
    print("\n=== Übung 4.7 : Neville-Aitken en x=2 ===")
    xs2 = np.array([0, 1, 3, 4], dtype=float)
    ys2 = np.array([12, 3, -3, 12], dtype=float)
    print(f"p(2) = {neville_aitken(xs2, ys2, 2.0)}")

    # -- Comparaison from-scratch vs NumPy --
    print("\n=== Comparaison avec numpy.polynomial ===")
    xs = np.array([0, 1, 2, 3, 4], dtype=float)
    ys = np.array([1, 0, 1, 0, 1], dtype=float)
    x_test = np.linspace(0, 4, 20)
    p_mine = interpolation_newton(xs, ys, x_test)
    coeffs = np.polyfit(xs, ys, len(xs) - 1)
    p_numpy = np.polyval(coeffs, x_test)
    print(f"||mine - numpy||_∞ = {np.max(np.abs(p_mine - p_numpy)):.2e}")

    # -- Tracés --
    print("\n=== Tracés ===")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    tracer_runge_equidistant(ax=axes[0, 0])
    tracer_runge_tchebychev(ax=axes[0, 1])
    tracer_polynome_nodal(n=15, ax=axes[1, 0])
    tracer_erreur_vs_degre(ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig("runge_phenomenon_demo.png", dpi=120)
    print("Figure sauvegardée : runge_phenomenon_demo.png")
