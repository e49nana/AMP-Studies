"""
splines.py
==========

Interpolation par splines cubiques naturelles.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", section 4.3.

Couvre :
    - Splines cubiques naturelles (S_{3,2}, Définition 4.12)
    - Construction du système tridiagonal pour les moments M_i
    - Évaluation par morceaux
    - Comparaison avec interpolation polynomiale (phénomène de Runge)
    - Comparaison from-scratch vs scipy.interpolate.CubicSpline

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Spline cubique naturelle
# ======================================================================

class SplineCubique:
    """
    Spline cubique naturelle (S_{3,2} avec M_0 = M_n = 0).

    Construction :
        1. Calculer les pas h_i = x_{i+1} - x_i
        2. Résoudre le système tridiagonal pour les moments M_i
           (conditions C² + bords naturels M_0 = M_n = 0)
        3. Sur chaque intervalle [x_i, x_{i+1}], le polynôme est :
           S_i(x) = M_i (x_{i+1}-x)³ / (6h_i)
                   + M_{i+1} (x-x_i)³ / (6h_i)
                   + (y_i/h_i - M_i h_i/6)(x_{i+1}-x)
                   + (y_{i+1}/h_i - M_{i+1} h_i/6)(x-x_i)
    """

    def __init__(self, xs: np.ndarray, ys: np.ndarray) -> None:
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        n = len(xs) - 1  # nombre d'intervalles
        if n < 2:
            raise ValueError("Il faut au moins 3 points.")

        self.xs = xs
        self.ys = ys
        self.n = n

        # Pas
        h = np.diff(xs)
        self.h = h

        # Système tridiagonal pour les moments M_1, ..., M_{n-1}
        # (M_0 = M_n = 0 pour spline naturelle)
        # Équation i (pour i = 1, ..., n-1) :
        #   h_{i-1} M_{i-1} + 2(h_{i-1}+h_i) M_i + h_i M_{i+1}
        #     = 6 * ((y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1})

        m = n - 1  # nombre d'inconnues
        if m == 0:
            self.M = np.zeros(n + 1)
            return

        # Diagonales du système tridiagonal
        diag_main = np.array([2 * (h[i] + h[i + 1]) for i in range(m)])
        diag_lower = np.array([h[i + 1] for i in range(m - 1)])
        diag_upper = np.array([h[i + 1] for i in range(m - 1)])
        rhs = np.array([
            6 * ((ys[i + 2] - ys[i + 1]) / h[i + 1] - (ys[i + 1] - ys[i]) / h[i])
            for i in range(m)
        ])

        # Résolution par Thomas (tridiagonal solver from-scratch)
        M_inner = self._thomas(diag_lower, diag_main, diag_upper, rhs)
        self.M = np.zeros(n + 1)
        self.M[1:n] = M_inner

    @staticmethod
    def _thomas(
        a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
    ) -> np.ndarray:
        """
        Algorithme de Thomas pour systèmes tridiagonaux.
        a : sous-diagonale (taille n-1)
        b : diagonale (taille n)
        c : sur-diagonale (taille n-1)
        d : second membre (taille n)
        """
        n = len(b)
        c_ = np.zeros(n)
        d_ = np.zeros(n)
        c_[0] = c[0] / b[0]
        d_[0] = d[0] / b[0]
        for i in range(1, n):
            denom = b[i] - a[i - 1] * c_[i - 1]
            if i < n - 1:
                c_[i] = c[i] / denom
            d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / denom
        x = np.zeros(n)
        x[-1] = d_[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_[i] - c_[i] * x[i + 1]
        return x

    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        """Évalue le spline en x (scalaire ou vecteur)."""
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        y = np.empty_like(x)

        for k, xk in enumerate(x):
            # Trouver l'intervalle [x_i, x_{i+1}]
            i = int(np.searchsorted(self.xs, xk, side="right")) - 1
            i = max(0, min(i, self.n - 1))

            hi = self.h[i]
            t1 = self.xs[i + 1] - xk
            t2 = xk - self.xs[i]

            y[k] = (
                self.M[i] * t1**3 / (6 * hi)
                + self.M[i + 1] * t2**3 / (6 * hi)
                + (self.ys[i] / hi - self.M[i] * hi / 6) * t1
                + (self.ys[i + 1] / hi - self.M[i + 1] * hi / 6) * t2
            )

        return float(y[0]) if scalar else y


# ======================================================================
#  2. Comparaison avec interpolation polynomiale
# ======================================================================

def runge(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + 25.0 * x**2)


def tracer_spline_vs_poly(
    n: int = 15,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Comparaison spline vs polynôme de Lagrange sur la fonction de Runge."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    a, b = -1, 1
    xs = np.linspace(a, b, n + 1)
    ys = runge(xs)
    x_fine = np.linspace(a, b, 500)

    # Spline
    spl = SplineCubique(xs, ys)
    y_spl = spl(x_fine)

    # Polynôme (Newton)
    from runge_phenomenon import interpolation_newton
    y_poly = interpolation_newton(xs, ys, x_fine)

    ax.plot(x_fine, runge(x_fine), "k-", linewidth=2.5, label="$f(x)$")
    ax.plot(x_fine, y_spl, "b-", linewidth=2, label=f"Spline cubique ($n={n}$)")
    ax.plot(x_fine, y_poly, "r--", linewidth=1.5, label=f"Polynôme deg. {n}")
    ax.plot(xs, ys, "ko", markersize=5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Section 4.3 — Spline vs polynôme sur la fonction de Runge")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def tracer_erreur_spline(ax: plt.Axes | None = None) -> plt.Axes:
    """Erreur max du spline vs polynôme en fonction de n."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x_fine = np.linspace(-1, 1, 1000)
    f_fine = runge(x_fine)
    ns = range(4, 50)
    err_spl, err_poly = [], []

    from runge_phenomenon import interpolation_newton

    for n in ns:
        xs = np.linspace(-1, 1, n + 1)
        ys = runge(xs)
        spl = SplineCubique(xs, ys)
        err_spl.append(np.max(np.abs(spl(x_fine) - f_fine)))
        p = interpolation_newton(xs, ys, x_fine)
        err_poly.append(np.max(np.abs(p - f_fine)))

    ax.semilogy(list(ns), err_spl, "bo-", markersize=3, label="Spline cubique")
    ax.semilogy(list(ns), err_poly, "rs-", markersize=3, label="Polynôme (éq.)")
    ax.set_xlabel("$n$ (nombre d'intervalles)")
    ax.set_ylabel("erreur max $\\|S - f\\|_\\infty$")
    ax.set_title("Spline converge, polynôme diverge")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


# ======================================================================
#  Démo
# ======================================================================

if __name__ == "__main__":
    print("=== Spline cubique naturelle : 5 points ===")
    xs = np.array([0, 1, 2, 3, 4], dtype=float)
    ys = np.array([0, 1, 0, 1, 0], dtype=float)
    spl = SplineCubique(xs, ys)
    print(f"  Moments M = {spl.M}")
    print(f"  S(0.5) = {spl(0.5):.6f}")
    print(f"  S(2.5) = {spl(2.5):.6f}")

    # Comparaison avec scipy
    from scipy.interpolate import CubicSpline
    spl_scipy = CubicSpline(xs, ys, bc_type="natural")
    print(f"\n  scipy S(0.5) = {spl_scipy(0.5):.6f}")
    print(f"  scipy S(2.5) = {spl_scipy(2.5):.6f}")
    x_test = np.linspace(0, 4, 50)
    err = np.max(np.abs(spl(x_test) - spl_scipy(x_test)))
    print(f"  ||mine - scipy||_∞ = {err:.2e}")

    print("\n=== Tracés ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_spline_vs_poly(n=15, ax=axes[0])
    tracer_erreur_spline(ax=axes[1])
    plt.tight_layout()
    plt.savefig("splines_demo.png", dpi=120)
    print("Figure sauvegardée : splines_demo.png")
