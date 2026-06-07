"""
metric_spaces.py
================

Espaces métriques : distance, convergence, complétude.

Couvre :
    - Définition axiomatique : d(x,y) ≥ 0, d(x,y) = 0 ⟺ x=y, symétrie, triangle
    - Métriques classiques : euclidienne, Manhattan, Chebyshev, discrète
    - Convergence dans un espace métrique
    - Suites de Cauchy et complétude
    - Ensembles ouverts, fermés, bornés
    - Théorème du point fixe de Banach (contraction)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Métriques classiques
# ======================================================================

def d_euclidienne(x: np.ndarray, y: np.ndarray) -> float:
    """d₂(x,y) = √(Σ(xᵢ-yᵢ)²)."""
    return float(np.sqrt(np.sum((x - y)**2)))


def d_manhattan(x: np.ndarray, y: np.ndarray) -> float:
    """d₁(x,y) = Σ|xᵢ-yᵢ|."""
    return float(np.sum(np.abs(x - y)))


def d_chebyshev(x: np.ndarray, y: np.ndarray) -> float:
    """d∞(x,y) = max|xᵢ-yᵢ|."""
    return float(np.max(np.abs(x - y)))


def d_p(x: np.ndarray, y: np.ndarray, p: float) -> float:
    """dₚ(x,y) = (Σ|xᵢ-yᵢ|ᵖ)^{1/p} (Minkowski)."""
    return float(np.sum(np.abs(x - y)**p)**(1/p))


def d_discrete(x: np.ndarray, y: np.ndarray) -> float:
    """d(x,y) = 0 si x=y, 1 sinon."""
    return 0.0 if np.allclose(x, y) else 1.0


def verifier_axiomes(d: Callable, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Vérifie les 4 axiomes d'une métrique."""
    return {
        "positivité": d(x, y) >= 0,
        "séparation": (d(x, x) == 0) and (not np.allclose(x, y) or d(x, y) == 0),
        "symétrie": np.isclose(d(x, y), d(y, x)),
        "triangle": d(x, z) <= d(x, y) + d(y, z) + 1e-10,
    }


# ======================================================================
#  2. Convergence et Cauchy
# ======================================================================

def est_convergente(suite: list[np.ndarray], d: Callable, tol: float = 1e-8) -> dict:
    """Teste si la suite converge (les termes consécutifs se rapprochent)."""
    if len(suite) < 3:
        return {"convergente": False, "limite": None}
    distances = [d(suite[i], suite[i+1]) for i in range(len(suite)-1)]
    convergente = distances[-1] < tol and all(distances[i] >= distances[i+1] - 1e-12 for i in range(len(distances)-1))
    return {"convergente": convergente, "limite": suite[-1] if convergente else None,
            "d_final": distances[-1]}


def est_cauchy(suite: list[np.ndarray], d: Callable, tol: float = 1e-6) -> bool:
    """Suite de Cauchy : pour tout ε, ∃N tel que d(x_m, x_n) < ε pour m,n > N."""
    n = len(suite)
    tail = suite[n//2:]
    for i in range(len(tail)):
        for j in range(i+1, len(tail)):
            if d(tail[i], tail[j]) > tol:
                return False
    return True


# ======================================================================
#  3. Point fixe de Banach
# ======================================================================

def point_fixe_banach(
    T: Callable, x0: np.ndarray, d: Callable,
    n_max: int = 100, tol: float = 1e-12,
) -> dict:
    """
    Théorème du point fixe de Banach :
    Si T est une contraction (∃ q < 1 : d(Tx,Ty) ≤ q·d(x,y))
    dans un espace métrique complet, alors T a un unique point fixe
    et x_{n+1} = T(x_n) converge vers celui-ci.
    """
    x = np.asarray(x0, dtype=float)
    trajectoire = [x.copy()]
    for k in range(1, n_max + 1):
        x_new = T(x)
        trajectoire.append(x_new.copy())
        dist = d(x_new, x)
        if dist < tol:
            break
        x = x_new

    # Estimer le facteur de contraction q
    if len(trajectoire) >= 3:
        d1 = d(trajectoire[-1], trajectoire[-2])
        d2 = d(trajectoire[-2], trajectoire[-3])
        q_est = d1 / d2 if d2 > 1e-15 else 0
    else:
        q_est = None

    return {
        "point_fixe": trajectoire[-1],
        "iterations": k,
        "q_estime": q_est,
        "trajectoire": trajectoire,
    }


# ======================================================================
#  4. Tracés
# ======================================================================

def tracer_boules_metriques(ax: plt.Axes | None = None) -> plt.Axes:
    """Compare les boules unité B(0,1) pour différentes métriques."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    theta = np.linspace(0, 2*np.pi, 500)
    centre = np.array([0, 0])

    for p, nom, color in [(1, "$d_1$ (Manhattan)", "blue"),
                            (2, "$d_2$ (euclidienne)", "red"),
                            (4, "$d_4$", "green"),
                            (np.inf, "$d_\\infty$ (Chebyshev)", "purple")]:
        if p == np.inf:
            # Carré [-1,1]²
            ax.plot([-1,1,1,-1,-1], [-1,-1,1,1,-1], color=color, linewidth=2, label=nom)
        else:
            # |x|^p + |y|^p = 1 → y = (1 - |x|^p)^{1/p}
            x = np.linspace(-1, 1, 500)
            y_pos = np.maximum(1 - np.abs(x)**p, 0)**(1/p)
            ax.plot(x, y_pos, color=color, linewidth=2, label=nom)
            ax.plot(x, -y_pos, color=color, linewidth=2)

    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("Boules unité $B(0,1)$ pour différentes métriques")
    ax.set_aspect("equal"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


def tracer_convergence_point_fixe(ax: plt.Axes | None = None) -> plt.Axes:
    """Illustre le théorème du point fixe de Banach."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # T(x) = cos(x), point fixe ≈ 0.7391 (Dottie number)
    T = lambda x: np.array([np.cos(x[0])])
    r = point_fixe_banach(T, np.array([0.0]), d_euclidienne, n_max=50)

    traj = [t[0] for t in r["trajectoire"]]
    ax.plot(traj, "bo-", markersize=5, linewidth=1.5, label="$x_{n+1} = \\cos(x_n)$")
    ax.axhline(traj[-1], color="red", linestyle="--",
                label=f"point fixe = {traj[-1]:.6f}")

    ax.set_xlabel("itération $n$"); ax.set_ylabel("$x_n$")
    ax.set_title(f"Point fixe de Banach ($q \\approx {r['q_estime']:.3f}$)")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_contraction(ax: plt.Axes | None = None) -> plt.Axes:
    """Visualise T(x) = cos(x) comme contraction."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    x = np.linspace(-1, 2, 200)
    ax.plot(x, x, "k--", linewidth=1, label="$y = x$")
    ax.plot(x, np.cos(x), "b-", linewidth=2, label="$y = \\cos(x)$")

    # Cobweb
    xn = 0.0
    for _ in range(15):
        xn1 = np.cos(xn)
        ax.plot([xn, xn], [xn, xn1], "r-", linewidth=0.8, alpha=0.5)
        ax.plot([xn, xn1], [xn1, xn1], "r-", linewidth=0.8, alpha=0.5)
        xn = xn1

    ax.plot(xn, xn, "ro", markersize=8, label=f"point fixe ≈ {xn:.4f}")
    ax.set_xlabel("$x$"); ax.set_ylabel("$T(x)$")
    ax.set_title("Diagramme cobweb : $T(x) = \\cos(x)$")
    ax.legend(); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 2); ax.set_ylim(-0.5, 2)
    return ax


if __name__ == "__main__":
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 6.0, 3.0])
    z = np.array([0.0, 0.0, 0.0])

    print("=== Métriques classiques ===\n")
    for nom, d in [("euclidienne", d_euclidienne), ("Manhattan", d_manhattan),
                    ("Chebyshev", d_chebyshev), ("discrète", d_discrete)]:
        print(f"  {nom:12s} : d(x,y) = {d(x, y):.4f}")

    print(f"\n  Minkowski : d_p(x,y)")
    for p in [1, 2, 3, 5, 10, 50]:
        print(f"    p = {p:>2} : {d_p(x, y, p):.4f}")
    print(f"    p → ∞ : {d_chebyshev(x, y):.4f}")

    print(f"\n=== Vérification des axiomes ===\n")
    ax = verifier_axiomes(d_euclidienne, x, y, z)
    for nom, ok in ax.items():
        print(f"  {nom:12s} : {ok} ✓")

    print(f"\n=== Point fixe de Banach ===\n")
    print(f"  T(x) = cos(x)")
    T = lambda x: np.array([np.cos(x[0])])
    r = point_fixe_banach(T, np.array([0.0]), d_euclidienne)
    print(f"  x* = {r['point_fixe'][0]:.10f}")
    print(f"  cos(x*) = {np.cos(r['point_fixe'][0]):.10f} = x* ✓")
    print(f"  q ≈ {r['q_estime']:.4f} < 1 (contraction)")
    print(f"  {r['iterations']} itérations")

    print(f"\n  T(x) = x/2 + 1 (point fixe = 2)")
    T2 = lambda x: np.array([x[0]/2 + 1])
    r2 = point_fixe_banach(T2, np.array([0.0]), d_euclidienne)
    print(f"  x* = {r2['point_fixe'][0]:.10f}, q ≈ {r2['q_estime']:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_boules_metriques(ax=axes[0])
    tracer_convergence_point_fixe(ax=axes[1])
    tracer_contraction(ax=axes[2])
    plt.tight_layout()
    plt.savefig("metric_spaces_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
