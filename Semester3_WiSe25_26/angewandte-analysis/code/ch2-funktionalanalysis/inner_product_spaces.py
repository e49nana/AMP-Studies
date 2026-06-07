"""
inner_product_spaces.py
=======================

Espaces à produit scalaire et espaces de Hilbert.

Couvre :
    - Produit scalaire : bilinéarité, symétrie, positivité
    - Inégalité de Cauchy-Schwarz : |⟨x,y⟩| ≤ ||x||·||y||
    - Projection orthogonale sur un sous-espace
    - Gram-Schmidt : orthonormalisation
    - Meilleure approximation dans un sous-espace (projection)
    - Bases orthonormées et identité de Parseval

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Produit scalaire
# ======================================================================

def produit_scalaire(x: np.ndarray, y: np.ndarray) -> float:
    """⟨x, y⟩ = Σ xᵢ yᵢ (produit scalaire standard dans R^n)."""
    return float(np.dot(x, y))


def norme_from_ps(x: np.ndarray) -> float:
    """||x|| = √⟨x, x⟩."""
    return np.sqrt(produit_scalaire(x, x))


def angle_entre(x: np.ndarray, y: np.ndarray) -> float:
    """θ = arccos(⟨x,y⟩ / (||x||·||y||))."""
    cos_th = produit_scalaire(x, y) / (norme_from_ps(x) * norme_from_ps(y))
    return np.arccos(np.clip(cos_th, -1, 1))


def cauchy_schwarz(x: np.ndarray, y: np.ndarray) -> dict:
    """Vérifie |⟨x,y⟩| ≤ ||x||·||y||."""
    lhs = abs(produit_scalaire(x, y))
    rhs = norme_from_ps(x) * norme_from_ps(y)
    return {"lhs": lhs, "rhs": rhs, "satisfait": lhs <= rhs + 1e-10}


# ======================================================================
#  2. Projection orthogonale
# ======================================================================

def projection_vecteur(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """proj_v(x) = ⟨x,v⟩/⟨v,v⟩ · v."""
    return produit_scalaire(x, v) / produit_scalaire(v, v) * v


def projection_sous_espace(x: np.ndarray, base_on: list[np.ndarray]) -> np.ndarray:
    """Projection de x sur le sous-espace engendré par une base orthonormée."""
    p = np.zeros_like(x, dtype=float)
    for e in base_on:
        p += produit_scalaire(x, e) * e
    return p


def erreur_approximation(x: np.ndarray, p: np.ndarray) -> float:
    """||x - proj|| = distance au sous-espace."""
    return norme_from_ps(x - p)


# ======================================================================
#  3. Gram-Schmidt
# ======================================================================

def gram_schmidt(vecteurs: list[np.ndarray]) -> list[np.ndarray]:
    """
    Orthonormalisation de Gram-Schmidt.
    Entrée : famille libre. Sortie : base orthonormée.
    """
    base_on = []
    for v in vecteurs:
        w = v.copy().astype(float)
        for e in base_on:
            w -= produit_scalaire(w, e) * e
        norm_w = norme_from_ps(w)
        if norm_w > 1e-12:
            base_on.append(w / norm_w)
    return base_on


def verifier_orthonormalite(base: list[np.ndarray]) -> dict:
    """Vérifie ⟨eᵢ, eⱼ⟩ = δᵢⱼ."""
    n = len(base)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = produit_scalaire(base[i], base[j])
    return {
        "matrice_Gram": G,
        "orthonormale": np.allclose(G, np.eye(n), atol=1e-10),
    }


# ======================================================================
#  4. Parseval
# ======================================================================

def parseval_check(x: np.ndarray, base_on: list[np.ndarray]) -> dict:
    """
    Identité de Parseval : ||x||² = Σ |⟨x, eᵢ⟩|² (si base ON complète).
    """
    norm_sq = norme_from_ps(x)**2
    coeff_sq = sum(produit_scalaire(x, e)**2 for e in base_on)
    return {"||x||²": norm_sq, "Σ|⟨x,eᵢ⟩|²": coeff_sq,
            "egalite": np.isclose(norm_sq, coeff_sq)}


# ======================================================================
#  5. Produit scalaire pour fonctions (L²)
# ======================================================================

def ps_L2(f, g, a: float = 0, b: float = 1, n: int = 1000) -> float:
    """⟨f, g⟩_{L²} = ∫_a^b f(x)g(x) dx (approx. par trapèzes)."""
    x = np.linspace(a, b, n)
    y = f(x) * g(x)
    return float(np.trapezoid(y, x))


def norme_L2(f, a: float = 0, b: float = 1) -> float:
    """||f||_{L²} = √⟨f, f⟩."""
    return np.sqrt(ps_L2(f, f, a, b))


# ======================================================================
#  6. Tracés
# ======================================================================

def tracer_projection(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    x = np.array([3, 4])
    v = np.array([1, 0.5])
    p = projection_vecteur(x, v)
    r = x - p

    ax.quiver(0, 0, x[0], x[1], angles="xy", scale_units="xy", scale=1,
              color="blue", width=0.015, label=f"$x = {x}$")
    ax.quiver(0, 0, v[0]*5, v[1]*5, angles="xy", scale_units="xy", scale=1,
              color="grey", width=0.008, alpha=0.3, label="sous-espace $V$")
    ax.quiver(0, 0, p[0], p[1], angles="xy", scale_units="xy", scale=1,
              color="green", width=0.015, label=f"proj = ({p[0]:.2f}, {p[1]:.2f})")
    ax.quiver(p[0], p[1], r[0], r[1], angles="xy", scale_units="xy", scale=1,
              color="red", width=0.012, label=f"résidu ⊥ V")

    ax.set_xlim(-1, 5); ax.set_ylim(-1, 5)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title("Projection orthogonale")
    ax.legend(fontsize=9)
    return ax


def tracer_gram_schmidt(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    v1 = np.array([2, 1])
    v2 = np.array([1, 3])
    base_on = gram_schmidt([v1, v2])
    e1, e2 = base_on

    # Vecteurs originaux
    for v, nom, c in [(v1, "$v_1$", "blue"), (v2, "$v_2$", "red")]:
        ax.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1,
                  color=c, width=0.012, alpha=0.4, label=f"{nom} = {v}")

    # Base ON
    for e, nom, c in [(e1, "$e_1$", "green"), (e2, "$e_2$", "orange")]:
        ax.quiver(0, 0, e[0]*2, e[1]*2, angles="xy", scale_units="xy", scale=1,
                  color=c, width=0.015, label=f"{nom} = ({e[0]:.3f}, {e[1]:.3f})")

    ax.set_xlim(-1, 4); ax.set_ylim(-1, 4)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title("Gram-Schmidt : $\\{v_1, v_2\\} \\to \\{e_1, e_2\\}$")
    ax.legend(fontsize=8)
    return ax


def tracer_meilleure_approx(ax: plt.Axes | None = None) -> plt.Axes:
    """Meilleure approximation polynomiale de sin(x) dans L²."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(0, np.pi, 300)

    # Base orthonormée de polynômes (Legendre-like sur [0,π])
    monomes = [lambda x: np.ones_like(x), lambda x: x, lambda x: x**2, lambda x: x**3]
    base_on = gram_schmidt([np.array([m(xi) for xi in np.linspace(0, np.pi, 500)]) for m in monomes])

    f_vals = np.sin(np.linspace(0, np.pi, 500))
    for deg in [1, 2, 3]:
        coeffs = [produit_scalaire(f_vals, base_on[k]) for k in range(deg+1)]
        approx = sum(c * base_on[k] for k, c in enumerate(coeffs))
        x_approx = np.linspace(0, np.pi, 500)
        err = norme_from_ps(f_vals - approx) / norme_from_ps(f_vals)
        ax.plot(x_approx, approx, "--", linewidth=1.5, label=f"deg {deg} (err = {err:.4f})")

    ax.plot(x, np.sin(x), "k-", linewidth=2.5, label="$\\sin(x)$")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("Meilleure approximation polynomiale dans $L^2[0, \\pi]$")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Produit scalaire ===\n")
    x = np.array([1, 2, 3], dtype=float)
    y = np.array([4, -5, 6], dtype=float)
    print(f"  ⟨x, y⟩ = {produit_scalaire(x, y)}")
    print(f"  ||x|| = {norme_from_ps(x):.4f}")
    print(f"  angle = {np.degrees(angle_entre(x, y)):.2f}°")
    cs = cauchy_schwarz(x, y)
    print(f"  Cauchy-Schwarz : |⟨x,y⟩| = {cs['lhs']:.4f} ≤ ||x||·||y|| = {cs['rhs']:.4f} ✓")

    print(f"\n=== Gram-Schmidt ===\n")
    v1 = np.array([1, 1, 0], dtype=float)
    v2 = np.array([1, 0, 1], dtype=float)
    v3 = np.array([0, 1, 1], dtype=float)
    base_on = gram_schmidt([v1, v2, v3])
    for i, e in enumerate(base_on):
        print(f"  e_{i+1} = {np.round(e, 6)}")
    check = verifier_orthonormalite(base_on)
    print(f"  Orthonormale ? {check['orthonormale']} ✓")

    print(f"\n=== Projection ===\n")
    x = np.array([1, 2, 3], dtype=float)
    p = projection_sous_espace(x, base_on[:2])
    print(f"  x = {x}")
    print(f"  proj sur span(e₁,e₂) = {np.round(p, 4)}")
    print(f"  ||x - proj|| = {erreur_approximation(x, p):.4f}")

    print(f"\n=== Parseval ===\n")
    base_R3 = gram_schmidt([np.array([1,0,0.]), np.array([0,1,0.]), np.array([0,0,1.])])
    pc = parseval_check(x, base_R3)
    print(f"  ||x||² = {pc['||x||²']:.4f}")
    print(f"  Σ|⟨x,eᵢ⟩|² = {pc['Σ|⟨x,eᵢ⟩|²']:.4f}")
    print(f"  Parseval ✓ ? {pc['egalite']}")

    print(f"\n=== L² : produit scalaire de fonctions ===\n")
    print(f"  ⟨sin, cos⟩ sur [0, 2π] = {ps_L2(np.sin, np.cos, 0, 2*np.pi):.6f} ≈ 0 (orthogonaux ✓)")
    print(f"  ||sin||_{'{L²[0,π]}'} = {norme_L2(np.sin, 0, np.pi):.6f} (exact: √(π/2) = {np.sqrt(np.pi/2):.6f})")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_projection(ax=axes[0])
    tracer_gram_schmidt(ax=axes[1])
    tracer_meilleure_approx(ax=axes[2])
    plt.tight_layout()
    plt.savefig("inner_product_spaces_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
