"""
svd_intro.py
============

Introduction à la décomposition en valeurs singulières (SVD).

Couvre :
    - SVD : A = UΣVᵀ (from-scratch via eigenvalues de AᵀA)
    - Valeurs singulières σᵢ = √(λᵢ(AᵀA))
    - Rang, noyau, image en termes de la SVD
    - Approximation de rang bas (meilleure approx. au sens de Frobenius)
    - Application : compression d'image
    - Relation avec le théorème spectral (cas symétrique)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def svd_from_scratch(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD from-scratch via les valeurs propres de AᵀA.

    1. AᵀA = V Σ² Vᵀ  (théorème spectral, car AᵀA est SPD)
    2. σᵢ = √(λᵢ)
    3. uᵢ = A vᵢ / σᵢ

    Renvoie (U, sigma, Vt) comme numpy.linalg.svd.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape

    # AᵀA → valeurs propres et vecteurs propres
    ATA = A.T @ A
    eigvals, V = np.linalg.eigh(ATA)

    # Trier par valeur décroissante
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    V = V[:, idx]

    # Valeurs singulières
    sigma = np.sqrt(np.maximum(eigvals, 0))

    # Construire U
    r = min(np.sum(sigma > 1e-12), m)  # rang effectif, borné par m
    U = np.zeros((m, m))
    for i in range(r):
        U[:, i] = A @ V[:, i] / sigma[i]

    # Compléter U en base orthonormée
    if r < m:
        Q, _ = np.linalg.qr(np.column_stack([U[:, :r], np.eye(m)]))
        U = Q[:, :m]

    return U, sigma[:min(m, n)], V.T


def approximation_rang_k(A: np.ndarray, k: int) -> np.ndarray:
    """
    Meilleure approximation de rang k au sens de Frobenius :
        A_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ.

    Théorème d'Eckart-Young : ||A - A_k||_F = √(Σᵢ₌ₖ₊₁ σᵢ²).
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def compression_ratio(A: np.ndarray, k: int) -> float:
    """Ratio de compression : k(m+n+1) / (m·n)."""
    m, n = A.shape
    return k * (m + n + 1) / (m * n)


def erreur_approximation(A: np.ndarray, k: int) -> float:
    """||A - A_k||_F / ||A||_F."""
    Ak = approximation_rang_k(A, k)
    return np.linalg.norm(A - Ak, "fro") / np.linalg.norm(A, "fro")


def tracer_valeurs_singulieres(A: np.ndarray, ax: plt.Axes | None = None) -> plt.Axes:
    """Trace les valeurs singulières en échelle log."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    s = np.linalg.svd(A, compute_uv=False)
    ax.semilogy(range(1, len(s)+1), s, "bo-", markersize=5)
    ax.set_xlabel("indice $i$"); ax.set_ylabel("$\\sigma_i$")
    ax.set_title(f"Valeurs singulières ({A.shape[0]}×{A.shape[1]})")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_approximation(A: np.ndarray, ks: tuple[int, ...] = (1, 3, 5, 10),
                          ax: plt.Axes | None = None) -> plt.Axes:
    """Trace l'erreur relative en fonction du rang k."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    r = min(A.shape)
    all_k = range(1, r)
    errs = [erreur_approximation(A, k) for k in all_k]
    ax.semilogy(list(all_k), errs, "b-", linewidth=2)
    for k in ks:
        if k < r:
            ax.plot(k, erreur_approximation(A, k), "ro", markersize=8)
            ax.annotate(f"k={k}", (k, erreur_approximation(A, k)),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("rang $k$"); ax.set_ylabel("$\\|A - A_k\\|_F / \\|A\\|_F$")
    ax.set_title("Approximation de rang bas (Eckart-Young)")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_geometrie_svd(A: np.ndarray, ax: plt.Axes | None = None) -> plt.Axes:
    """Montre A = rotation × étirement × rotation."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    U, s, Vt = np.linalg.svd(A)
    theta = np.linspace(0, 2*np.pi, 100)
    cercle = np.array([np.cos(theta), np.sin(theta)])

    # Étapes : x → Vᵀx (rotation) → Σ·(Vᵀx) (étirement) → U·Σ·Vᵀx (rotation)
    step1 = Vt @ cercle
    step2 = np.diag(s) @ step1
    step3 = U @ step2

    ax.plot(cercle[0], cercle[1], "b-", alpha=0.3, label="x (cercle)")
    ax.plot(step3[0], step3[1], "r-", linewidth=2, label="Ax (ellipse)")

    # Axes = valeurs singulières
    for i in range(2):
        v = U[:, i] * s[i]
        ax.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1,
                  color="green", width=0.012, label=f"σ_{i+1}u_{i+1} ({s[i]:.2f})")

    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title("SVD : cercle → ellipse (axes = σᵢuᵢ)")
    ax.legend(fontsize=8)
    lim = max(s) * 1.3
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    return ax


if __name__ == "__main__":
    print("=== SVD from-scratch ===")
    A = np.array([[3, 2, 2], [2, 3, -2]], dtype=float)
    U, s, Vt = svd_from_scratch(A)
    U_np, s_np, Vt_np = np.linalg.svd(A, full_matrices=True)
    print(f"  A = {A.tolist()}")
    print(f"  σ (mine)  = {np.round(s, 6)}")
    print(f"  σ (numpy) = {np.round(s_np, 6)}")
    print(f"  ||σ - σ_np|| = {np.linalg.norm(s - s_np[:len(s)]):.2e}")

    print(f"\n=== Propriétés ===")
    print(f"  rang(A) = nombre de σ > 0 = {np.sum(s > 1e-10)}")
    print(f"  ||A||_2 = σ_max = {s[0]:.4f}")
    print(f"  ||A||_F = √(Σσ²) = {np.sqrt(np.sum(s**2)):.4f} "
          f"(direct : {np.linalg.norm(A, 'fro'):.4f})")

    print(f"\n=== Approximation de rang bas ===")
    rng = np.random.default_rng(42)
    B = rng.standard_normal((20, 15))
    for k in [1, 3, 5, 10]:
        err = erreur_approximation(B, k)
        ratio = compression_ratio(B, k)
        print(f"  k={k:>2} : erreur = {err:.4f}, compression = {ratio:.2%}")

    print(f"\n=== Lien avec le théorème spectral ===")
    S = np.array([[4, 2], [2, 3]], dtype=float)
    U_s, s_s, Vt_s = np.linalg.svd(S)
    eigvals = np.linalg.eigvalsh(S)
    print(f"  A symétrique : σ = {np.round(s_s, 4)}, |λ| = {np.round(np.sort(np.abs(eigvals))[::-1], 4)}")
    print(f"  → Pour matrices symétriques positives : σᵢ = λᵢ")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_valeurs_singulieres(B, ax=axes[0])
    tracer_approximation(B, ks=(1, 3, 5, 10), ax=axes[1])
    tracer_geometrie_svd(np.array([[3, 1], [1, 2]], dtype=float), ax=axes[2])
    plt.tight_layout()
    plt.savefig("svd_intro_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
