"""
power_iteration.py
==================

Méthodes de Vektoriteration (Von-Mises) et inverse Wielandt
pour les problèmes de valeurs propres.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", section 6.4.

Couvre :
    - Vektoriteration directe (Von-Mises, section 6.4.1) → λ_1 (dominant)
    - Vektoriteration inverse (Wielandt, section 6.4.2) → λ le plus proche de λ̂
    - Übung 6.8
    - Convergence linéaire avec rate |λ₂/λ₁| (Satz 6.7)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ResultatEigenpair:
    """Valeur propre + vecteur propre + historique."""
    eigenvalue: float
    eigenvector: np.ndarray
    iterations: int
    converge: bool
    methode: str
    historique_eigenvalue: list[float] = field(default_factory=list)


# ======================================================================
#  1. Vektoriteration directe (Von-Mises, section 6.4.1)
# ======================================================================

def von_mises(
    A: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-12,
    n_max: int = 10_000,
) -> ResultatEigenpair:
    """
    Potenzmethode : converge vers le couple (λ₁, v₁) avec |λ₁| = ρ(A).

    Algorithme (section 6.4.1) :
        1. y = A x
        2. ℓ = ||y||_2
        3. Correction du signe via la composante dominante
        4. µ⁺ = ℓ̃,  x⁺ = y / ℓ̃

    Convergence linéaire avec rate |λ₂/λ₁| (Satz 6.7).
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if x0 is None:
        x0 = np.random.default_rng(42).standard_normal(n)
    x = x0 / np.linalg.norm(x0, 2)

    hist_ev = []
    mu = 0.0
    converge = False

    for k in range(1, n_max + 1):
        y = A @ x
        ell = np.linalg.norm(y, 2)
        if ell == 0:
            break

        # Correction du signe (point 4 du script)
        i_max = np.argmax(np.abs(y))
        sign = 1.0 if x[i_max] * y[i_max] >= 0 else -1.0
        ell_tilde = sign * ell

        mu_new = ell_tilde
        x = y / ell_tilde
        hist_ev.append(mu_new)

        if abs(mu_new - mu) < tol * (1 + abs(mu_new)):
            mu = mu_new
            converge = True
            break
        mu = mu_new

    return ResultatEigenpair(
        eigenvalue=mu, eigenvector=x, iterations=k,
        converge=converge, methode="Von-Mises (directe)",
        historique_eigenvalue=hist_ev,
    )


# ======================================================================
#  2. Inverse Vektoriteration (Wielandt, section 6.4.2)
# ======================================================================

def wielandt(
    A: np.ndarray,
    sigma: float,
    x0: np.ndarray | None = None,
    tol: float = 1e-12,
    n_max: int = 1_000,
) -> ResultatEigenpair:
    """
    Inverse Iteration : converge vers le couple (λ_i, v_i) avec λ_i
    le plus proche de σ (le "shift").

    Algorithme : appliquer Von-Mises à B = (A - σI)⁻¹.
    En pratique : résoudre (A - σI) y = x à chaque pas (formule 6.3).

    Convergence linéaire avec rate |λ_i - σ| / |λ_k - σ| (k ≠ i).
    Plus σ est proche de λ_i, plus la convergence est rapide.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if x0 is None:
        x0 = np.random.default_rng(42).standard_normal(n)
    x = x0 / np.linalg.norm(x0, 2)

    B = A - sigma * np.eye(n)
    # LU une fois pour toutes
    from scipy.linalg import lu_factor, lu_solve
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lu_piv = lu_factor(B)
        except np.linalg.LinAlgError:
            # σ est exactement une valeur propre → on perturbe légèrement
            eps = np.finfo(float).eps * np.linalg.norm(A, np.inf)
            B = A - (sigma + eps) * np.eye(n)
            lu_piv = lu_factor(B)

    hist_ev = []
    mu = sigma
    converge = False

    for k in range(1, n_max + 1):
        try:
            y = lu_solve(lu_piv, x)
            if not np.all(np.isfinite(y)):
                raise ValueError("NaN/inf in solve")
        except (ValueError, np.linalg.LinAlgError):
            # σ tombé sur une valeur propre — convergence immédiate
            mu = sigma
            converge = True
            hist_ev.append(mu)
            break
        ell = np.linalg.norm(y, 2)
        if ell == 0:
            break

        i_max = np.argmax(np.abs(y))
        sign = 1.0 if x[i_max] * y[i_max] >= 0 else -1.0
        theta = sign * ell  # ≈ 1/(λ_i - σ)

        mu_new = sigma + 1.0 / theta
        x = y / (sign * ell)
        hist_ev.append(mu_new)

        if abs(mu_new - mu) < tol * (1 + abs(mu_new)):
            mu = mu_new
            converge = True
            break
        mu = mu_new

    return ResultatEigenpair(
        eigenvalue=mu, eigenvector=x, iterations=k,
        converge=converge, methode=f"Wielandt (σ={sigma})",
        historique_eigenvalue=hist_ev,
    )


# ======================================================================
#  3. Tracés
# ======================================================================

def tracer_convergence_eigenvalue(
    resultats: list[ResultatEigenpair],
    lambda_exact: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for r in resultats:
        if lambda_exact is not None:
            erreurs = [abs(ev - lambda_exact) for ev in r.historique_eigenvalue]
            ylabel = "$|\\mu_k - \\lambda_1|$"
        else:
            erreurs = r.historique_eigenvalue
            ylabel = "$\\mu_k$"
        ax.semilogy(erreurs, "o-", markersize=3, label=r.methode)
    ax.set_xlabel("itération $k$")
    ax.set_ylabel(ylabel)
    ax.set_title("Convergence vers la valeur propre")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


# ======================================================================
#  Démo
# ======================================================================

if __name__ == "__main__":
    print("=== Übung 6.8 : matrice 2×2 ===")
    A = np.array([[-1.5, 3.5], [3.5, -1.5]])
    eigvals = np.linalg.eigvalsh(A)
    print(f"Valeurs propres exactes : {eigvals}")

    res = von_mises(A, x0=np.array([1.0, 3.0]))
    print(f"\nVon-Mises : {res.eigenvalue:.10f} en {res.iterations} itérations")
    print(f"  vecteur propre : {res.eigenvector}")
    rate_theo = abs(eigvals[0] / eigvals[1])
    print(f"  rate théorique |λ₂/λ₁| = {rate_theo:.4f}")

    print("\n=== Wielandt : chercher λ le plus proche de σ ===")
    A3 = np.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=float)
    eigvals3 = np.sort(np.linalg.eigvalsh(A3))
    print(f"Valeurs propres exactes : {eigvals3}")

    for sigma in [1.5, 3.0, 4.5]:
        res_w = wielandt(A3, sigma)
        print(f"  σ = {sigma:.1f} → λ = {res_w.eigenvalue:.10f} en {res_w.iterations} it.")

    print("\n=== Tracé ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Von-Mises convergence
    res_vm = von_mises(A3, tol=1e-14)
    tracer_convergence_eigenvalue([res_vm], lambda_exact=eigvals3[-1], ax=axes[0])
    axes[0].set_title("Von-Mises → λ dominant")

    # Wielandt convergence pour différents σ
    resultats_w = []
    for sigma in [1.5, 2.5, 3.5]:
        resultats_w.append(wielandt(A3, sigma, tol=1e-14))
    tracer_convergence_eigenvalue(resultats_w, lambda_exact=eigvals3[0], ax=axes[1])
    axes[1].set_title("Wielandt → λ le plus proche de σ")

    plt.tight_layout()
    plt.savefig("power_iteration_demo.png", dpi=120)
    print("Figure sauvegardée : power_iteration_demo.png")
