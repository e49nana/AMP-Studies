"""
jacobi_gauss_seidel.py
======================

Méthodes itératives pour les systèmes linéaires : Jacobi (Gesamtschritt-)
et Gauss-Seidel (Einzelschrittverfahren).

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", section 2.5.

Couvre :
    - Jacobi (formule 2.17)
    - Gauss-Seidel (formule 2.18)
    - Critère d'arrêt a-posteriori (formule 2.23, Banach)
    - Critère de convergence par dominance diagonale stricte (Satz 2.37)
    - Critère de convergence par rayon spectral (Satz 2.35)
    - Calcul de la matrice d'itération S et de la rate de convergence
    - Application : équation de la chaleur 2D (Beispiel 2.24)

Notations du script :
    A x = b avec décomposition A = L + D + R
        D : diagonale
        L : strict lower triangular
        R : strict upper triangular
    Jacobi      : x^+ = -D⁻¹(L+R) x + D⁻¹ b      => S_J = -D⁻¹(L+R)
    Gauss-Seidel: x^+ = -(L+D)⁻¹ R x + (L+D)⁻¹ b => S_GS = -(L+D)⁻¹ R

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ----------------------------------------------------------------------
#  1. Résultat d'une itération
# ----------------------------------------------------------------------

@dataclass
class ResultatIteratif:
    """Stocke la solution et l'historique de convergence."""
    x: np.ndarray
    iterations: int
    converge: bool
    historique_residus: list[float] = field(default_factory=list)
    historique_x: list[np.ndarray] = field(default_factory=list)
    methode: str = ""

    def __repr__(self) -> str:
        statut = "convergé" if self.converge else "divergé / max atteint"
        return (
            f"ResultatIteratif({self.methode}, {statut} "
            f"en {self.iterations} itérations, "
            f"résidu final = {self.historique_residus[-1]:.2e})"
        )


# ----------------------------------------------------------------------
#  2. Jacobi (formule 2.17 du script)
# ----------------------------------------------------------------------

def jacobi(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    n_max: int = 10_000,
    stocker_x: bool = False,
) -> ResultatIteratif:
    """
    Méthode de Jacobi (Gesamtschrittverfahren).

    Formule 2.17 : x_i^+ = (1/a_ii) * (b_i - Σ_{j ≠ i} a_ij x_j).

    Toutes les composantes sont mises à jour simultanément à partir de
    l'itéré précédent — d'où le nom "Gesamtschritt".

    Paramètres
    ----------
    A : np.ndarray (n, n)
        Matrice du système. Diagonale supposée non nulle.
    b : np.ndarray (n,)
    x0 : vecteur initial (par défaut : zéros).
    tol : tolérance sur ||x^+ - x||_∞ pour critère d'arrêt.
    n_max : nombre maximal d'itérations.
    stocker_x : si True, garde tous les itérés (utile pour les animations).

    Renvoie
    -------
    ResultatIteratif
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    x = np.asarray(x0, dtype=float).copy()

    diag = np.diag(A)
    if np.any(diag == 0):
        raise ValueError("Diagonale nulle : Jacobi non applicable.")

    # Matrice "hors-diagonale" : A - diag(diag(A))
    R_hors_diag = A - np.diag(diag)

    historique_residus = [float(np.linalg.norm(A @ x - b, np.inf))]
    historique_x = [x.copy()] if stocker_x else []

    converge = False
    for k in range(1, n_max + 1):
        # Vectorisation de la formule 2.17 :
        # x_i^+ = (b_i - Σ_{j ≠ i} a_ij x_j) / a_ii
        x_new = (b - R_hors_diag @ x) / diag

        diff = float(np.linalg.norm(x_new - x, np.inf))
        x = x_new
        historique_residus.append(float(np.linalg.norm(A @ x - b, np.inf)))
        if stocker_x:
            historique_x.append(x.copy())

        if diff < tol:
            converge = True
            break

    return ResultatIteratif(
        x=x, iterations=k, converge=converge,
        historique_residus=historique_residus,
        historique_x=historique_x, methode="Jacobi",
    )


# ----------------------------------------------------------------------
#  3. Gauss-Seidel (formule 2.18 du script)
# ----------------------------------------------------------------------

def gauss_seidel(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    n_max: int = 10_000,
    stocker_x: bool = False,
) -> ResultatIteratif:
    """
    Méthode de Gauss-Seidel (Einzelschrittverfahren).

    Formule 2.18 :
        x_i^+ = (1/a_ii) * (b_i - Σ_{j<i} a_ij x_j^+ - Σ_{j>i} a_ij x_j).

    Différence avec Jacobi : on utilise les composantes déjà mises à
    jour x_j^+ pour j < i. D'où le nom "Einzelschritt" — chaque
    composante est traitée individuellement et la mise à jour est
    immédiate.

    Conséquence : pas vectorisable proprement, on doit boucler sur i.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    x = np.asarray(x0, dtype=float).copy()

    diag = np.diag(A)
    if np.any(diag == 0):
        raise ValueError("Diagonale nulle : Gauss-Seidel non applicable.")

    historique_residus = [float(np.linalg.norm(A @ x - b, np.inf))]
    historique_x = [x.copy()] if stocker_x else []

    converge = False
    for k in range(1, n_max + 1):
        x_old = x.copy()
        for i in range(n):
            somme = A[i, :i] @ x[:i] + A[i, i + 1:] @ x[i + 1:]
            x[i] = (b[i] - somme) / A[i, i]
        diff = float(np.linalg.norm(x - x_old, np.inf))
        historique_residus.append(float(np.linalg.norm(A @ x - b, np.inf)))
        if stocker_x:
            historique_x.append(x.copy())
        if diff < tol:
            converge = True
            break

    return ResultatIteratif(
        x=x, iterations=k, converge=converge,
        historique_residus=historique_residus,
        historique_x=historique_x, methode="Gauss-Seidel",
    )


# ----------------------------------------------------------------------
#  4. Diagnostic théorique : matrices d'itération et convergence
# ----------------------------------------------------------------------

def matrice_iteration_jacobi(A: np.ndarray) -> np.ndarray:
    """
    S_J = -D⁻¹ (L + R)  où A = L + D + R (L strict, R strict, D diagonale).
    """
    A = np.asarray(A, dtype=float)
    D = np.diag(np.diag(A))
    return -np.linalg.inv(D) @ (A - D)


def matrice_iteration_gauss_seidel(A: np.ndarray) -> np.ndarray:
    """
    S_GS = -(L + D)⁻¹ R.
    """
    A = np.asarray(A, dtype=float)
    L_plus_D = np.tril(A)
    R = np.triu(A, k=1)
    return -np.linalg.solve(L_plus_D, R)


def rayon_spectral(M: np.ndarray) -> float:
    """ρ(M) = max |λ| pour λ valeur propre de M (Satz 2.35)."""
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def est_strictement_diag_dominant(A: np.ndarray) -> bool:
    """
    Définition 2.36 : Σ_{j≠i} |a_ij| < |a_ii| pour tout i.
    Si oui, Jacobi converge pour tout x_0 (Satz 2.37).
    """
    A = np.asarray(A, dtype=float)
    abs_A = np.abs(A)
    diag = np.diag(abs_A)
    sommes_hors_diag = abs_A.sum(axis=1) - diag
    return bool(np.all(sommes_hors_diag < diag))


@dataclass
class DiagnosticConvergence:
    """Résumé du diagnostic théorique pour une matrice donnée."""
    diag_dominante: bool
    rho_jacobi: float
    rho_gauss_seidel: float

    @property
    def jacobi_converge(self) -> bool:
        return self.rho_jacobi < 1.0

    @property
    def gauss_seidel_converge(self) -> bool:
        return self.rho_gauss_seidel < 1.0

    @property
    def rate_jacobi(self) -> float:
        """Asymptotic rate (chiffres décimaux gagnés par itération)."""
        return -np.log10(self.rho_jacobi) if self.rho_jacobi > 0 else float("inf")

    @property
    def rate_gauss_seidel(self) -> float:
        return -np.log10(self.rho_gauss_seidel) if self.rho_gauss_seidel > 0 else float("inf")

    def __repr__(self) -> str:
        return (
            f"DiagnosticConvergence(\n"
            f"  diag dominante       : {self.diag_dominante}\n"
            f"  ρ(S_Jacobi)          : {self.rho_jacobi:.4f}  -> "
            f"{'converge' if self.jacobi_converge else 'diverge'}\n"
            f"  ρ(S_Gauss-Seidel)    : {self.rho_gauss_seidel:.4f}  -> "
            f"{'converge' if self.gauss_seidel_converge else 'diverge'}\n"
            f"  rate Jacobi          : {self.rate_jacobi:.4f} chiffres/itération\n"
            f"  rate Gauss-Seidel    : {self.rate_gauss_seidel:.4f} chiffres/itération\n"
            f")"
        )


def diagnostiquer(A: np.ndarray) -> DiagnosticConvergence:
    """Calcule tous les indicateurs théoriques de convergence."""
    return DiagnosticConvergence(
        diag_dominante=est_strictement_diag_dominant(A),
        rho_jacobi=rayon_spectral(matrice_iteration_jacobi(A)),
        rho_gauss_seidel=rayon_spectral(matrice_iteration_gauss_seidel(A)),
    )


# ----------------------------------------------------------------------
#  5. Application : équation de la chaleur 2D (Beispiel 2.24)
# ----------------------------------------------------------------------

def systeme_chaleur_2d(
    n: int,
    bord_haut: float = 100.0,
    bord_bas: float = 0.0,
    bord_gauche: float = 0.0,
    bord_droit: float = 0.0,
    sparse_format: bool = False,
) -> tuple[np.ndarray | sparse.csr_matrix, np.ndarray]:
    """
    Discrétise l'équation de Laplace -ΔT = 0 sur [0,1]² avec conditions
    de Dirichlet, sur un grille n×n de points intérieurs (donc N = n² inconnues).

    Schéma à 5 points : pour chaque (i,j) intérieur,
        4 T_{i,j} - T_{i-1,j} - T_{i+1,j} - T_{i,j-1} - T_{i,j+1} = 0

    Aux bords, le terme correspondant passe au second membre.

    Paramètres
    ----------
    n : nombre de points intérieurs par côté.
    bord_* : valeurs aux 4 bords.
    sparse_format : si True, renvoie une matrice CSR (indispensable pour n grand).

    Renvoie
    -------
    (A, b) : matrice N×N et second membre de longueur N = n².
    """
    N = n * n
    main_diag = 4.0 * np.ones(N)
    off1 = -np.ones(N - 1)
    # Empêche les voisins gauche-droite de "wrapper" entre lignes
    off1[np.arange(1, N) % n == 0] = 0.0
    offn = -np.ones(N - n)

    diagonals = [main_diag, off1, off1, offn, offn]
    offsets = [0, -1, 1, -n, n]
    A = sparse.diags(diagonals, offsets, shape=(N, N), format="csr")

    b = np.zeros(N)
    # Conditions de bord
    for j in range(n):
        b[j] += bord_bas       # ligne i=0
        b[(n - 1) * n + j] += bord_haut  # ligne i=n-1
    for i in range(n):
        b[i * n] += bord_gauche
        b[i * n + n - 1] += bord_droit

    if sparse_format:
        return A, b
    return A.toarray(), b


# ----------------------------------------------------------------------
#  6. Tracés
# ----------------------------------------------------------------------

def tracer_convergence(
    resultats: list[ResultatIteratif],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Trace ||r_k||_∞ en échelle semi-log pour plusieurs méthodes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    couleurs = {"Jacobi": "tab:blue", "Gauss-Seidel": "tab:red"}
    for r in resultats:
        c = couleurs.get(r.methode, "k")
        ax.semilogy(r.historique_residus, color=c, label=r.methode, linewidth=2)
    ax.set_xlabel("itération $k$")
    ax.set_ylabel(r"$\|r_k\|_\infty = \|A x_k - b\|_\infty$")
    ax.set_title("Convergence des méthodes itératives")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax


def tracer_solution_chaleur(
    x: np.ndarray, n: int, ax: plt.Axes | None = None,
) -> plt.Axes:
    """Affiche la solution de chaleur 2D comme heatmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    T = x.reshape((n, n))
    im = ax.imshow(T, origin="lower", cmap="hot", extent=[0, 1, 0, 1])
    ax.set_title(f"Température (Beispiel 2.24, n={n})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="T")
    return ax


# ----------------------------------------------------------------------
#  Démo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Übung 2.26 : système 3×3 du script ===")
    A = np.array([[5, 1, -1], [3, -10, 2], [1, -2, 5]], dtype=float)
    b = np.array([9, 8, 7], dtype=float)
    x0 = np.array([1, 1, 1], dtype=float)

    print(diagnostiquer(A))

    res_j = jacobi(A, b, x0=x0, tol=1e-12)
    res_gs = gauss_seidel(A, b, x0=x0, tol=1e-12)
    x_ref = np.linalg.solve(A, b)
    print(f"\nSolution exacte : {x_ref}")
    print(f"Jacobi      : {res_j}")
    print(f"  -> x = {res_j.x}")
    print(f"Gauss-Seidel: {res_gs}")
    print(f"  -> x = {res_gs.x}")

    print("\n=== Beispiel 2.24 : équation de la chaleur 2D, n=20 ===")
    n = 20
    A_h, b_h = systeme_chaleur_2d(n, bord_haut=100.0)
    diag = diagnostiquer(A_h)
    print(f"Système {n*n}×{n*n}")
    print(f"ρ(S_J)  = {diag.rho_jacobi:.6f}, ρ(S_GS) = {diag.rho_gauss_seidel:.6f}")
    print(f"Asymptotiquement : Gauss-Seidel ≈ {diag.rate_gauss_seidel/diag.rate_jacobi:.2f}× plus rapide")

    res_j = jacobi(A_h, b_h, tol=1e-6, n_max=20_000)
    res_gs = gauss_seidel(A_h, b_h, tol=1e-6, n_max=20_000)
    print(f"Jacobi      : {res_j.iterations} itérations")
    print(f"Gauss-Seidel: {res_gs.iterations} itérations")

    # Tracé de la convergence et de la solution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_convergence([res_j, res_gs], ax=axes[0])
    tracer_solution_chaleur(res_gs.x, n, ax=axes[1])
    plt.tight_layout()
    plt.savefig("jacobi_gauss_seidel_demo.png", dpi=120)
    print("Figure sauvegardée : jacobi_gauss_seidel_demo.png")
