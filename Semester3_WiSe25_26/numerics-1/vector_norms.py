"""
vector_norms.py
===============

Implémentation et étude des normes vectorielles sur R^n.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", chapitre 1.2.

Couvre :
    - les 4 axiomes d'une norme (Définition 1.2)
    - les p-normes (1, 2, ∞)
    - l'équivalence des normes en dimension finie (Satz 1.5, Beispiel 1.6)
    - la visualisation des boules unitaires (Übung 1.3)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
#  1. Implémentations from-scratch des p-normes
# ----------------------------------------------------------------------

def norm_p(x: np.ndarray, p: float = 2.0) -> float:
    """
    Calcule la p-norme d'un vecteur : ||x||_p = (Σ |x_i|^p)^(1/p).

    Implémentation pédagogique, sans `np.linalg.norm`.

    Paramètres
    ----------
    x : np.ndarray
        Vecteur de R^n.
    p : float
        Ordre de la norme (p >= 1).

    Renvoie
    -------
    float
        La valeur de ||x||_p.

    Lève
    ----
    ValueError
        Si p < 1 (l'inégalité triangulaire échoue sinon).
    """
    if p < 1:
        raise ValueError(f"p doit être >= 1 pour définir une norme (reçu p={p}).")
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.abs(x) ** p) ** (1.0 / p))


def norm_inf(x: np.ndarray) -> float:
    """Norme du maximum : ||x||_∞ = max_i |x_i|."""
    x = np.asarray(x, dtype=float)
    return float(np.max(np.abs(x)))


def norm_1(x: np.ndarray) -> float:
    """Norme 1 : ||x||_1 = Σ |x_i|."""
    return norm_p(x, p=1)


def norm_2(x: np.ndarray) -> float:
    """Norme euclidienne : ||x||_2 = sqrt(Σ x_i^2)."""
    return norm_p(x, p=2)


# ----------------------------------------------------------------------
#  2. Vérification numérique des axiomes (Définition 1.2)
# ----------------------------------------------------------------------

@dataclass
class AxiomReport:
    """Rapport de vérification des 4 axiomes pour une norme donnée."""
    positivite: bool
    definitude: bool
    homogeneite: bool
    inegalite_triangulaire: bool

    @property
    def tous_verifies(self) -> bool:
        return all([
            self.positivite,
            self.definitude,
            self.homogeneite,
            self.inegalite_triangulaire,
        ])

    def __repr__(self) -> str:
        marque = lambda b: "✓" if b else "✗"
        return (
            f"AxiomReport(\n"
            f"  positivité            : {marque(self.positivite)}\n"
            f"  définitude            : {marque(self.definitude)}\n"
            f"  homogénéité           : {marque(self.homogeneite)}\n"
            f"  inégalité triangulaire: {marque(self.inegalite_triangulaire)}\n"
            f")"
        )


def verifier_axiomes(
    norm: Callable[[np.ndarray], float],
    n: int = 5,
    n_essais: int = 1000,
    tol: float = 1e-10,
    seed: int | None = 42,
) -> AxiomReport:
    """
    Vérifie numériquement les 4 axiomes d'une norme par échantillonnage aléatoire.

    Ce n'est pas une preuve — c'est un sanity check : si l'un des axiomes échoue
    pour un échantillon aléatoire, la fonction n'est sûrement pas une norme.
    Si tous passent, c'est un indice fort (mais pas une preuve).

    Paramètres
    ----------
    norm : Callable
        Fonction prenant un vecteur et renvoyant un float.
    n : int
        Dimension de l'espace testé.
    n_essais : int
        Nombre de tirages aléatoires par axiome.
    tol : float
        Tolérance numérique.
    seed : int | None
        Graine du générateur aléatoire pour la reproductibilité.

    Renvoie
    -------
    AxiomReport
    """
    rng = np.random.default_rng(seed)

    # Positivité : ||x|| >= 0
    positivite = True
    for _ in range(n_essais):
        x = rng.standard_normal(n)
        if norm(x) < -tol:
            positivite = False
            break

    # Définitude : ||x|| > 0 si x != 0  (et ||0|| = 0)
    definitude = abs(norm(np.zeros(n))) < tol
    if definitude:
        for _ in range(n_essais):
            x = rng.standard_normal(n)
            if np.linalg.norm(x) > tol and norm(x) <= tol:
                definitude = False
                break

    # Homogénéité positive : ||λ x|| = |λ| · ||x||
    homogeneite = True
    for _ in range(n_essais):
        x = rng.standard_normal(n)
        lam = rng.standard_normal()
        if abs(norm(lam * x) - abs(lam) * norm(x)) > tol * (1 + norm(x)):
            homogeneite = False
            break

    # Inégalité triangulaire : ||x + y|| <= ||x|| + ||y||
    triangulaire = True
    for _ in range(n_essais):
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        if norm(x + y) > norm(x) + norm(y) + tol:
            triangulaire = False
            break

    return AxiomReport(positivite, definitude, homogeneite, triangulaire)


# ----------------------------------------------------------------------
#  3. Équivalence des normes (Beispiel 1.6)
# ----------------------------------------------------------------------

def constantes_equivalence_empirique(
    norm_a: Callable[[np.ndarray], float],
    norm_b: Callable[[np.ndarray], float],
    n: int = 5,
    n_essais: int = 10_000,
    seed: int | None = 42,
) -> tuple[float, float]:
    """
    Estime empiriquement les constantes L, M telles que :
        ||x||_a <= L · ||x||_b   et   ||x||_b <= M · ||x||_a.

    Ces constantes existent toujours en dimension finie (Satz 1.5),
    mais peuvent dépendre de la dimension n.

    Renvoie
    -------
    (L, M) : tuple[float, float]
        Bornes empiriques (inférieures aux vraies valeurs théoriques).
    """
    rng = np.random.default_rng(seed)
    L_max, M_max = 0.0, 0.0
    for _ in range(n_essais):
        x = rng.standard_normal(n)
        nb = norm_b(x)
        na = norm_a(x)
        if nb > 0:
            L_max = max(L_max, na / nb)
        if na > 0:
            M_max = max(M_max, nb / na)
    return L_max, M_max


# ----------------------------------------------------------------------
#  4. Visualisation des boules unitaires (Übung 1.3)
# ----------------------------------------------------------------------

def boule_unitaire_2d(p: float, n_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """
    Renvoie un échantillonnage du bord de la boule unitaire {x ∈ R^2 : ||x||_p = 1}.

    Méthode : on génère des points sur le cercle unitaire en angle, puis on
    les renormalise par leur p-norme.

    Paramètres
    ----------
    p : float
        Ordre de la norme (p >= 1, ou np.inf).
    n_points : int
        Nombre de points sur le bord.

    Renvoie
    -------
    (xs, ys) : deux np.ndarray de longueur n_points.
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])

    if np.isinf(p):
        normes = np.max(np.abs(pts), axis=1)
    else:
        normes = np.sum(np.abs(pts) ** p, axis=1) ** (1.0 / p)

    pts = pts / normes[:, None]
    return pts[:, 0], pts[:, 1]


def tracer_boules_unitaires(
    valeurs_p: tuple[float, ...] = (1, 1.5, 2, 4, np.inf),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Trace les boules unitaires en R^2 pour différentes valeurs de p.

    Reproduit l'illustration de l'Übung 1.3 du script.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    for p in valeurs_p:
        x, y = boule_unitaire_2d(p)
        label = "p = ∞" if np.isinf(p) else f"p = {p}"
        ax.plot(x, y, label=label, linewidth=2)

    ax.set_aspect("equal")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Boules unitaires $\\{x \\in \\mathbb{R}^2 : \\|x\\|_p \\leq 1\\}$")
    ax.legend(loc="upper right")
    return ax


# ----------------------------------------------------------------------
#  5. Comparaison from-scratch ↔ NumPy
# ----------------------------------------------------------------------

def comparer_avec_numpy(x: np.ndarray, ps: tuple[float, ...] = (1, 2, 4, np.inf)) -> None:
    """
    Affiche un tableau comparatif entre l'implémentation from-scratch
    et `np.linalg.norm`. Sert de test de cohérence.
    """
    print(f"Vecteur x = {x}")
    print(f"{'p':>6} | {'from-scratch':>16} | {'numpy':>16} | {'écart':>10}")
    print("-" * 60)
    for p in ps:
        if np.isinf(p):
            mine = norm_inf(x)
        else:
            mine = norm_p(x, p)
        ref = float(np.linalg.norm(x, ord=p))
        ecart = abs(mine - ref)
        p_str = "inf" if np.isinf(p) else f"{p}"
        print(f"{p_str:>6} | {mine:>16.10f} | {ref:>16.10f} | {ecart:>10.2e}")


# ----------------------------------------------------------------------
#  Point d'entrée — petite démo si exécuté directement
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Démo : vecteur (3, -4) ===")
    x = np.array([3.0, -4.0])
    comparer_avec_numpy(x)

    print("\n=== Vérification des axiomes pour la norme 2 ===")
    print(verifier_axiomes(norm_2))

    print("\n=== Constantes d'équivalence empiriques (n=5) ===")
    L, M = constantes_equivalence_empirique(norm_inf, norm_1, n=5)
    print(f"||x||_∞ ≤ L · ||x||_1 avec L_emp ≈ {L:.4f}  (théorie : 1)")
    print(f"||x||_1 ≤ M · ||x||_∞ avec M_emp ≈ {M:.4f}  (théorie : n = 5)")

    print("\n=== Tracé des boules unitaires ===")
    tracer_boules_unitaires()
    plt.tight_layout()
    plt.savefig("boules_unitaires.png", dpi=120)
    print("Figure sauvegardée : boules_unitaires.png")
