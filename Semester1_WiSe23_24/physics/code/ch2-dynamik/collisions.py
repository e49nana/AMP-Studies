"""
collisions.py
=============

Chocs élastiques et inélastiques en 1D et 2D.

Couvre :
    - Conservation de la quantité de mouvement : Σ m·v = const
    - Choc élastique : conservation de E_cin → formules exactes
    - Choc parfaitement inélastique : les corps restent collés
    - Coefficient de restitution e = |v₂'-v₁'| / |v₁-v₂|
    - Chocs 2D (billard)
    - Simulation et visualisation

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ChocResult:
    """Résultat d'un choc."""
    v1_avant: float
    v2_avant: float
    v1_apres: float
    v2_apres: float
    type_choc: str
    e: float  # coefficient de restitution
    delta_Ecin: float  # perte d'énergie cinétique


# ======================================================================
#  1. Chocs 1D
# ======================================================================

def choc_elastique_1d(m1: float, m2: float, v1: float, v2: float) -> ChocResult:
    """
    Choc parfaitement élastique (e = 1) :
        v₁' = ((m₁-m₂)v₁ + 2m₂v₂) / (m₁+m₂)
        v₂' = ((m₂-m₁)v₂ + 2m₁v₁) / (m₁+m₂)
    Conservation de p ET de E_cin.
    """
    M = m1 + m2
    v1p = ((m1 - m2) * v1 + 2 * m2 * v2) / M
    v2p = ((m2 - m1) * v2 + 2 * m1 * v1) / M
    Ecin_avant = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
    Ecin_apres = 0.5 * m1 * v1p**2 + 0.5 * m2 * v2p**2
    return ChocResult(v1, v2, v1p, v2p, "élastique", 1.0, Ecin_apres - Ecin_avant)


def choc_inelastique_1d(m1: float, m2: float, v1: float, v2: float) -> ChocResult:
    """
    Choc parfaitement inélastique (e = 0) : les corps restent collés.
        v' = (m₁v₁ + m₂v₂) / (m₁+m₂).
    Perte maximale d'énergie cinétique.
    """
    v_common = (m1 * v1 + m2 * v2) / (m1 + m2)
    Ecin_avant = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
    Ecin_apres = 0.5 * (m1 + m2) * v_common**2
    return ChocResult(v1, v2, v_common, v_common, "inélastique", 0.0,
                      Ecin_apres - Ecin_avant)


def choc_partiel_1d(
    m1: float, m2: float, v1: float, v2: float, e: float,
) -> ChocResult:
    """
    Choc avec coefficient de restitution e ∈ [0, 1].
        v₁' = (m₁v₁ + m₂v₂ + m₂·e·(v₂-v₁)) / (m₁+m₂)
        v₂' = (m₁v₁ + m₂v₂ + m₁·e·(v₁-v₂)) / (m₁+m₂)
    """
    M = m1 + m2
    v1p = (m1*v1 + m2*v2 + m2*e*(v2 - v1)) / M
    v2p = (m1*v1 + m2*v2 + m1*e*(v1 - v2)) / M
    Ecin_avant = 0.5*m1*v1**2 + 0.5*m2*v2**2
    Ecin_apres = 0.5*m1*v1p**2 + 0.5*m2*v2p**2
    return ChocResult(v1, v2, v1p, v2p, f"partiel (e={e})", e,
                      Ecin_apres - Ecin_avant)


def verifier_conservation(m1, m2, r: ChocResult) -> dict:
    """Vérifie conservation de p (toujours) et E (si élastique)."""
    p_avant = m1 * r.v1_avant + m2 * r.v2_avant
    p_apres = m1 * r.v1_apres + m2 * r.v2_apres
    E_avant = 0.5*m1*r.v1_avant**2 + 0.5*m2*r.v2_avant**2
    E_apres = 0.5*m1*r.v1_apres**2 + 0.5*m2*r.v2_apres**2
    return {
        "p conservée": np.isclose(p_avant, p_apres),
        "E conservée": np.isclose(E_avant, E_apres),
        "ΔE/E": (E_apres - E_avant) / E_avant if E_avant > 0 else 0,
    }


# ======================================================================
#  2. Cas spéciaux
# ======================================================================

def cas_speciaux() -> None:
    """Cas remarquables du choc élastique."""
    print("=== Cas spéciaux (choc élastique) ===\n")

    # Masses égales
    r = choc_elastique_1d(1, 1, 5, 0)
    print(f"  Masses égales : v₁=5 → v₁'={r.v1_apres:.1f}, v₂'={r.v2_apres:.1f}")
    print(f"    → Échange complet des vitesses (billard)\n")

    # Masse lourde frappe légère
    r = choc_elastique_1d(100, 1, 5, 0)
    print(f"  m₁≫m₂ : v₁≈{r.v1_apres:.2f} (quasi inchangé), v₂'≈{r.v2_apres:.1f} (≈2v₁)")

    # Masse légère frappe lourde
    r = choc_elastique_1d(1, 100, 5, 0)
    print(f"  m₁≪m₂ : v₁'≈{r.v1_apres:.2f} (rebond), v₂'≈{r.v2_apres:.3f} (quasi immobile)")


# ======================================================================
#  3. Choc 2D (billard)
# ======================================================================

def choc_2d_elastique(
    m1: float, m2: float,
    v1: np.ndarray, v2: np.ndarray,
    n: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Choc élastique 2D. n = vecteur normal au point de contact.
    Seules les composantes normales sont échangées.
    """
    n = n / np.linalg.norm(n)
    v1n = np.dot(v1, n)
    v2n = np.dot(v2, n)
    M = m1 + m2
    v1n_new = ((m1-m2)*v1n + 2*m2*v2n) / M
    v2n_new = ((m2-m1)*v2n + 2*m1*v1n) / M
    v1_new = v1 + (v1n_new - v1n) * n
    v2_new = v2 + (v2n_new - v2n) * n
    return v1_new, v2_new


# ======================================================================
#  4. Tracés
# ======================================================================

def tracer_perte_energie(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    es = np.linspace(0, 1, 100)
    m1, v1 = 1, 5
    for m2 in [0.5, 1, 2, 5]:
        pertes = []
        for e in es:
            r = choc_partiel_1d(m1, m2, v1, 0, e)
            E0 = 0.5*m1*v1**2
            pertes.append(-r.delta_Ecin / E0 * 100)
        ax.plot(es, pertes, linewidth=2, label=f"m₂/m₁ = {m2/m1}")

    ax.set_xlabel("coefficient de restitution $e$")
    ax.set_ylabel("perte d'énergie cinétique (%)")
    ax.set_title("Perte d'énergie vs restitution")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_choc_2d(ax: plt.Axes | None = None) -> plt.Axes:
    """Simule un choc de billard."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    m1, m2 = 1, 1
    v1 = np.array([3.0, 1.0])
    v2 = np.array([0.0, 0.0])
    n = np.array([1.0, 0.3])

    v1p, v2p = choc_2d_elastique(m1, m2, v1, v2, n)

    # Trajectoires
    t = np.linspace(0, 2, 100)
    for vi, vf, nom, c in [(v1, v1p, "boule 1", "blue"), (v2, v2p, "boule 2", "red")]:
        # Avant
        ax.plot(-vi[0]*t, -vi[1]*t, f"{c}", linestyle="--", alpha=0.3)
        # Après
        ax.plot(vf[0]*t, vf[1]*t, f"{c}", linewidth=2, label=f"{nom} après")
        ax.quiver(0, 0, vf[0], vf[1], angles="xy", scale_units="xy", scale=1,
                  color=c, width=0.015)

    ax.plot(0, 0, "ko", markersize=15, label="point de contact")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title("Choc 2D élastique (billard)")
    ax.legend(fontsize=9)
    ax.set_xlim(-4, 5); ax.set_ylim(-3, 4)
    return ax


if __name__ == "__main__":
    print("=== Choc élastique 1D ===\n")
    m1, m2 = 2, 3
    r = choc_elastique_1d(m1, m2, 5, -2)
    print(f"  m₁={m1}, m₂={m2}, v₁=5, v₂=-2")
    print(f"  → v₁'={r.v1_apres:.4f}, v₂'={r.v2_apres:.4f}")
    v = verifier_conservation(m1, m2, r)
    print(f"  p conservée ? {v['p conservée']} ✓")
    print(f"  E conservée ? {v['E conservée']} ✓")

    print(f"\n=== Choc inélastique ===\n")
    r = choc_inelastique_1d(m1, m2, 5, -2)
    print(f"  → v' = {r.v1_apres:.4f} (collés)")
    print(f"  ΔE = {r.delta_Ecin:.2f} J (perte)")
    v = verifier_conservation(m1, m2, r)
    print(f"  p conservée ? {v['p conservée']} ✓")
    print(f"  ΔE/E = {v['ΔE/E']*100:.1f}%")

    print()
    cas_speciaux()

    print(f"\n=== Choc 2D ===\n")
    v1 = np.array([3.0, 1.0])
    v2 = np.array([0.0, 0.0])
    v1p, v2p = choc_2d_elastique(1, 1, v1, v2, np.array([1.0, 0.3]))
    print(f"  v₁ = {v1} → v₁' = {np.round(v1p, 4)}")
    print(f"  v₂ = {v2} → v₂' = {np.round(v2p, 4)}")
    print(f"  p conservée ? {np.allclose(v1+v2, v1p+v2p)} ✓")
    E_avant = 0.5*(np.dot(v1,v1)+np.dot(v2,v2))
    E_apres = 0.5*(np.dot(v1p,v1p)+np.dot(v2p,v2p))
    print(f"  E conservée ? {np.isclose(E_avant, E_apres)} ✓")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tracer_perte_energie(ax=axes[0])
    tracer_choc_2d(ax=axes[1])
    plt.tight_layout()
    plt.savefig("collisions_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
