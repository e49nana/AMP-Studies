"""
power_efficiency.py
===================

Puissance, rendement et machines simples.

Couvre :
    - Puissance P = dW/dt = F·v
    - Puissance instantanée et moyenne
    - Rendement η = P_utile / P_fournie
    - Machines simples : levier, poulie, plan incliné
    - Avantage mécanique
    - Application : puissance moteur, cycliste, ascenseur

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


G = 9.81


# ======================================================================
#  1. Puissance
# ======================================================================

def puissance_constante(F: float, v: float, theta: float = 0) -> float:
    """P = F·v·cos(θ)."""
    return F * v * np.cos(theta)


def puissance_moyenne(W: float, dt: float) -> float:
    """P_moy = W / Δt."""
    return W / dt


def puissance_instantanee(
    F_t: np.ndarray, v_t: np.ndarray,
) -> np.ndarray:
    """P(t) = F(t)·v(t) (composante parallèle)."""
    return F_t * v_t


# ======================================================================
#  2. Rendement
# ======================================================================

def rendement(P_utile: float, P_fournie: float) -> float:
    """η = P_utile / P_fournie (entre 0 et 1)."""
    return P_utile / P_fournie if P_fournie > 0 else 0


def puissance_perdue(P_fournie: float, eta: float) -> float:
    """P_perdue = P_fournie · (1 - η)."""
    return P_fournie * (1 - eta)


# ======================================================================
#  3. Machines simples
# ======================================================================

def levier(F_effort: float = None, F_charge: float = None,
           d_effort: float = None, d_charge: float = None) -> dict:
    """
    Levier : F_e · d_e = F_c · d_c.
    Avantage mécanique AM = F_c / F_e = d_e / d_c.
    """
    if F_effort is not None and d_effort is not None and d_charge is not None:
        F_charge = F_effort * d_effort / d_charge
    elif F_charge is not None and d_effort is not None and d_charge is not None:
        F_effort = F_charge * d_charge / d_effort
    AM = d_effort / d_charge if d_charge > 0 else float("inf")
    return {"F_effort": F_effort, "F_charge": F_charge, "AM": AM}


def plan_incline_machine(m: float, h: float, L: float, mu: float = 0) -> dict:
    """
    Plan incliné comme machine simple.
    Sans frottement : F_effort = mg·h/L.
    AM = mg / F_effort = L/h.
    """
    theta = np.arcsin(h / L)
    F_parallele = m * G * np.sin(theta)
    F_frottement = mu * m * G * np.cos(theta)
    F_effort = F_parallele + F_frottement
    AM_ideal = L / h
    eta = F_parallele / F_effort if F_effort > 0 else 1
    return {
        "F_effort": F_effort,
        "AM_ideal": AM_ideal,
        "rendement": eta,
        "theta_deg": np.degrees(theta),
    }


def poulie_composee(n_brins: int, m: float, eta_par_brin: float = 0.95) -> dict:
    """
    Système de poulies : AM = n (nombre de brins porteurs).
    F_effort = mg / (n · η^n).
    """
    F_ideal = m * G / n_brins
    F_reel = m * G / (n_brins * eta_par_brin**n_brins)
    return {
        "n_brins": n_brins,
        "F_ideal": F_ideal,
        "F_reel": F_reel,
        "AM": n_brins,
        "rendement": eta_par_brin**n_brins,
    }


# ======================================================================
#  4. Applications
# ======================================================================

def puissance_cycliste(m: float, v: float, pente: float = 0,
                        Cd_A: float = 0.5, Cr: float = 0.005) -> dict:
    """
    Puissance d'un cycliste :
        P = P_aero + P_roulement + P_gravité
        P_aero = ½ρCdA·v³
        P_roulement = Cr·mg·v
        P_gravité = mg·sin(θ)·v
    """
    rho = 1.225  # kg/m³
    P_aero = 0.5 * rho * Cd_A * v**3
    P_roulement = Cr * m * G * v
    theta = np.arctan(pente / 100) if pente != 0 else 0
    P_gravite = m * G * np.sin(theta) * v
    P_total = P_aero + P_roulement + P_gravite
    return {
        "P_aero": P_aero, "P_roulement": P_roulement,
        "P_gravite": P_gravite, "P_total": P_total,
    }


def puissance_ascenseur(m_charge: float, m_cabine: float, v: float,
                          m_contrepoids: float = None) -> dict:
    """Puissance pour un ascenseur avec contrepoids optionnel."""
    if m_contrepoids is None:
        m_contrepoids = m_cabine + m_charge / 2
    F_net = (m_cabine + m_charge - m_contrepoids) * G
    P = F_net * v
    return {"F_net": abs(F_net), "P": abs(P), "contrepoids": m_contrepoids}


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_puissance_cycliste(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    v_kmh = np.linspace(5, 50, 200)
    v_ms = v_kmh / 3.6
    m = 80  # cycliste + vélo

    for pente in [0, 3, 6]:
        P = [puissance_cycliste(m, v, pente)["P_total"] for v in v_ms]
        ax.plot(v_kmh, P, linewidth=2, label=f"pente {pente}%")

    ax.axhline(250, color="green", linestyle="--", alpha=0.5, label="amateur (250 W)")
    ax.axhline(400, color="orange", linestyle="--", alpha=0.5, label="pro (400 W)")
    ax.set_xlabel("vitesse (km/h)"); ax.set_ylabel("puissance (W)")
    ax.set_title("Puissance d'un cycliste (80 kg)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 800)
    return ax


def tracer_rendement_poulies(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    n_brins = np.arange(1, 11)
    for eta_brin in [1.0, 0.95, 0.9, 0.85]:
        rendements = [eta_brin**n for n in n_brins]
        ax.plot(n_brins, [r*100 for r in rendements], "o-", markersize=5,
                label=f"η/brin = {eta_brin:.0%}")

    ax.set_xlabel("nombre de brins"); ax.set_ylabel("rendement total (%)")
    ax.set_title("Poulies : plus de brins = plus de pertes")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def tracer_decomposition_puissance(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    v_kmh = np.linspace(5, 45, 200)
    v_ms = v_kmh / 3.6
    m = 80

    P_aero = [puissance_cycliste(m, v, 0)["P_aero"] for v in v_ms]
    P_roul = [puissance_cycliste(m, v, 0)["P_roulement"] for v in v_ms]

    ax.fill_between(v_kmh, 0, P_roul, alpha=0.4, label="roulement", color="green")
    ax.fill_between(v_kmh, P_roul, [a+r for a,r in zip(P_aero, P_roul)],
                     alpha=0.4, label="aérodynamique", color="blue")

    ax.set_xlabel("vitesse (km/h)"); ax.set_ylabel("puissance (W)")
    ax.set_title("Décomposition : aéro domine à haute vitesse ($\\propto v^3$)")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Puissance ===\n")
    print(f"  Voiture : F=3000N, v=30m/s → P = {puissance_constante(3000, 30):.0f} W = {puissance_constante(3000, 30)/1000:.0f} kW")
    print(f"  Haltérophile : 100kg × 2m en 1s → P = {puissance_moyenne(100*G*2, 1):.0f} W")

    print(f"\n=== Machines simples ===\n")
    r = levier(F_effort=50, d_effort=2, d_charge=0.5)
    print(f"  Levier : F_e=50N, d_e=2m, d_c=0.5m → F_c={r['F_charge']:.0f}N, AM={r['AM']:.0f}")

    for n in [1, 2, 4, 6]:
        r = poulie_composee(n, 100)
        print(f"  Poulie {n} brins : F={r['F_reel']:.1f}N (idéal {r['F_ideal']:.1f}N), η={r['rendement']:.1%}")

    print(f"\n=== Plan incliné ===\n")
    for angle in [10, 20, 30]:
        h, L = 1, 1/np.sin(np.radians(angle))
        r = plan_incline_machine(50, h, L, mu=0.1)
        print(f"  θ={angle}° : F={r['F_effort']:.1f}N, AM_idéal={r['AM_ideal']:.1f}, η={r['rendement']:.1%}")

    print(f"\n=== Cycliste (80 kg) ===\n")
    for v_kmh in [15, 25, 35]:
        r = puissance_cycliste(80, v_kmh/3.6, 0)
        print(f"  {v_kmh} km/h plat : P = {r['P_total']:.0f} W "
              f"(aéro {r['P_aero']:.0f}, roul. {r['P_roulement']:.0f})")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tracer_puissance_cycliste(ax=axes[0])
    tracer_decomposition_puissance(ax=axes[1])
    tracer_rendement_poulies(ax=axes[2])
    plt.tight_layout()
    plt.savefig("power_efficiency_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
