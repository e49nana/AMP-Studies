"""
floating_point.py
=================

Exploration pratique de l'arithmétique flottante IEEE 754.

Référence : Tim Kröger, "Numerische Mathematik 1 für AMP", sections 1.3 et 1.4.

Couvre :
    - Structure IEEE 754 double precision (Beispiel 1.12)
    - Calcul expérimental de ε_mach
    - Overflow et underflow (10^308)
    - Condition des opérations élémentaires (section 1.3.3)
    - Associativité perdue en arithmétique flottante
    - Somme de Kahan (stabilisation de la sommation)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP, WiSe 2024/2025
"""

from __future__ import annotations

import struct

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Propriétés IEEE 754 (Beispiel 1.12)
# ======================================================================

def info_ieee754() -> dict[str, float | int]:
    """
    Propriétés du type float64 (double precision IEEE 754).

    Beispiel 1.12 : ε_mach = 1.110 · 10⁻¹⁶, overflow ≈ 10^308,
    underflow ≈ 10^-308.
    """
    info = np.finfo(np.float64)
    return {
        "eps_mach": float(info.eps),
        "plus_petit_normalise": float(info.tiny),
        "plus_grand": float(info.max),
        "mantisse_bits": info.nmant,
        "exposant_min": info.minexp,
        "exposant_max": info.maxexp,
        "precision_decimale": info.precision,
    }


# ======================================================================
#  2. Calcul expérimental de ε_mach (Übung 1.11)
# ======================================================================

def epsilon_machine_experimental() -> float:
    """
    Calcule ε_mach expérimentalement :
    le plus petit ε > 0 tel que fl(1 + ε) > 1.

    Méthode : on divise par 2 jusqu'à ce que 1 + ε == 1 en flottant.
    """
    eps = 1.0
    while (1.0 + eps) > 1.0:
        eps_prev = eps
        eps /= 2.0
    return eps_prev


def compter_chiffres_significatifs(exact: float, approx: float) -> int:
    """Nombre de chiffres décimaux corrects."""
    if exact == 0:
        return 0 if approx != 0 else 16
    err_rel = abs(approx - exact) / abs(exact)
    if err_rel == 0:
        return 16
    return max(0, int(-np.log10(err_rel)))


# ======================================================================
#  3. Représentation binaire
# ======================================================================

def float_to_bits(x: float) -> str:
    """Représentation binaire IEEE 754 d'un float64."""
    packed = struct.pack(">d", x)
    bits = "".join(f"{byte:08b}" for byte in packed)
    return f"{bits[0]} | {bits[1:12]} | {bits[12:]}"


def analyser_float(x: float) -> dict[str, str | int | float]:
    """Décompose un float64 en signe, exposant, mantisse."""
    packed = struct.pack(">d", x)
    bits = "".join(f"{byte:08b}" for byte in packed)
    signe = int(bits[0])
    exposant_brut = int(bits[1:12], 2)
    exposant = exposant_brut - 1023  # biais IEEE 754
    mantisse_bits = bits[12:]
    mantisse = 1.0 + sum(int(b) * 2**(-i - 1)
                         for i, b in enumerate(mantisse_bits))
    return {
        "valeur": x,
        "signe": (-1)**signe,
        "exposant_brut": exposant_brut,
        "exposant": exposant,
        "mantisse_1.xxx": mantisse,
        "bits": f"{bits[0]} | {bits[1:12]} | {bits[12:]}",
        "reconstruction": (-1)**signe * mantisse * 2**exposant,
    }


# ======================================================================
#  4. Associativité perdue
# ======================================================================

def demo_non_associativite() -> list[tuple[str, float]]:
    """
    Montre que l'addition flottante n'est pas associative :
    (a + b) + c ≠ a + (b + c) dans certains cas.
    """
    cas = []
    # Cas classique : grand + petit + petit
    a = 1e16
    b = 1.0
    c = -1e16
    cas.append(("(a+b)+c", (a + b) + c))
    cas.append(("a+(b+c)", a + (b + c)))

    # Cas somme de petits termes
    n = 10**7
    termes = np.full(n, 1e-7)
    somme_gauche = 0.0
    for t in termes:
        somme_gauche += t
    somme_droite = 0.0
    for t in reversed(termes):
        somme_droite += t
    cas.append(("somme gauche→droite", somme_gauche))
    cas.append(("somme droite→gauche", somme_droite))
    cas.append(("valeur exacte", 1.0))

    return cas


# ======================================================================
#  5. Somme de Kahan (algorithme de sommation compensée)
# ======================================================================

def somme_naive(termes: np.ndarray) -> float:
    """Somme simple — accumule les erreurs d'arrondi."""
    s = 0.0
    for t in termes:
        s += t
    return s


def somme_kahan(termes: np.ndarray) -> float:
    """
    Somme de Kahan : compense les erreurs d'arrondi en gardant
    un terme correcteur c.

    L'erreur ne croît plus en O(n · ε_mach) mais reste O(ε_mach)
    indépendamment de n.
    """
    s = 0.0
    c = 0.0  # compensation
    for t in termes:
        y = t - c
        temp = s + y
        c = (temp - s) - y  # erreur d'arrondi de cette étape
        s = temp
    return s


# ======================================================================
#  6. Condition des opérations élémentaires (section 1.3.3)
# ======================================================================

def condition_operation(
    operation: str, x1: float, x2: float,
) -> dict[str, float]:
    """
    Calcule les fehlerverstärkende Faktoren φ₁, φ₂ et les conditions
    relatives pour les 4 opérations élémentaires.
    """
    if operation == "+":
        f = x1 + x2
        phi1 = x1 / f if f != 0 else float("inf")
        phi2 = x2 / f if f != 0 else float("inf")
    elif operation == "-":
        f = x1 - x2
        phi1 = x1 / f if f != 0 else float("inf")
        phi2 = -x2 / f if f != 0 else float("inf")
    elif operation == "*":
        phi1, phi2 = 1.0, 1.0
    elif operation == "/":
        phi1, phi2 = 1.0, -1.0
    else:
        raise ValueError(f"Opération inconnue : {operation}")

    return {
        "φ₁": phi1,
        "φ₂": phi2,
        "cond_rel_∞": abs(phi1) + abs(phi2),
        "cond_rel_1": max(abs(phi1), abs(phi2)),
    }


# ======================================================================
#  7. Tracé : erreur de sommation
# ======================================================================

def tracer_erreur_sommation(ax: plt.Axes | None = None) -> plt.Axes:
    """Compare l'erreur de la somme naïve vs Kahan en fonction de n."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ns = np.logspace(2, 7, 20, dtype=int)
    err_naive, err_kahan = [], []
    for n in ns:
        termes = np.full(n, 1.0 / n)
        exact = 1.0
        err_naive.append(abs(somme_naive(termes) - exact))
        err_kahan.append(abs(somme_kahan(termes) - exact))

    ax.loglog(ns, err_naive, "rs-", label="somme naïve", markersize=4)
    ax.loglog(ns, [max(e, 1e-17) for e in err_kahan], "bo-",
              label="somme de Kahan", markersize=4)
    eps = np.finfo(float).eps
    ax.loglog(ns, ns * eps, "r--", alpha=0.5, label="$n \\cdot \\varepsilon_{mach}$")
    ax.axhline(eps, color="b", linestyle=":", alpha=0.5, label="$\\varepsilon_{mach}$")
    ax.set_xlabel("nombre de termes $n$")
    ax.set_ylabel("erreur $|S_{calc} - 1|$")
    ax.set_title("Somme naïve vs Kahan : l'erreur croît avec $n$ sans compensation")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_densite_flottants(ax: plt.Axes | None = None) -> plt.Axes:
    """Visualise la densité non uniforme des flottants autour de 0."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    # Flottants entre 2^k et 2^{k+1} ont le même espacement = 2^{k-52}
    exponents = range(-5, 5)
    for k in exponents:
        start = 2.0**k
        spacing = 2.0**(k - 52)
        # Montrer quelques flottants dans cet intervalle
        pts = start + spacing * np.arange(0, min(20, int(start / spacing)))
        ax.plot(pts, np.zeros_like(pts), "|", color="tab:blue", markersize=10)

    ax.set_xlabel("$x$")
    ax.set_title("Densité des flottants : plus serrés près de 0, plus espacés loin")
    ax.set_yticks([])
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    return ax


# ======================================================================
#  Démo
# ======================================================================

if __name__ == "__main__":
    print("=== IEEE 754 double precision (Beispiel 1.12) ===")
    for k, v in info_ieee754().items():
        print(f"  {k:25s} : {v}")

    print(f"\n=== ε_mach expérimental (Übung 1.11) ===")
    eps_exp = epsilon_machine_experimental()
    eps_theo = np.finfo(float).eps
    print(f"  ε_mach expérimental  = {eps_exp:.6e}")
    print(f"  ε_mach np.finfo      = {eps_theo:.6e}")
    print(f"  ratio                = {eps_exp / eps_theo:.6f}")

    print(f"\n=== Représentation binaire ===")
    for x in [1.0, 0.1, -3.14, 0.0]:
        info = analyser_float(x)
        print(f"  {x:>8} : {info['bits']}")
        print(f"           reconstruction = {info['reconstruction']}")

    print(f"\n=== Non-associativité ===")
    for label, val in demo_non_associativite():
        print(f"  {label:25s} = {val}")

    print(f"\n=== Condition des opérations (section 1.3.3) ===")
    print(f"  {'op':>3} | {'x₁':>10} {'x₂':>10} | {'cond_rel_∞':>12}")
    print("  " + "-" * 42)
    for op, x1, x2 in [
        ("+", 1.0, 2.0), ("+", 1.0, -0.999),
        ("-", 1.001, 1.0), ("*", 3.0, 7.0), ("/", 10.0, 3.0),
    ]:
        c = condition_operation(op, x1, x2)
        print(f"  {op:>3} | {x1:>10.3f} {x2:>10.3f} | {c['cond_rel_∞']:>12.2f}")

    print("\n=== Tracés ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_erreur_sommation(ax=axes[0])
    tracer_densite_flottants(ax=axes[1])
    plt.tight_layout()
    plt.savefig("floating_point_demo.png", dpi=120)
    print("Figure sauvegardée : floating_point_demo.png")
