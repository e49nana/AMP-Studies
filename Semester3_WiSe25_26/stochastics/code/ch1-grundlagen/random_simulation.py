"""
random_simulation.py
====================

Simulation Monte Carlo et loi des grands nombres.

Couvre :
    - Loi des grands nombres : X̄_n → E[X] quand n → ∞
    - Monte Carlo : estimation de π par points aléatoires
    - Monte Carlo : intégration numérique stochastique
    - Simulation de marches aléatoires
    - Convergence : erreur ∝ 1/√n (lente mais universelle)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
#  1. Loi des grands nombres
# ======================================================================

def loi_grands_nombres_de(n_lancers: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Lance un dé n fois et trace la moyenne cumulée.
    E[X] = 3.5 pour un dé équilibré.
    """
    rng = np.random.default_rng(seed)
    lancers = rng.integers(1, 7, n_lancers)
    moyennes = np.cumsum(lancers) / np.arange(1, n_lancers + 1)
    return np.arange(1, n_lancers + 1), moyennes


def loi_grands_nombres_piece(n_lancers: int, p: float = 0.5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Fréquence relative de face → p."""
    rng = np.random.default_rng(seed)
    lancers = rng.random(n_lancers) < p
    freq = np.cumsum(lancers) / np.arange(1, n_lancers + 1)
    return np.arange(1, n_lancers + 1), freq


# ======================================================================
#  2. Monte Carlo : estimation de π
# ======================================================================

def monte_carlo_pi(n_points: int, seed: int = 42) -> tuple[float, np.ndarray, np.ndarray]:
    """
    π/4 = aire du quart de cercle unité.
    On tire des points uniformes dans [0,1]² et on compte ceux dans le cercle.
    π ≈ 4 · (points dans le cercle) / (total).
    """
    rng = np.random.default_rng(seed)
    x = rng.random(n_points)
    y = rng.random(n_points)
    inside = x**2 + y**2 <= 1
    pi_est = 4 * np.cumsum(inside) / np.arange(1, n_points + 1)
    return pi_est[-1], x, y


def convergence_pi(n_max: int = 100_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Erreur |π_est - π| en fonction de n."""
    rng = np.random.default_rng(seed)
    x = rng.random(n_max)
    y = rng.random(n_max)
    inside = x**2 + y**2 <= 1
    pi_est = 4 * np.cumsum(inside) / np.arange(1, n_max + 1)
    ns = np.arange(1, n_max + 1)
    return ns, np.abs(pi_est - np.pi)


# ======================================================================
#  3. Monte Carlo : intégration
# ======================================================================

def monte_carlo_integral(
    f, a: float, b: float, n: int, seed: int = 42,
) -> tuple[float, float]:
    """
    ∫_a^b f(x) dx ≈ (b-a) · (1/n) Σ f(xᵢ) avec xᵢ ~ U(a,b).
    Erreur ∝ σ/√n.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, n)
    fx = np.array([f(xi) for xi in x])
    I = (b - a) * np.mean(fx)
    err = (b - a) * np.std(fx) / np.sqrt(n)
    return I, err


# ======================================================================
#  4. Marche aléatoire
# ======================================================================

def marche_aleatoire_1d(n_steps: int, seed: int = 42) -> np.ndarray:
    """S_n = Σ Xᵢ avec Xᵢ = ±1 équiprobable."""
    rng = np.random.default_rng(seed)
    steps = rng.choice([-1, 1], n_steps)
    return np.concatenate([[0], np.cumsum(steps)])


def marche_aleatoire_2d(n_steps: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Marche aléatoire sur Z² (4 directions)."""
    rng = np.random.default_rng(seed)
    directions = rng.integers(0, 4, n_steps)
    dx = np.array([1, -1, 0, 0])
    dy = np.array([0, 0, 1, -1])
    x = np.concatenate([[0], np.cumsum(dx[directions])])
    y = np.concatenate([[0], np.cumsum(dy[directions])])
    return x, y


# ======================================================================
#  5. Tracés
# ======================================================================

def tracer_lgn(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for seed in range(5):
        ns, moyennes = loi_grands_nombres_de(10000, seed)
        ax.semilogx(ns, moyennes, linewidth=1, alpha=0.5)

    ax.axhline(3.5, color="red", linewidth=2, linestyle="--", label="$E[X] = 3.5$")
    ax.set_xlabel("nombre de lancers $n$"); ax.set_ylabel("moyenne $\\bar{X}_n$")
    ax.set_title("Loi des grands nombres (dé à 6 faces)")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(2.5, 4.5)
    return ax


def tracer_monte_carlo_pi(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    n = 5000
    pi_est, x, y = monte_carlo_pi(n)
    inside = x**2 + y**2 <= 1

    ax.scatter(x[inside], y[inside], s=1, c="blue", alpha=0.3)
    ax.scatter(x[~inside], y[~inside], s=1, c="red", alpha=0.3)
    theta = np.linspace(0, np.pi/2, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2)
    ax.set_aspect("equal")
    ax.set_title(f"Monte Carlo π ≈ {pi_est:.4f} ($n = {n}$)")
    ax.grid(True, alpha=0.3)
    return ax


def tracer_convergence_mc(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ns, errs = convergence_pi(100_000)
    ax.loglog(ns, errs, "b-", alpha=0.3, linewidth=0.5)
    # Moyenne glissante
    window = 100
    smooth = np.convolve(errs, np.ones(window)/window, mode="valid")
    ax.loglog(ns[window-1:], smooth, "b-", linewidth=2, label="erreur (lissée)")
    ax.loglog(ns, 2/np.sqrt(ns), "r--", linewidth=2, label="$O(1/\\sqrt{n})$")
    ax.set_xlabel("$n$"); ax.set_ylabel("$|\\hat{\\pi} - \\pi|$")
    ax.set_title("Convergence Monte Carlo : $O(1/\\sqrt{n})$")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    return ax


def tracer_marches(ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for seed in range(8):
        walk = marche_aleatoire_1d(1000, seed)
        ax.plot(walk, linewidth=1, alpha=0.5)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("pas $n$"); ax.set_ylabel("position $S_n$")
    ax.set_title("Marches aléatoires 1D ($S_n \\sim \\sqrt{n}$)")
    ax.grid(True, alpha=0.3)
    return ax


if __name__ == "__main__":
    print("=== Loi des grands nombres (dé) ===\n")
    ns, moyennes = loi_grands_nombres_de(100_000)
    for n in [10, 100, 1000, 10000, 100000]:
        print(f"  n = {n:>6} : X̄ = {moyennes[n-1]:.4f} (E[X] = 3.5)")

    print(f"\n=== Monte Carlo π ===\n")
    for n in [100, 1000, 10000, 100000]:
        pi_est, _, _ = monte_carlo_pi(n)
        print(f"  n = {n:>6} : π ≈ {pi_est:.6f} (err = {abs(pi_est - np.pi):.4f})")

    print(f"\n=== Monte Carlo intégration ===\n")
    print(f"  ∫₀¹ x² dx = 1/3 :")
    for n in [100, 1000, 10000, 100000]:
        I, err = monte_carlo_integral(lambda x: x**2, 0, 1, n)
        print(f"    n={n:>6} : I = {I:.6f} ± {err:.6f}")

    print(f"\n  ∫₀^π sin(x) dx = 2 :")
    I, err = monte_carlo_integral(np.sin, 0, np.pi, 100000)
    print(f"    n=100000 : I = {I:.6f} ± {err:.6f}")

    print(f"\n=== Marche aléatoire ===\n")
    walk = marche_aleatoire_1d(10000)
    print(f"  1000 pas : position finale = {walk[1000]}")
    print(f"  10000 pas : position finale = {walk[-1]}")
    print(f"  E[S_n] = 0, E[|S_n|] ≈ √(2n/π) ≈ {np.sqrt(2*10000/np.pi):.0f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tracer_lgn(ax=axes[0, 0])
    tracer_monte_carlo_pi(ax=axes[0, 1])
    tracer_convergence_mc(ax=axes[1, 0])
    tracer_marches(ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig("random_simulation_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
