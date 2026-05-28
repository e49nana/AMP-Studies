"""
joint_distributions.py
======================

Lois jointes, marginales, covariance et corrélation.

Couvre :
    - Loi jointe P(X=x, Y=y) et tableau de contingence
    - Lois marginales par sommation
    - Covariance : Cov(X,Y) = E[XY] - E[X]E[Y]
    - Corrélation : ρ = Cov(X,Y) / (σ_X σ_Y)
    - Indépendance : P(X=x, Y=y) = P(X=x)·P(Y=y)
    - Variance d'une somme : Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y)

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


class LoiJointe:
    """Loi jointe discrète de (X, Y)."""

    def __init__(self, x_vals: np.ndarray, y_vals: np.ndarray, probs: np.ndarray) -> None:
        """probs[i, j] = P(X = x_vals[i], Y = y_vals[j])."""
        self.x_vals = np.asarray(x_vals, dtype=float)
        self.y_vals = np.asarray(y_vals, dtype=float)
        self.P = np.asarray(probs, dtype=float)

    def marginale_X(self) -> np.ndarray:
        """P(X = xᵢ) = Σⱼ P(X=xᵢ, Y=yⱼ)."""
        return self.P.sum(axis=1)

    def marginale_Y(self) -> np.ndarray:
        """P(Y = yⱼ) = Σᵢ P(X=xᵢ, Y=yⱼ)."""
        return self.P.sum(axis=0)

    def esperance_X(self) -> float:
        return float(np.dot(self.x_vals, self.marginale_X()))

    def esperance_Y(self) -> float:
        return float(np.dot(self.y_vals, self.marginale_Y()))

    def esperance_XY(self) -> float:
        """E[XY] = Σᵢⱼ xᵢyⱼ P(X=xᵢ, Y=yⱼ)."""
        return float(np.sum(np.outer(self.x_vals, self.y_vals) * self.P))

    def covariance(self) -> float:
        """Cov(X,Y) = E[XY] - E[X]E[Y]."""
        return self.esperance_XY() - self.esperance_X() * self.esperance_Y()

    def variance_X(self) -> float:
        pX = self.marginale_X()
        EX = self.esperance_X()
        return float(np.dot(self.x_vals**2, pX)) - EX**2

    def variance_Y(self) -> float:
        pY = self.marginale_Y()
        EY = self.esperance_Y()
        return float(np.dot(self.y_vals**2, pY)) - EY**2

    def correlation(self) -> float:
        """ρ(X,Y) = Cov(X,Y) / (σ_X · σ_Y)."""
        sx = np.sqrt(self.variance_X())
        sy = np.sqrt(self.variance_Y())
        if sx * sy == 0:
            return 0
        return self.covariance() / (sx * sy)

    def est_independant(self, tol: float = 1e-10) -> bool:
        """X ⊥ Y ssi P(X,Y) = P(X)·P(Y) pour tout (x,y)."""
        produit = np.outer(self.marginale_X(), self.marginale_Y())
        return bool(np.allclose(self.P, produit, atol=tol))

    def variance_somme(self) -> float:
        """Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y)."""
        return self.variance_X() + self.variance_Y() + 2 * self.covariance()

    def afficher_tableau(self) -> None:
        """Affiche le tableau de contingence."""
        pX = self.marginale_X()
        pY = self.marginale_Y()
        print(f"  {'':>6} |", end="")
        for y in self.y_vals:
            print(f" Y={y:>4.1f} |", end="")
        print(f" P(X)")
        print("  " + "-" * (10 + 10 * len(self.y_vals)))
        for i, x in enumerate(self.x_vals):
            print(f"  X={x:>3.1f} |", end="")
            for j in range(len(self.y_vals)):
                print(f" {self.P[i,j]:>6.4f} |", end="")
            print(f" {pX[i]:.4f}")
        print("  " + "-" * (10 + 10 * len(self.y_vals)))
        print(f"  {'P(Y)':>6} |", end="")
        for j in range(len(self.y_vals)):
            print(f" {pY[j]:>6.4f} |", end="")
        print(f" {np.sum(self.P):.4f}")


# ======================================================================
#  Tracés
# ======================================================================

def tracer_correlation_exemples(ax: plt.Axes | None = None) -> plt.Axes:
    """Montre des nuages de points avec différentes corrélations."""
    if ax is None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    else:
        axes = [ax]*4

    rng = np.random.default_rng(42)
    n = 300

    for ax_i, rho_target, nom in zip(axes,
        [0.95, 0.5, 0, -0.8],
        ["ρ ≈ 0.95", "ρ ≈ 0.5", "ρ ≈ 0", "ρ ≈ -0.8"]):

        if rho_target == 0:
            x = rng.normal(0, 1, n)
            y = rng.normal(0, 1, n)
        else:
            mean = [0, 0]
            cov = [[1, rho_target], [rho_target, 1]]
            data = rng.multivariate_normal(mean, cov, n)
            x, y = data[:, 0], data[:, 1]

        rho_obs = np.corrcoef(x, y)[0, 1]
        ax_i.scatter(x, y, s=5, alpha=0.5)
        ax_i.set_title(f"{nom}\n(obs: {rho_obs:.2f})")
        ax_i.set_aspect("equal")
        ax_i.set_xlim(-3.5, 3.5); ax_i.set_ylim(-3.5, 3.5)
        ax_i.grid(True, alpha=0.3)

    return axes[0]


def tracer_loi_jointe_heatmap(loi: LoiJointe, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(loi.P, origin="lower", cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(loi.y_vals)))
    ax.set_xticklabels([f"{y:.0f}" for y in loi.y_vals])
    ax.set_yticks(range(len(loi.x_vals)))
    ax.set_yticklabels([f"{x:.0f}" for x in loi.x_vals])
    ax.set_xlabel("$Y$"); ax.set_ylabel("$X$")
    ax.set_title(f"Loi jointe (ρ = {loi.correlation():.3f})")
    plt.colorbar(im, ax=ax, label="$P(X, Y)$")
    return ax


if __name__ == "__main__":
    print("=== Loi jointe (deux dés) ===\n")
    # Exemple : X = dé 1, Y = dé 2 (indépendants)
    vals = np.arange(1, 7, dtype=float)
    P_indep = np.full((6, 6), 1/36)
    loi_indep = LoiJointe(vals, vals, P_indep)
    loi_indep.afficher_tableau()
    print(f"\n  E[X] = {loi_indep.esperance_X():.4f}")
    print(f"  Cov(X,Y) = {loi_indep.covariance():.6f}")
    print(f"  ρ(X,Y) = {loi_indep.correlation():.6f}")
    print(f"  Indépendants ? {loi_indep.est_independant()} ✓")

    print(f"\n=== Loi jointe (corrélée) ===\n")
    # X = note exam 1, Y = note exam 2 (corrélés)
    x_vals = np.array([1, 2, 3, 4, 5], dtype=float)
    y_vals = np.array([1, 2, 3, 4, 5], dtype=float)
    P_corr = np.array([
        [0.10, 0.04, 0.02, 0.01, 0.01],
        [0.04, 0.10, 0.04, 0.02, 0.01],
        [0.02, 0.04, 0.10, 0.04, 0.02],
        [0.01, 0.02, 0.04, 0.10, 0.04],
        [0.01, 0.01, 0.02, 0.04, 0.10],
    ])
    P_corr /= P_corr.sum()  # normaliser
    loi_corr = LoiJointe(x_vals, y_vals, P_corr)
    print(f"  E[X] = {loi_corr.esperance_X():.4f}")
    print(f"  E[Y] = {loi_corr.esperance_Y():.4f}")
    print(f"  Cov(X,Y) = {loi_corr.covariance():.4f}")
    print(f"  ρ(X,Y) = {loi_corr.correlation():.4f}")
    print(f"  Indépendants ? {loi_corr.est_independant()}")

    print(f"\n=== Var(X+Y) ===\n")
    print(f"  Var(X) = {loi_corr.variance_X():.4f}")
    print(f"  Var(Y) = {loi_corr.variance_Y():.4f}")
    print(f"  2·Cov  = {2*loi_corr.covariance():.4f}")
    print(f"  Var(X+Y) = {loi_corr.variance_somme():.4f}")
    print(f"  Si indép: Var(X)+Var(Y) = {loi_corr.variance_X()+loi_corr.variance_Y():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tracer_loi_jointe_heatmap(loi_corr, ax=axes[0])
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 3.5))
    tracer_correlation_exemples(axes2[0])
    # Save both
    fig.tight_layout()
    fig.savefig("joint_distributions_demo.png", dpi=120)
    fig2.tight_layout()
    fig2.savefig("correlation_examples.png", dpi=120)
    print("\nFigures sauvegardées.")
