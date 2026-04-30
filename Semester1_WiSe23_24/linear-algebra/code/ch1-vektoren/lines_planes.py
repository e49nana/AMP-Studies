"""
lines_planes.py
===============

Droites et plans en R³ : représentations et intersections.

Couvre :
    - Droite paramétrique : r(t) = p + t·d
    - Plan paramétrique : r(s,t) = p + s·u + t·v
    - Plan cartésien : n·x = n·p (ou ax + by + cz = d)
    - Intersection droite-plan
    - Intersection deux plans → droite
    - Distance point-plan, point-droite

Auteur : Emmanuel Nanan — TH Nürnberg, AMP
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class Droite3D:
    """Droite r(t) = point + t · direction."""
    point: np.ndarray
    direction: np.ndarray

    def evaluer(self, t: float) -> np.ndarray:
        return self.point + t * self.direction

    def __repr__(self) -> str:
        return f"Droite(point={self.point}, dir={self.direction})"


@dataclass
class Plan3D:
    """Plan défini par un point et une normale."""
    point: np.ndarray
    normale: np.ndarray

    @classmethod
    def from_parametrique(cls, point: np.ndarray, u: np.ndarray, v: np.ndarray) -> Plan3D:
        """Crée un plan à partir de la forme paramétrique r = p + su + tv."""
        normale = np.cross(u, v)
        return cls(point=point, normale=normale)

    @classmethod
    def from_equation(cls, a: float, b: float, c: float, d: float) -> Plan3D:
        """Crée un plan à partir de ax + by + cz = d."""
        n = np.array([a, b, c], dtype=float)
        p = np.zeros(3)
        if a != 0: p[0] = d / a
        elif b != 0: p[1] = d / b
        elif c != 0: p[2] = d / c
        return cls(point=p, normale=n)

    @property
    def equation(self) -> str:
        a, b, c = self.normale
        d = np.dot(self.normale, self.point)
        return f"{a:.2g}x + {b:.2g}y + {c:.2g}z = {d:.2g}"

    def __repr__(self) -> str:
        return f"Plan({self.equation})"


def intersection_droite_plan(d: Droite3D, p: Plan3D) -> np.ndarray | None:
    """
    Point d'intersection droite-plan.

    t = ⟨n, p_plan - p_droite⟩ / ⟨n, dir⟩.
    Renvoie None si parallèles (⟨n, dir⟩ = 0).
    """
    denom = np.dot(p.normale, d.direction)
    if abs(denom) < 1e-12:
        return None  # parallèle
    t = np.dot(p.normale, p.point - d.point) / denom
    return d.evaluer(t)


def distance_point_plan(point: np.ndarray, plan: Plan3D) -> float:
    """Distance = |⟨n, point - p⟩| / ||n||."""
    return abs(np.dot(plan.normale, point - plan.point)) / np.linalg.norm(plan.normale)


def distance_point_droite(point: np.ndarray, droite: Droite3D) -> float:
    """Distance = ||AP × d|| / ||d|| avec A = droite.point."""
    AP = point - droite.point
    return np.linalg.norm(np.cross(AP, droite.direction)) / np.linalg.norm(droite.direction)


def intersection_deux_plans(p1: Plan3D, p2: Plan3D) -> Droite3D | None:
    """
    Intersection de deux plans = une droite.
    Direction = n₁ × n₂. Point = résoudre le système.
    Renvoie None si parallèles.
    """
    d = np.cross(p1.normale, p2.normale)
    if np.linalg.norm(d) < 1e-12:
        return None  # parallèles

    # Trouver un point sur la droite d'intersection
    # On fixe la composante la plus grande de d à 0
    i = np.argmax(np.abs(d))
    A = np.zeros((2, 3))
    A[0] = p1.normale
    A[1] = p2.normale
    b = np.array([np.dot(p1.normale, p1.point), np.dot(p2.normale, p2.point)])

    # Résoudre le sous-système 2×2 (sans la colonne i)
    cols = [j for j in range(3) if j != i]
    A2 = A[:, cols]
    x2 = np.linalg.solve(A2, b)
    point = np.zeros(3)
    point[cols[0]] = x2[0]
    point[cols[1]] = x2[1]

    return Droite3D(point=point, direction=d)


def tracer_plan_et_droite(
    plan: Plan3D, droite: Droite3D, ax: plt.Axes | None = None,
) -> plt.Axes:
    """Visualise un plan et une droite en 3D."""
    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

    # Plan comme surface
    xx, yy = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))
    n = plan.normale
    if abs(n[2]) > 1e-10:
        d = np.dot(n, plan.point)
        zz = (d - n[0]*xx - n[1]*yy) / n[2]
        ax.plot_surface(xx, yy, zz, alpha=0.3, color="cyan")

    # Droite
    ts = np.linspace(-3, 3, 50)
    pts = np.array([droite.evaluer(t) for t in ts])
    ax.plot(pts[:,0], pts[:,1], pts[:,2], "r-", linewidth=2, label="Droite")

    # Intersection
    inter = intersection_droite_plan(droite, plan)
    if inter is not None:
        ax.scatter(*inter, color="black", s=100, zorder=5, label=f"Intersection {np.round(inter,2)}")

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend()
    ax.set_title("Droite ∩ Plan")
    return ax


if __name__ == "__main__":
    print("=== Droite et Plan ===")
    d = Droite3D(point=np.array([1,0,0], dtype=float),
                 direction=np.array([0,1,1], dtype=float))
    p = Plan3D(point=np.array([0,0,2], dtype=float),
               normale=np.array([0,0,1], dtype=float))
    print(f"Droite : {d}")
    print(f"Plan   : {p}")

    inter = intersection_droite_plan(d, p)
    print(f"Intersection : {inter}")

    print(f"\n=== Distances ===")
    Q = np.array([0, 0, 5], dtype=float)
    print(f"Distance ({Q}) au plan : {distance_point_plan(Q, p):.4f}")
    print(f"Distance ({Q}) à la droite : {distance_point_droite(Q, d):.4f}")

    print(f"\n=== Intersection de deux plans ===")
    p1 = Plan3D.from_equation(1, 0, 0, 2)  # x = 2
    p2 = Plan3D.from_equation(0, 1, 0, 3)  # y = 3
    droite_inter = intersection_deux_plans(p1, p2)
    print(f"Plan 1 : {p1}")
    print(f"Plan 2 : {p2}")
    print(f"Intersection : {droite_inter}")

    tracer_plan_et_droite(p, d)
    plt.savefig("lines_planes_demo.png", dpi=120)
    print("\nFigure sauvegardée.")
