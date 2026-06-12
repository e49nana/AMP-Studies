# Chapitre 1 — Mehrdimensionale Analysis

> **Résumé de révision** — Angewandte Analysis für AMP, S3-S4

## 1.1 Dérivées partielles

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x + h e_i) - f(x)}{h}$$

**Gradient :** $\nabla f = \left(\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}\right)$ — direction de plus forte croissance.

**Hessienne :** $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ — matrice symétrique (Schwarz).

**Jacobienne :** pour $F : \mathbb{R}^n \to \mathbb{R}^m$, $J_{ij} = \frac{\partial F_i}{\partial x_j}$.

**Différentielle totale :** $df = \nabla f \cdot dx$ (approximation linéaire).

## 1.2 Points critiques

$\nabla f(x_0) = 0$ → point critique. Classification par la Hessienne :

| Condition (2D) | $D = f_{xx}f_{yy} - f_{xy}^2$ | Type |
|---|---|---|
| $D > 0$, $f_{xx} > 0$ | — | minimum local |
| $D > 0$, $f_{xx} < 0$ | — | maximum local |
| $D < 0$ | — | point selle |
| $D = 0$ | — | indéterminé |

En dim $n$ : toutes les valeurs propres de $H$ positives → min, toutes négatives → max, signes mixtes → selle.

## 1.3 Optimisation multivariable

| Méthode | Formule | Convergence |
|---|---|---|
| Gradient descent | $x^+ = x - \alpha \nabla f$ | linéaire |
| Newton | $x^+ = x - H^{-1} \nabla f$ | quadratique |
| Lagrange | $\nabla f = \lambda \nabla g$ sous $g = 0$ | — |

**Lagrange :** optimiser $f$ sous contrainte $g(x) = 0$ → résoudre $\nabla f = \lambda \nabla g$ et $g = 0$.

## 1.4 Calcul vectoriel

| Opérateur | Formule | Interprétation |
|---|---|---|
| Gradient | $\nabla f$ | direction de croissance |
| Divergence | $\text{div}\,\vec{F} = \nabla \cdot \vec{F}$ | source (+) ou puits (−) |
| Rotationnel | $\text{rot}\,\vec{F} = \nabla \times \vec{F}$ | tourbillon |
| Laplacien | $\Delta f = \nabla^2 f = \text{div}(\text{grad}\,f)$ | diffusion |

**Champ conservatif :** $\vec{F} = \nabla \phi \Rightarrow \text{rot}\,\vec{F} = 0$.

## 1.5 Intégrales multiples

$$\iint f\,dA, \qquad \iiint f\,dV$$

| Coordonnées | Jacobien |
|---|---|
| Polaires $(r, \theta)$ | $dA = r\,dr\,d\theta$ |
| Cylindriques $(r, \theta, z)$ | $dV = r\,dr\,d\theta\,dz$ |
| Sphériques $(r, \phi, \theta)$ | $dV = r^2 \sin\phi\,dr\,d\phi\,d\theta$ |

**Applications :** aire, volume, centre de masse, moment d'inertie.

## Programmes associés

| Module | Contenu |
|---|---|
| `partial_derivatives.py` | Gradient, Hessienne, Jacobienne, classification |
| `multivariable_optimization.py` | GD, Newton, Lagrange, Rosenbrock |
| `vector_calculus.py` | div, rot, Laplacien, champs conservatifs |
| `multiple_integrals.py` | Intégrales doubles/triples, coordonnées |
