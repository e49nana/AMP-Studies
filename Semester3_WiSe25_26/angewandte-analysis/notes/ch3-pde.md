# Chapitre 3 — Partielle Differentialgleichungen

> **Résumé de révision** — Angewandte Analysis für AMP, S3-S4

## 3.1 Classification

EDP linéaire du 2e ordre : $Au_{xx} + Bu_{xy} + Cu_{yy} + \cdots = 0$. Discriminant $D = B^2 - 4AC$.

| Type | $D$ | Exemple | Nature |
|---|---|---|---|
| Elliptique | $D < 0$ | Laplace $\Delta u = 0$ | stationnaire |
| Parabolique | $D = 0$ | Chaleur $u_t = \alpha u_{xx}$ | diffusion |
| Hyperbolique | $D > 0$ | Ondes $u_{tt} = c^2 u_{xx}$ | propagation |

## 3.2 Équation de la chaleur

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

**FTCS (explicite) :** $u_j^{n+1} = u_j^n + r(u_{j+1}^n - 2u_j^n + u_{j-1}^n)$ avec $r = \alpha\Delta t/\Delta x^2$.

**Stabilité :** $r \leq 0.5$ (Von Neumann : $|g(\theta)| = |1 - 4r\sin^2(\theta/2)| \leq 1$).

**BTCS (implicite) :** inconditionnellement stable, résout un système tridiagonal.

**Crank-Nicolson :** $O(\Delta t^2, \Delta x^2)$, inconditionnellement stable — le meilleur compromis.

## 3.3 Équation des ondes

$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$

**Schéma centré :** $u_j^{n+1} = 2u_j^n - u_j^{n-1} + r^2(u_{j+1}^n - 2u_j^n + u_{j-1}^n)$.

**Condition CFL :** $r = c\Delta t/\Delta x \leq 1$.

**d'Alembert :** $u(x,t) = \frac{1}{2}[f(x-ct) + f(x+ct)]$ (deux ondes contra-propagatives).

## 3.4 Équation de Laplace/Poisson

$$\Delta u = 0 \text{ (Laplace)}, \qquad \Delta u = f \text{ (Poisson)}$$

**Jacobi :** $u_{i,j}^{new} = \frac{1}{4}(u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - h^2 f_{ij})$.

**Gauss-Seidel :** utilise les valeurs déjà mises à jour → convergence ~2× plus rapide.

**Propriété de la valeur moyenne :** pour $\Delta u = 0$, $u(P) = $ moyenne des voisins.

## Programmes associés

| Module | Contenu |
|---|---|
| `heat_equation.py` | FTCS, BTCS, stabilité, solution analytique |
| `wave_equation.py` | Schéma centré, CFL, d'Alembert, Dirichlet/Neumann |
| `laplace_equation.py` | Jacobi, Gauss-Seidel, Poisson, valeur moyenne |
| `finite_differences_pde.py` | Classification, Crank-Nicolson, Von Neumann |
