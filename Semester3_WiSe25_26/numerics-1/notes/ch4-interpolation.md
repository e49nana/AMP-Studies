# Chapitre 4 — Interpolation

> **Résumé de révision** — Kröger, *Numerische Mathematik 1 für AMP*, §4.1–4.3

## 4.1 Problème

Étant donnés $n+1$ points $(x_0, y_0), \dots, (x_n, y_n)$ avec $x_i$ distincts, trouver $p \in \mathcal{P}_n$ tel que $p(x_i) = y_i$.

**Satz 4.2** — Il existe **un unique** tel polynôme.

## 4.2 Trois représentations du même polynôme

### Lagrange (§4.2.2)

$$p(x) = \sum_{i=0}^n y_i \, L_i^n(x), \qquad L_i^n(x) = \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}$$

Propriété : $L_i(x_j) = \delta_{ij}$. Coût : $O(n^2)$ par point.

### Neville-Aitken (Lemma 4.6)

Formule récursive :
$$p_{j,k}(x) = \frac{(x_k - x) \, p_{j,k-1}(x) + (x - x_j) \, p_{j+1,k}(x)}{x_k - x_j}$$

Avantage : pas besoin des coefficients, évalue directement $p(x)$. Coût : $O(n^2)$.

### Newton (Satz 4.8)

$$p(x) = \sum_{k=0}^n c_k \prod_{j=0}^{k-1} (x - x_j), \qquad c_k = f[x_0, \dots, x_k]$$

**Différences divisées (formule 4.5)** :
$$f[x_j, \dots, x_{j+k}] = \frac{f[x_{j+1}, \dots, x_{j+k}] - f[x_j, \dots, x_{j+k-1}]}{x_{j+k} - x_j}$$

**Évaluation par Horner (formule 4.6)** : $O(n)$ par point.

**Avantage clé** : ajouter un point ne coûte que $O(n)$ (pas de recalcul).

### Comparaison

| Méthode | Construction | Évaluation | Ajout d'un point |
|---|---|---|---|
| Lagrange | $O(1)$ | $O(n^2)$ | $O(n^2)$ recalcul |
| Neville | $O(1)$ | $O(n^2)$ | $O(n)$ |
| Newton | $O(n^2)$ | $O(n)$ | $O(n)$ |

## 4.2.6 Borne d'erreur (Satz 4.10)

$$|p(x) - f(x)| \leq \frac{|\omega(x)|}{(n+1)!} \cdot \max_{\xi \in [a,b]} |f^{(n+1)}(\xi)|$$

avec le **polynôme nodal** $\omega(x) = \prod_{i=0}^n (x - x_i)$.

### Phénomène de Runge

Pour $f(x) = 1/(1+25x^2)$ sur $[-1,1]$ avec nœuds équidistants : $\|p_n - f\|_\infty \to \infty$ quand $n \to \infty$.

**Cause** : $|\omega(x)|$ explose aux bords avec des nœuds équidistants.

### Nœuds de Tchebychev

$$x_k = \frac{a+b}{2} + \frac{b-a}{2} \cos\left(\frac{(2k+1)\pi}{2(n+1)}\right)$$

Minimisent $\|\omega\|_\infty$ sur $[a,b]$. Résultat : convergence exponentielle.

## 4.3 Splines cubiques (Définition 4.12)

Polynômes de degré 3 par morceaux, raccordés $C^2$. Conditions de bord « naturelles » : $S''(a) = S''(b) = 0$.

Construction : résoudre un **système tridiagonal** pour les moments $M_i = S''(x_i)$.

**Avantage** : pas de phénomène de Runge, convergent sur nœuds équidistants.

## Programmes associés

| Module | Contenu |
|---|---|
| `runge_phenomenon.py` | Lagrange, Newton, Neville, Runge, Tchebychev |
| `lagrange_interpolation.py` | Base de Lagrange + forme barycentrique |
| `neville_aitken.py` | Tableau complet, Übung 4.7 |
| `newton_divided_differences.py` | Différences divisées, Übung 4.5 |
| `splines.py` | Splines cubiques naturelles, Thomas |
