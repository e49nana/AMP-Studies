# Chapitre 2 — Diskrete Verteilungen

> **Résumé de révision** — Stochastik für AMP, S3

## 2.1 Distributions classiques

| Distribution | $P(X=k)$ | $E[X]$ | $\text{Var}(X)$ | FGP $G(s)$ |
|---|---|---|---|---|
| Bernoulli($p$) | $p^k(1-p)^{1-k}$ | $p$ | $p(1-p)$ | $1-p+ps$ |
| Binomiale($n,p$) | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | $(1-p+ps)^n$ |
| Poisson($\lambda$) | $\lambda^k e^{-\lambda}/k!$ | $\lambda$ | $\lambda$ | $e^{\lambda(s-1)}$ |
| Géométrique($p$) | $(1-p)^{k-1}p$ | $1/p$ | $(1-p)/p^2$ | $ps/(1-(1-p)s)$ |
| Hypergéom.($N,K,n$) | $\frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}$ | $nK/N$ | complexe | — |

**Approximation de Poisson :** $B(n,p) \approx Po(np)$ quand $n$ grand, $p$ petit, $np$ modéré.

## 2.2 Espérance et variance

$$E[X] = \sum_k k \cdot P(X=k), \qquad \text{Var}(X) = E[X^2] - (E[X])^2$$

**Linéarité :** $E[aX+b] = aE[X]+b$, $\text{Var}(aX+b) = a^2\text{Var}(X)$

**Inégalités :**
- **Markov :** $P(X \geq a) \leq E[X]/a$ (pour $X \geq 0$)
- **Tchebychev :** $P(|X-\mu| \geq k\sigma) \leq 1/k^2$ (universelle, mais lâche)

## 2.3 Lois jointes

- **Marginales :** $P(X=x) = \sum_y P(X=x, Y=y)$
- **Covariance :** $\text{Cov}(X,Y) = E[XY] - E[X]E[Y]$
- **Corrélation :** $\rho = \text{Cov}(X,Y)/(\sigma_X \sigma_Y) \in [-1,1]$
- **Indépendance :** $P(X,Y) = P(X) \cdot P(Y) \Rightarrow \text{Cov} = 0$ (réciproque fausse !)
- **Somme :** $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$

## 2.4 Fonctions génératrices

$$G_X(s) = E[s^X] = \sum_k P(X=k) \cdot s^k$$

**Moments :** $G'(1) = E[X]$, $G''(1) + G'(1) - (G'(1))^2 = \text{Var}(X)$

**Somme d'indépendantes :** $G_{X+Y}(s) = G_X(s) \cdot G_Y(s)$
- $Po(\lambda_1) + Po(\lambda_2) = Po(\lambda_1 + \lambda_2)$
- $B(n_1,p) + B(n_2,p) = B(n_1+n_2, p)$

## Programmes associés

| Module | Contenu |
|---|---|
| `discrete_distributions.py` | PMF, moments, approximation Poisson |
| `expected_value.py` | E[X], Var, König, Tchebychev/Markov |
| `joint_distributions.py` | Tableau de contingence, Cov, ρ |
| `generating_functions.py` | FGP, moments, somme de v.a. |
