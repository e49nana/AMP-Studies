# Chapitre 6 — Anwendungen

> **Résumé de révision** — Lineare Algebra für AMP, S1

## 6.1 Moindres carrés (pont vers Numerik)

Système surdéterminé $Ax = b$ ($m > n$) : pas de solution exacte.

**Objectif :** minimiser $\|Ax - b\|_2$.

**Solution :** $x^* = (A^T A)^{-1} A^T b$ (équations normales).

**Interprétation géométrique :** $Ax^*$ est la **projection orthogonale** de $b$ sur $\text{Im}(A)$. Le résidu $r = b - Ax^*$ est perpendiculaire à $\text{Im}(A)$.

## 6.2 Chaînes de Markov

**Matrice stochastique :** $P_{ij} \geq 0$, $\sum_i P_{ij} = 1$ (colonnes somment à 1).

**Distribution stationnaire :** $P\pi = \pi$ avec $\sum \pi_i = 1$.

C'est un problème de **vecteur propre** : $\pi$ est le vecteur propre de $P$ pour $\lambda = 1$.

**Convergence :** $P^k x_0 \to \pi$ pour toute distribution initiale $x_0$ (si $P$ ergodique). Vitesse : $|\lambda_2|$.

## 6.3 Transformations linéaires en R²

| Transformation | Matrice | det |
|---|---|---|
| Rotation $\theta$ | $\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$ | $1$ |
| Réflexion (axe $\theta$) | $\begin{pmatrix} \cos 2\theta & \sin 2\theta \\ \sin 2\theta & -\cos 2\theta \end{pmatrix}$ | $-1$ |
| Projection (dir. $\theta$) | $\begin{pmatrix} \cos^2\theta & \cos\theta\sin\theta \\ \cos\theta\sin\theta & \sin^2\theta \end{pmatrix}$ | $0$ |
| Cisaillement | $\begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}$ | $1$ |
| Homothétie | $\begin{pmatrix} s_x & 0 \\ 0 & s_y \end{pmatrix}$ | $s_x s_y$ |

## 6.4 SVD (Singulärwertzerlegung)

$$A = U \Sigma V^T$$

- $U$ : $m \times m$ orthogonale (vecteurs singuliers gauches)
- $\Sigma$ : $m \times n$ diagonale ($\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$)
- $V$ : $n \times n$ orthogonale (vecteurs singuliers droits)

**Valeurs singulières :** $\sigma_i = \sqrt{\lambda_i(A^T A)}$.

**Propriétés :**
- $\text{rang}(A) =$ nombre de $\sigma_i > 0$
- $\|A\|_2 = \sigma_1$, $\|A\|_F = \sqrt{\sum \sigma_i^2}$
- **Eckart-Young :** $A_k = \sum_{i=1}^k \sigma_i u_i v_i^T$ est la meilleure approximation de rang $k$

**Cas symétrique :** $A = A^T$ positif $\Rightarrow$ SVD = théorème spectral ($\sigma_i = \lambda_i$).

## Programmes associés

| Module | Contenu |
|---|---|
| `least_squares_intro.py` | Projection, résidus, régression |
| `markov_chains.py` | Matrices stochastiques, PageRank |
| `linear_transformations_2d.py` | Galerie de 8 transformations |
| `svd_intro.py` | SVD from-scratch, Eckart-Young, géométrie |
