# Chapitre 2 — Funktionalanalysis Grundlagen

> **Résumé de révision** — Angewandte Analysis für AMP, S3-S4

## 2.1 Espaces métriques

**Métrique :** $d : X \times X \to \mathbb{R}$ avec positivité, séparation, symétrie, inégalité triangulaire.

| Métrique | Formule | Boule unité |
|---|---|---|
| Euclidienne $d_2$ | $\sqrt{\sum(x_i-y_i)^2}$ | cercle |
| Manhattan $d_1$ | $\sum|x_i-y_i|$ | losange |
| Chebyshev $d_\infty$ | $\max|x_i-y_i|$ | carré |

**Suite de Cauchy :** $\forall \varepsilon > 0, \exists N : m,n \geq N \Rightarrow d(x_m, x_n) < \varepsilon$.

**Complet :** toute suite de Cauchy converge (dans l'espace).

**Point fixe de Banach :** si $T$ est une contraction ($d(Tx,Ty) \leq q \cdot d(x,y)$, $q < 1$) dans un espace métrique complet, alors $T$ a un unique point fixe et $x_{n+1} = T(x_n)$ converge.

## 2.2 Espaces normés

**Norme :** $\|x\| \geq 0$, $\|\alpha x\| = |\alpha|\|x\|$, $\|x+y\| \leq \|x\| + \|y\|$.

**Normes $p$ :** $\|x\|_p = \left(\sum |x_i|^p\right)^{1/p}$, $\|x\|_\infty = \max|x_i|$.

**Équivalence en dim finie :** $\exists c_1, c_2 > 0 : c_1\|x\|_q \leq \|x\|_p \leq c_2\|x\|_q$ (toutes les normes donnent la même topologie).

**Banach :** espace normé complet.

**Norme d'opérateur :** $\|A\| = \sup_{\|x\|=1} \|Ax\|$. Pour $\|\cdot\|_2$ : $\|A\|_2 = \sigma_{\max}(A)$.

## 2.3 Espaces de Hilbert

**Produit scalaire :** $\langle \cdot, \cdot \rangle$ bilinéaire, symétrique, défini positif. Induit $\|x\| = \sqrt{\langle x, x \rangle}$.

**Cauchy-Schwarz :** $|\langle x, y \rangle| \leq \|x\| \cdot \|y\|$.

**Projection orthogonale :** $\text{proj}_V(x) = \sum_i \langle x, e_i \rangle e_i$ (base ON de $V$).

→ **Meilleure approximation** dans un sous-espace : la projection minimise $\|x - p\|$.

**Gram-Schmidt :** orthonormalise une famille libre.

**Parseval :** $\|x\|^2 = \sum |\langle x, e_i \rangle|^2$ (si base ON complète).

## 2.4 Espaces de fonctions

| Espace | Norme | Complet ? |
|---|---|---|
| $C[a,b]$ avec $\|\cdot\|_\infty$ | $\max|f(x)|$ | oui (Banach) |
| $C[a,b]$ avec $\|\cdot\|_2$ | $\sqrt{\int|f|^2}$ | **non** |
| $L^2[a,b]$ | $\sqrt{\int|f|^2}$ | oui (Hilbert) |

**Convergences :** uniforme $\Rightarrow$ ponctuelle, uniforme $\Rightarrow$ $L^2$, mais pas l'inverse.

**Fourier dans $L^2$ :** la série de Fourier = projection orthogonale sur $\text{span}\{1, \cos(nx), \sin(nx)\}$ = **meilleure approximation** en norme $L^2$.

## Programmes associés

| Module | Contenu |
|---|---|
| `metric_spaces.py` | Métriques, Cauchy, point fixe de Banach |
| `normed_spaces.py` | Normes $p$, équivalence, norme d'opérateur |
| `inner_product_spaces.py` | Cauchy-Schwarz, projection, Gram-Schmidt, Parseval |
| `function_spaces.py` | $C[a,b]$ vs $L^2$, convergences, Fourier dans $L^2$ |
