# Chapitre 2 — Lineare Gleichungssysteme

> **Résumé de révision** — Kröger, *Numerische Mathematik 1 für AMP*, §2.1–2.5

## 2.1 Normes matricielles

**Définition 2.1** — Norme induite : $\|A\| = \sup_{x \neq 0} \frac{\|Ax\|}{\|x\|}$.

| Norme vectorielle | Norme induite | Formule | Nom |
|---|---|---|---|
| $\|\cdot\|_\infty$ | $\|A\|_\infty$ | $\max_i \sum_j |a_{ij}|$ | Zeilensummennorm (Satz 2.5) |
| $\|\cdot\|_1$ | $\|A\|_1$ | $\max_j \sum_i |a_{ij}|$ | Spaltensummennorm (Satz 2.7) |
| $\|\cdot\|_2$ | $\|A\|_2$ | $\sqrt{\lambda_{\max}(A^*A)}$ | Spektralnorm (Satz 2.9) |

**Satz 2.2** — Submultiplikativité : $\|AB\| \leq \|A\| \cdot \|B\|$.

**Satz 2.10** — Frobenius : $\|A\|_F = \sqrt{\sum_{ij} |a_{ij}|^2}$. Verträgliche mais pas induite.

**Satz 2.14** — $\rho(A) \leq \|A\|$ pour toute norme matricielle.

## 2.2 Conditionnement de matrices

**Satz 2.17** — $\kappa(A) = \|A\| \cdot \|A^{-1}\|$. Borne fondamentale :
$$\frac{\|\Delta x\|}{\|x\|} \leq \kappa(A) \cdot \frac{\|\Delta b\|}{\|b\|}$$

**Règle pratique** : $\kappa(A) = 10^k$ → perte de $k$ chiffres décimaux.

## 2.3 Gauss et décomposition LR

**Algorithme de Gauss** — Élimination par étapes. Coût : $\frac{1}{3}n^3$ (Satz 2.20).

**Décomposition $PA = LR$** — $L$ triangulaire inf. (diag = 1), $R$ triangulaire sup. Avantage : résoudre $Ax = b$ pour un nouveau $b$ ne coûte que $n^2$.

**Pivotstrategien (§2.3.5)** :
- *Aucun* : risque de division par zéro ou Auslöschung
- *Partiel (Spaltenpivotsuche)* : **standard** — échange la ligne avec le plus grand $|a_{ik}|$
- *Total* : plus stable mais $O(n^3)$ supplémentaire

**Determinante (§2.3.3)** — « Abfallprodukt » : $\det(A) = (-1)^p \prod r_{ii}$.

**Nachiteration (§2.3.7)** — Améliore itérativement : $r = A\tilde{x} - b$, résoudre $Ay = r$, $\tilde{x}_2 = \tilde{x} - y$.

## 2.4 Cholesky

Pour $A$ symétrique définie positive : $A = LL^T$. Coût : $\frac{1}{6}n^3$ (moitié du Gauss).

## 2.5 Méthodes itératives

**Motivation (Beispiel 2.24)** — Équation de la chaleur 2D, grille $1000 \times 1000$ : Gauss prendrait 1 mois et 8 To. Itératif : quelques secondes par pas.

**Jacobi (formule 2.17, Gesamtschritt)** :
$$x_i^+ = \frac{1}{a_{ii}}\left(b_i - \sum_{j \neq i} a_{ij} x_j\right)$$

**Gauss-Seidel (formule 2.18, Einzelschritt)** — Utilise les $x_j^+$ déjà calculés pour $j < i$.

**Convergence** :
- **Satz 2.35** — Converge ssi $\rho(S) < 1$ (rayon spectral de la matrice d'itération)
- **Satz 2.37** — Converge si $A$ strictement diagonalement dominante
- Rate asymptotique : $-\log_{10}(\rho(S))$ chiffres par itération
- Pour les matrices Stieltjes : GS est **2× plus rapide** que Jacobi

## Programmes associés

| Module | Contenu |
|---|---|
| `matrix_norms.py` | 5 normes + $\rho(A) \leq \|A\|$ + $\kappa(A)$ |
| `gauss_elimination.py` | 3 pivots, Übung 2.21 |
| `lu_decomposition.py` | LR réutilisable, Nachiteration, Cholesky |
| `jacobi_gauss_seidel.py` | Jacobi, GS, chaleur 2D |
| `error_analysis_linsys.py` | Erreur vs $\kappa(A)$ |
| `heat_equation_demo.py` | Benchmark Jacobi/GS/SciPy |
