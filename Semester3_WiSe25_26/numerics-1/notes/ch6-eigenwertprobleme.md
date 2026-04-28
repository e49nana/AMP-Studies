# Chapitre 6 — Eigenwertprobleme

> **Résumé de révision** — Kröger, *Numerische Mathematik 1 für AMP*, §6.1–6.7

## 6.1 Rappels

**Problème** : trouver $\lambda \in \mathbb{C}$, $v \neq 0$ tels que $Av = \lambda v$.

**Polynôme caractéristique** : $\det(A - \lambda I) = 0$. En théorie ça marche, en pratique c'est **catastrophique** numériquement (théorème d'Abel pour $n \geq 5$, instabilité des coefficients).

**Satz 6.1** — Toute matrice $A \in \mathbb{C}^{n \times n}$ a exactement $n$ valeurs propres (comptées avec multiplicité).

## 6.4 Vektoriteration (Von-Mises)

### Méthode directe (§6.4.1)

Itérer $y = Ax$, normaliser, extraire $\lambda$ :
1. $y = Ax$
2. $\ell = \|y\|_2$
3. $x^+ = y / \tilde{\ell}$ (avec correction du signe)

**Convergence** vers $(\lambda_1, v_1)$ avec $|\lambda_1| = \rho(A)$.

**Satz 6.7** — Rate linéaire : $|\lambda_2 / \lambda_1|$. Rapide si $\lambda_1$ est bien séparé.

### Inverse iteration (Wielandt, §6.4.2)

Appliquer Von-Mises à $(A - \sigma I)^{-1}$ → converge vers la valeur propre **la plus proche de $\sigma$**.

En pratique : résoudre $(A - \sigma I) y = x$ à chaque pas (une seule LR).

Rate : $|\lambda_i - \sigma| / |\lambda_k - \sigma|$. Plus $\sigma$ est proche, plus c'est rapide.

## 6.5 Méthode de Jacobi

Pour matrices **symétriques** uniquement.

**Principe** : appliquer itérativement des rotations de Givens $A^{(k+1)} = U^T A^{(k)} U$ pour annuler les éléments hors-diagonaux.

**Mesure de convergence** : $\text{off}(A) = \sqrt{\sum_{i \neq j} a_{ij}^2} \to 0$.

**Deux variantes** :
- *Classique* : annuler le plus grand $|a_{pq}|$ à chaque pas
- *Cyclique* : parcourir systématiquement tous les $(p,q)$

**Avantage** : calcule aussi les vecteurs propres (via accumulation des rotations).

## 6.6 Cercles de Gershgorin

**Satz 6.12** — Chaque valeur propre est dans au moins un cercle :
$$K_i = \{z \in \mathbb{C} : |z - a_{ii}| \leq r_i\}, \qquad r_i = \sum_{k \neq i} |a_{ik}|$$

**Korollar 6.13** — Si un cercle est **disjoint** des autres, il contient exactement une valeur propre.

**Korollar 6.14** — On peut aussi utiliser les rayons par colonnes ($A^T$). L'intersection donne un encadrement plus serré.

## 6.7 QR-Algorithmus

### Principe (§6.7.2)

Itérer : $Q_k R_k = A_k$ (QR), $A_{k+1} = R_k Q_k$.

La suite $A_k$ converge vers une matrice triangulaire supérieure (Schur).

### Accélérations

- **Hessenberg (§6.7.4)** — Réduction préalable : chaque pas QR passe de $O(n^3)$ à $O(n^2)$.
- **Shift (§6.7.3)** — $A_k - \mu_k I = Q_k R_k$, $A_{k+1} = R_k Q_k + \mu_k I$. Convergence cubique pour matrices symétriques.

C'est la méthode utilisée par `numpy.linalg.eigvals` (variante Francis avec double shift implicite).

## Tableau comparatif

| Méthode | Pour | Coût/it. | Convergence | Vecteurs propres |
|---|---|---|---|---|
| Von-Mises | $\lambda$ dominant | $O(n^2)$ | linéaire | oui (1 seul) |
| Wielandt | $\lambda$ proche de $\sigma$ | $O(n^2)$ + 1 LR | linéaire (rapide) | oui (1 seul) |
| Jacobi | toutes (sym.) | $O(n^3)$ | quadratique | oui (toutes) |
| QR | toutes | $O(n^2)$ (Hess.) | cubique (shift) | non directement |

## Programmes associés

| Module | Contenu |
|---|---|
| `power_iteration.py` | Von-Mises + Wielandt |
| `gershgorin_circles.py` | Cercles + visualisation |
| `jacobi_eigenvalue.py` | Classique + cyclique |
| `qr_algorithm.py` | QR + Hessenberg + shift |
