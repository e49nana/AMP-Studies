# Chapitre 4 — Determinanten

> **Résumé de révision** — Lineare Algebra für AMP, S1

## 4.1 Définition et calcul

| Méthode | Coût | Usage |
|---|---|---|
| Leibniz (somme sur $S_n$) | $O(n!)$ | théorique uniquement |
| Sarrus (3×3) | $O(1)$ | calcul à la main |
| Laplace (cofacteurs) | $O(n!)$ | calcul à la main pour $n \leq 4$ |
| Gauss (échelonnement) | $O(n^3)$ | **la seule méthode pratique** |

**Par Gauss :** $\det(A) = (-1)^p \prod r_{ii}$ ($p$ = nombre d'échanges de lignes).

## 4.2 Propriétés

| Propriété | Formule |
|---|---|
| Transposée | $\det(A^T) = \det(A)$ |
| Produit | $\det(AB) = \det(A) \cdot \det(B)$ |
| Inverse | $\det(A^{-1}) = 1/\det(A)$ |
| Scalaire | $\det(\lambda A) = \lambda^n \det(A)$ |
| Échange de lignes | $\det \to -\det$ |
| $L_i \leftarrow \lambda L_i$ | $\det \to \lambda \det$ |
| $L_i \leftarrow L_i + \lambda L_j$ | $\det$ inchangé |

**Critère d'inversibilité :** $A$ inversible $\Leftrightarrow$ $\det(A) \neq 0$.

## 4.3 Interprétation géométrique

$|\det(A)| =$ volume du parallélépipède engendré par les colonnes de $A$.

$\text{sign}(\det) =$ orientation (positive = même orientation que la base canonique).

## 4.4 Cramer et cofacteurs

**Règle de Cramer :** $x_i = \frac{\det(A_i)}{\det(A)}$ où $A_i$ = $A$ avec colonne $i$ remplacée par $b$.

Élégant mais $O(n^4)$ → inutilisable pour $n > 5$.

**Inverse par cofacteurs :** $A^{-1} = \frac{1}{\det A} \cdot \text{adj}(A)$ avec $\text{adj}(A) = \text{cof}(A)^T$.

## Programmes associés

| Module | Contenu |
|---|---|
| `determinant.py` | Leibniz, Sarrus, Laplace, Gauss — comparaison |
| `determinant_properties.py` | Vérification numérique, parallélogrammes |
| `cramer.py` | Cramer, cofacteurs, benchmark vs Gauss |
