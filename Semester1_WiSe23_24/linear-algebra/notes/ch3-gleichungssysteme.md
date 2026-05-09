# Chapitre 3 — Lineare Gleichungssysteme

> **Résumé de révision** — Lineare Algebra für AMP, S1

## 3.1 Gauss-Jordan et RREF

L'algorithme transforme $[A|b]$ en forme échelonnée réduite par opérations élémentaires sur les lignes.

**RREF :** chaque colonne pivot a exactement un 1, tout le reste est 0.

## 3.2 Structure des solutions

| Cas | Condition | Ensemble des solutions |
|---|---|---|
| Unique | $\text{rang}(A) = \text{rang}([A|b]) = n$ | $\{x^*\}$ |
| Infini | $\text{rang}(A) = \text{rang}([A|b]) < n$ | $x_{part} + \text{Kern}(A)$ |
| Incompatible | $\text{rang}(A) < \text{rang}([A|b])$ | $\emptyset$ |

**Principe de superposition :** la solution générale de $Ax = b$ est $x = x_{part} + x_{hom}$ avec $x_{hom} \in \text{Kern}(A)$.

## 3.3 Rang

- $\text{rang}(A) =$ nombre de pivots dans la RREF
- Rang ligne $=$ rang colonne (toujours)
- $\text{rang}(AB) \leq \min(\text{rang}(A), \text{rang}(B))$
- $A$ inversible $\Leftrightarrow$ $\text{rang}(A) = n$

**Rangsatz :** $\text{rang}(A) + \dim \text{Kern}(A) = n$.

## 3.4 Inverse par Gauss-Jordan

$[A | I] \xrightarrow{\text{RREF}} [I | A^{-1}]$

**Formules explicites :**
- $2 \times 2$ : $A^{-1} = \frac{1}{ad-bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$
- $n \times n$ : $A^{-1} = \frac{1}{\det A} \text{adj}(A)$ (cofacteurs, inefficace)

## Programmes associés

| Module | Contenu |
|---|---|
| `gauss_jordan.py` | RREF, résolution, variables libres |
| `solution_structure.py` | Analyse complète, superposition, visualisation |
| `matrix_rank.py` | Rang, Rangsatz, critère de compatibilité |
| `inverse_by_gauss.py` | $[A|I] \to [I|A^{-1}]$, formules 2×2 et 3×3 |
