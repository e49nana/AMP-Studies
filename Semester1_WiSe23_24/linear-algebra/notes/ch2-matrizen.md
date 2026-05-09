# Chapitre 2 — Matrizen

> **Résumé de révision** — Lineare Algebra für AMP, S1

## 2.1 Opérations matricielles

- **Addition :** $(A + B)_{ij} = a_{ij} + b_{ij}$ (même taille)
- **Multiplication scalaire :** $(\lambda A)_{ij} = \lambda a_{ij}$
- **Multiplication :** $(AB)_{ij} = \sum_k a_{ik} b_{kj}$ — **non commutative** en général
- **Transposée :** $(A^T)_{ij} = a_{ji}$, $(AB)^T = B^T A^T$
- **Trace :** $\text{tr}(A) = \sum_i a_{ii}$, $\text{tr}(AB) = \text{tr}(BA)$

## 2.2 Matrices élémentaires

| Type | Notation | Effet | det |
|---|---|---|---|
| Permutation $P_{ij}$ | échange lignes $i, j$ | $P^2 = I$ | $-1$ |
| Dilatation $D_i(\lambda)$ | $L_i \leftarrow \lambda L_i$ | $D^{-1} = D_i(1/\lambda)$ | $\lambda$ |
| Transvection $L_{ij}(\lambda)$ | $L_i \leftarrow L_i + \lambda L_j$ | $L^{-1} = L_{ij}(-\lambda)$ | $1$ |

**Gauss = produit de matrices élémentaires :** $E_k \cdots E_1 A = R$.

## 2.3 Types de matrices

| Type | Condition | Propriétés |
|---|---|---|
| Symétrique | $A = A^T$ | valeurs propres réelles |
| Orthogonale | $Q^T Q = I$ | $\det = \pm 1$, conserve longueurs |
| Définie positive | $x^T A x > 0$ pour $x \neq 0$ | $\lambda_i > 0$ |
| Idempotente | $A^2 = A$ | projection |
| Nilpotente | $A^k = 0$ | $\lambda_i = 0$ |

**Décomposition universelle :** $A = \frac{A + A^T}{2} + \frac{A - A^T}{2}$ (symétrique + antisymétrique).

## 2.4 Applications linéaires

$f : \mathbb{R}^n \to \mathbb{R}^m$ est linéaire ssi $f(\alpha x + \beta y) = \alpha f(x) + \beta f(y)$.

**Matrice associée :** $A = [f(e_1) | f(e_2) | \cdots | f(e_n)]$.

**Composition :** $f \circ g \leftrightarrow A_f \cdot A_g$.

## 2.5 Changement de base

**Matrice de passage :** $P = B_2^{-1} B_1$.

- Vecteur : $[v]_{B_2} = P^{-1} [v]_{B_1}$
- Matrice : $[A]_{B_2} = P^{-1} A P$ (transformation de similitude)
- Les valeurs propres sont **invariantes** par changement de base.

## Programmes associés

| Module | Contenu |
|---|---|
| `matrix_operations.py` | Classe Matrice, multiplication, inverse, puissances |
| `elementary_matrices.py` | Permutation, dilatation, transvection |
| `matrix_types.py` | Classification automatique, décomposition sym+antisym |
| `linear_maps.py` | Rotations, réflexions, projections, visualisation |
| `coordinate_transform.py` | Matrice de passage, changement de base |
