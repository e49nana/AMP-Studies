# Chapitre 1 — Vektoren und Rechenoperationen

> **Résumé de révision** — Lineare Algebra für AMP, S1

## 1.1 Vecteurs dans R^n

Un vecteur $v \in \mathbb{R}^n$ est un $n$-uplet de nombres réels : $v = (v_1, \dots, v_n)^T$.

**Opérations fondamentales :**
- Addition : $(u + v)_i = u_i + v_i$
- Multiplication scalaire : $(\lambda v)_i = \lambda v_i$
- Combinaison linéaire : $\alpha_1 v_1 + \alpha_2 v_2 + \cdots + \alpha_k v_k$

## 1.2 Skalarprodukt (produit scalaire)

$$\langle u, v \rangle = \sum_{i=1}^n u_i v_i = u^T v$$

**Propriétés clés :**
- Symétrie : $\langle u, v \rangle = \langle v, u \rangle$
- Bilinéarité : linéaire en chaque argument
- Positivité : $\langle v, v \rangle \geq 0$, $= 0$ ssi $v = 0$

**Angle entre vecteurs :** $\cos \theta = \frac{\langle u, v \rangle}{\|u\| \cdot \|v\|}$

**Cauchy-Schwarz :** $|\langle u, v \rangle| \leq \|u\| \cdot \|v\|$

**Orthogonalité :** $u \perp v \Leftrightarrow \langle u, v \rangle = 0$

## 1.3 Kreuzprodukt (produit vectoriel, R³ uniquement)

$$u \times v = \begin{pmatrix} u_2 v_3 - u_3 v_2 \\ u_3 v_1 - u_1 v_3 \\ u_1 v_2 - u_2 v_1 \end{pmatrix}$$

- $\|u \times v\| =$ aire du parallélogramme
- $u \times v \perp u$ et $u \times v \perp v$
- Anticommutativité : $u \times v = -(v \times u)$

**Spatprodukt :** $\langle u \times v, w \rangle = \det(u, v, w) =$ volume (signé) du parallélépipède.

## 1.4 Projection orthogonale

$$\text{proj}_v(u) = \frac{\langle u, v \rangle}{\langle v, v \rangle} \cdot v$$

Décomposition : $u = \text{proj}_v(u) + u_\perp$ avec $u_\perp \perp v$.

## 1.5 Droites et plans en R³

| Objet | Forme paramétrique | Forme cartésienne |
|---|---|---|
| Droite | $r(t) = p + t \cdot d$ | — |
| Plan | $r(s,t) = p + su + tv$ | $n \cdot x = n \cdot p$ |

**Distances :**
- Point-plan : $\frac{|n \cdot (Q - P)|}{\|n\|}$
- Point-droite : $\frac{\|AP \times d\|}{\|d\|}$

## 1.6 Indépendance linéaire

$v_1, \dots, v_k$ sont linéairement indépendants ssi $\alpha_1 v_1 + \cdots + \alpha_k v_k = 0 \Rightarrow \alpha_i = 0 \;\forall i$.

**Rang :** nombre maximal de vecteurs indépendants dans une famille.

**Base :** famille libre et génératrice. En R^n : exactement $n$ vecteurs.

## 1.7 Sous-espaces

- $\text{Kern}(A) = \{x : Ax = 0\}$ (noyau)
- $\text{Bild}(A) =$ espace des colonnes (image)
- **Rangsatz :** $\dim \text{Kern}(A) + \dim \text{Bild}(A) = n$

## Programmes associés

| Module | Contenu |
|---|---|
| `vectors_2d_3d.py` | Classe Vecteur, combinaisons linéaires, visualisation |
| `dot_product.py` | Produit scalaire, angles, projection, Cauchy-Schwarz |
| `cross_product.py` | Produit vectoriel, aires, volumes |
| `lines_planes.py` | Droites/plans R³, intersections, distances |
| `linear_independence.py` | Rang, base, coordonnées |
| `subspaces.py` | Kern, Bild, RREF, Rangsatz |
