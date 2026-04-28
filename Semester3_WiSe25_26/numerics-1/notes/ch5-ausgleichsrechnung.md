# Chapitre 5 — Lineare Ausgleichsrechnung

> **Résumé de révision** — Kröger, *Numerische Mathematik 1 für AMP*, §5.1–5.5

## 5.1 Problème

Système surdéterminé $Ax = b$ avec $A \in \mathbb{R}^{N \times n}$, $N > n$ : pas de solution exacte.

**Objectif** : trouver $x^*$ qui minimise $\|Ax - b\|_2$.

## 5.2 Équations normales (Satz 5.2)

$$A^T A \, x = A^T b$$

Simple mais **instable** : $\kappa(A^T A) = \kappa(A)^2$. On perd le double de chiffres.

## 5.3 Décomposition QR par Householder

### Réflexion de Householder (Satz 5.6)

Pour un vecteur $a$, on construit $w = a + \text{sign}(a_1)\|a\|_2 e_1$ :
$$Q = I - \frac{2 w w^T}{w^T w}$$

Appliquée colonne par colonne → $QA = R$ avec $Q$ orthogonale, $R$ triangulaire sup.

### Résolution (§5.3.2)

1. $QA = R$ → $QA = \begin{pmatrix} R_1 \\ 0 \end{pmatrix}$
2. $c = Qb = \begin{pmatrix} c_1 \\ c_2 \end{pmatrix}$
3. $R_1 x = c_1$ (substitution arrière)
4. Résidu minimal : $\|r\|_2 = \|c_2\|_2$

### Avantage sur les équations normales (§5.4)

$\kappa(R_1) = \kappa(A)$ vs $\kappa(A^T A) = \kappa(A)^2$. QR perd moitié moins de chiffres.

## 5.5 Applications

### 5.5.1 Régression linéaire

$y = a + bx$ → $A = \begin{pmatrix} 1 & x_1 \\ \vdots & \vdots \\ 1 & x_N \end{pmatrix}$.

### 5.5.2 Ajustement polynomial

$y = a_0 + a_1 x + \cdots + a_d x^d$ → matrice de Vandermonde.

### 5.5.3 Linéarisation de modèles non-linéaires

| Modèle | Transformation |
|---|---|
| $y = a \cdot e^{bx}$ | $\ln y = \ln a + bx$ |
| $y = a \cdot x^b$ | $\ln y = \ln a + b \ln x$ |
| $y = a/(b+x)$ | $1/y = b/a + x/a$ |
| $y = ax/(b+x)$ | $1/y = 1/a + (b/a)/x$ |

## Coûts

| Méthode | Construction | Par sec. membre |
|---|---|---|
| Éq. normales + Cholesky | $n^2 N + \frac{1}{6}n^3$ | $n^2$ |
| Householder QR | $2n^2 N$ | $nN$ |

QR coûte ~2× plus cher mais est beaucoup plus stable.

## Programmes associés

| Module | Contenu |
|---|---|
| `householder_qr.py` | QR from-scratch, moindres carrés |
| `normal_equations.py` | Éq. normales + démo $\kappa^2$ |
| `least_squares_fitting.py` | Régression poly/exp/puissance |
| `nonlinear_to_linear.py` | 5 modèles linéarisés |
