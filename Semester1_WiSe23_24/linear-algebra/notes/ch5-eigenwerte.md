# Chapitre 5 — Eigenwerte und Eigenvektoren

> **Résumé de révision** — Lineare Algebra für AMP, S1

## 5.1 Définition

$Av = \lambda v$ avec $v \neq 0$. $\lambda$ = valeur propre (Eigenwert), $v$ = vecteur propre (Eigenvektor).

**Polynôme caractéristique :** $p(\lambda) = \det(A - \lambda I)$. Les racines sont les valeurs propres.

**Calcul pour 2×2 :** $\lambda = \frac{\text{tr}(A) \pm \sqrt{\text{tr}(A)^2 - 4\det(A)}}{2}$.

## 5.2 Eigenräume et multiplicités

| Multiplicité | Définition |
|---|---|
| Algébrique $m_a$ | ordre de $\lambda$ comme racine de $p(\lambda)$ |
| Géométrique $m_g$ | $\dim \text{Kern}(A - \lambda I)$ |

Toujours $1 \leq m_g \leq m_a$.

## 5.3 Relations fondamentales

- $\text{tr}(A) = \sum \lambda_i$ (somme des valeurs propres)
- $\det(A) = \prod \lambda_i$ (produit des valeurs propres)
- $A$ singulière $\Leftrightarrow$ $0$ est valeur propre

## 5.4 Diagonalisation

$A = P D P^{-1}$ avec $D = \text{diag}(\lambda_1, \dots, \lambda_n)$, $P = [v_1 | \cdots | v_n]$.

**Critère :** $A$ est diagonalisable ssi $m_g = m_a$ pour toute valeur propre.

**Application :** $A^k = P D^k P^{-1}$ (puissances triviales via la diagonale).

## 5.5 Théorème spectral

Pour $A$ **symétrique** ($A = A^T$) :
- Toutes les valeurs propres sont **réelles**
- Les vecteurs propres sont **orthogonaux**
- $A = Q \Lambda Q^T$ avec $Q$ orthogonale

**Quotient de Rayleigh :** $R(x) = \frac{x^T A x}{x^T x}$, $\lambda_{\min} \leq R(x) \leq \lambda_{\max}$.

## 5.6 Stabilité et applications

- $|λ| < 1$ pour tout $\lambda$ → $A^k \to 0$ (asymptotiquement stable)
- $|λ| > 1$ pour un $\lambda$ → $A^k$ diverge (instable)
- **Markov :** $P\pi = \pi$ → l'état stationnaire est le vecteur propre de $\lambda = 1$

## Programmes associés

| Module | Contenu |
|---|---|
| `eigenvalues.py` | Polynôme caractéristique, Faddeev-LeVerrier |
| `eigenspaces.py` | Eigenräume, $m_a$ vs $m_g$, diagonalisabilité |
| `diagonalization.py` | $A = PDP^{-1}$, $A^k$, Fibonacci |
| `matrix_powers.py` | Markov, stabilité, systèmes dynamiques |
| `spectral_theorem.py` | $A = Q\Lambda Q^T$, Rayleigh, ellipse |
