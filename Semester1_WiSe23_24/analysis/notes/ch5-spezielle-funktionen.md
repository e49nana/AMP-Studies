# Chapitre 5 — Spezielle Funktionen und Reihen

> **Résumé de révision** — Analysis / Mathematik 1 für AMP, S1

## 5.1 Séries de Fourier

Toute fonction $f$ de période $T = 2\pi$ se décompose en :
$$f(x) = \frac{a_0}{2} + \sum_{n=1}^\infty \left(a_n \cos(nx) + b_n \sin(nx)\right)$$

**Coefficients :**
$$a_0 = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\,dx, \quad a_n = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\cos(nx)\,dx, \quad b_n = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\sin(nx)\,dx$$

**Signaux classiques :**

| Signal | Coefficients non nuls |
|---|---|
| Carré | $b_n = 4/(n\pi)$ pour $n$ impair |
| Dent de scie | $b_n = 2(-1)^{n+1}/(n\pi)$ |
| Triangle | $a_n$ seulement ($n$ impair) |

**Phénomène de Gibbs :** aux discontinuités, la somme partielle dépasse de ~9% — ce dépassement **ne disparaît jamais** quand $N \to \infty$.

**Identité de Parseval :** $\frac{a_0^2}{2} + \sum(a_n^2 + b_n^2) = \frac{1}{\pi}\int_{-\pi}^{\pi} |f(x)|^2\,dx$

## 5.2 Séries entières

$$\sum_{n=0}^\infty a_n (x - a)^n$$

**Rayon de convergence :**
- Hadamard : $1/R = \limsup |a_n|^{1/n}$
- d'Alembert : $R = \lim |a_n / a_{n+1}|$ (si la limite existe)

**Propriétés sur $|x-a| < R$ :**
- Convergence uniforme et absolue
- Dérivation/intégration terme à terme (même rayon $R$)
- Convergence aux bords : à vérifier au cas par cas

## 5.3 Fonctions spéciales

### Fonction Gamma
$$\Gamma(x) = \int_0^\infty t^{x-1} e^{-t}\,dt$$
- $\Gamma(n) = (n-1)!$ pour $n \in \mathbb{N}^*$
- $\Gamma(x+1) = x\Gamma(x)$ (récurrence)
- $\Gamma(1/2) = \sqrt{\pi}$
- Stirling : $n! \approx \sqrt{2\pi n}\,(n/e)^n$

### Fonction Beta
$$B(a,b) = \int_0^1 t^{a-1}(1-t)^{b-1}\,dt = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$$

### Fonctions hyperboliques
$$\sinh x = \frac{e^x - e^{-x}}{2}, \quad \cosh x = \frac{e^x + e^{-x}}{2}, \quad \tanh x = \frac{\sinh x}{\cosh x}$$

Identité fondamentale : $\cosh^2 x - \sinh^2 x = 1$.

### Fonction erreur
$$\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2}\,dt$$

$\text{erf}(\infty) = 1$, lié à la distribution normale : $\Phi(x) = \frac{1}{2}[1 + \text{erf}(x/\sqrt{2})]$.

## Programmes associés

| Module | Contenu |
|---|---|
| `fourier_series.py` | Coefficients, signaux, Gibbs, Parseval |
| `power_series.py` | Rayon de convergence, dérivation terme à terme |
| `special_functions.py` | Gamma, Beta, hyperboliques, erf |
