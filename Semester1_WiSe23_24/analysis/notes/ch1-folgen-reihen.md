# Chapitre 1 — Folgen und Reihen

> **Résumé de révision** — Analysis / Mathematik 1 für AMP, S1

## 1.1 Suites (Folgen)

**Définition** — Une suite $(a_n)_{n \in \mathbb{N}}$ est une application $\mathbb{N} \to \mathbb{R}$.

**Convergence** — $(a_n)$ converge vers $L$ si $\forall \varepsilon > 0, \exists N : n \geq N \Rightarrow |a_n - L| < \varepsilon$.

**Critères pratiques :**

| Propriété | Condition |
|---|---|
| Monotone bornée | monotone + bornée → converge |
| Sandwich | $a_n \leq b_n \leq c_n$, $a_n \to L$, $c_n \to L$ → $b_n \to L$ |
| Suite géométrique | $q^n \to 0$ ssi $|q| < 1$ |

**Suites classiques :**
- Arithmétique : $a_n = a_0 + nd$ → diverge (sauf $d = 0$)
- Géométrique : $a_n = a_0 q^n$ → converge ssi $|q| < 1$, limite $= 0$
- Héron : $x_{n+1} = \frac{1}{2}(x_n + a/x_n) \to \sqrt{a}$ (convergence quadratique)
- $(1 + 1/n)^n \to e \approx 2.71828$

## 1.2 Séries (Reihen)

**Définition** — $\sum_{k=0}^\infty a_k = \lim_{n \to \infty} S_n$ avec $S_n = \sum_{k=0}^n a_k$.

**Séries de référence :**

| Série | Converge ? | Somme |
|---|---|---|
| Géométrique $\sum q^k$ | $|q| < 1$ | $1/(1-q)$ |
| Harmonique $\sum 1/k$ | **non** | $\infty$ |
| Harm. alternée $\sum (-1)^k/k$ | oui | $\ln 2$ |
| Riemann $\sum 1/k^p$ | $p > 1$ | $\pi^2/6$ si $p=2$ |
| Exponentielle $\sum x^k/k!$ | toujours | $e^x$ |

## 1.3 Critères de convergence

| Critère | Condition | Conclusion |
|---|---|---|
| **d'Alembert** (quotient) | $L = \lim |a_{n+1}/a_n|$ | $L < 1$ → converge, $L > 1$ → diverge |
| **Cauchy** (racine) | $L = \lim \sup |a_n|^{1/n}$ | idem |
| **Leibniz** (alternée) | $|a_n| \searrow 0$, signes alternent | converge |
| **Comparaison** | $|a_n| \leq b_n$ et $\sum b_n$ converge | $\sum a_n$ converge |
| **Nécessaire** | $a_n \not\to 0$ | diverge |

## 1.4 Séries de Taylor

$$f(x) = \sum_{k=0}^\infty \frac{f^{(k)}(a)}{k!}(x-a)^k$$

**Reste de Lagrange :** $|R_n(x)| \leq \frac{M}{(n+1)!}|x-a|^{n+1}$ avec $M = \max |f^{(n+1)}|$.

**Développements classiques (autour de $a = 0$) :**

| Fonction | Série | Rayon |
|---|---|---|
| $e^x$ | $\sum x^k/k!$ | $\infty$ |
| $\sin x$ | $\sum (-1)^k x^{2k+1}/(2k+1)!$ | $\infty$ |
| $\cos x$ | $\sum (-1)^k x^{2k}/(2k)!$ | $\infty$ |
| $\ln(1+x)$ | $\sum (-1)^{k+1} x^k/k$ | $1$ |
| $\arctan x$ | $\sum (-1)^k x^{2k+1}/(2k+1)$ | $1$ |
| $(1+x)^\alpha$ | $\sum \binom{\alpha}{k} x^k$ | $1$ |

## Programmes associés

| Module | Contenu |
|---|---|
| `sequences.py` | Suites, Héron, (1+1/n)^n, analyse automatique |
| `series.py` | Séries classiques, critères d'Alembert/Cauchy/Leibniz |
| `taylor_series.py` | Développements, reste, rayon de convergence |
| `limits.py` | Limites, ε-δ, formes indéterminées |
