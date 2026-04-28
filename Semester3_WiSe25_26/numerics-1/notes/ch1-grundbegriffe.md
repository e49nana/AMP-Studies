# Chapitre 1 — Grundbegriffe der Numerik

> **Résumé de révision** — Kröger, *Numerische Mathematik 1 für AMP*, §1.1–1.4

## 1.1 Vocabulaire fondamental

La **mathématique numérique** transforme des problèmes mathématiques continus en algorithmes exécutables sur machine. Trois sources d'erreur à distinguer :

- **Erreur de modélisation** (hors cours) — simplification de la réalité
- **Erreur de discrétisation** — remplacement du continu par du discret (chapitres 3–6)
- **Erreur d'arrondi** — arithmétique flottante à précision finie (ce chapitre)

## 1.2 Normes vectorielles

**Définition 1.2 (4 axiomes)** — Une norme sur $V$ est une application $\|\cdot\| : V \to \mathbb{R}$ vérifiant :

1. $\|x\| \geq 0$ (positivité)
2. $\|x\| = 0 \Leftrightarrow x = 0$ (définitude)
3. $\|\lambda x\| = |\lambda| \cdot \|x\|$ (homogénéité)
4. $\|x + y\| \leq \|x\| + \|y\|$ (inégalité triangulaire)

**p-normes** — Pour $p \geq 1$ :
$$\|x\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}, \qquad \|x\|_\infty = \max_i |x_i|$$

**Satz 1.5 (Équivalence)** — En dimension finie, toutes les normes sont équivalentes : $\exists\, L, M > 0$ tels que $\|x\|_a \leq L \|x\|_b$ et $\|x\|_b \leq M \|x\|_a$.

**Beispiel 1.6** — Pour $\|\cdot\|_1$ et $\|\cdot\|_\infty$ : $\|x\|_\infty \leq \|x\|_1 \leq n \cdot \|x\|_\infty$.

**Übung 1.3** — Les boules unitaires $\{x : \|x\|_p \leq 1\}$ passent du losange ($p=1$) au disque ($p=2$) au carré ($p=\infty$).

## 1.3 Conditionnement

**Définition 1.7** — Condition absolue : $\text{cond}_{abs} = \sup \frac{\|f(\tilde{x}) - f(x)\|}{\|\tilde{x} - x\|}$.

**Définition 1.8** — Condition relative : $\text{cond}_{rel} = \text{cond}_{abs} \cdot \frac{\|x\|}{\|f(x)\|}$.

**Satz 1.9** — Pour $f : \mathbb{R} \to \mathbb{R}$ différentiable :
$$\text{cond}_{abs}(f, x) = |f'(x)|, \qquad \text{cond}_{rel}(f, x) = \frac{|f'(x)| \cdot |x|}{|f(x)|}$$

**Auslöschung (§1.3.3)** — La soustraction $x_1 - x_2$ a $\text{cond}_{rel} = \frac{|x_1| + |x_2|}{|x_1 - x_2|}$. Quand $x_1 \approx x_2$, le conditionnement explose.

## 1.4 Stabilité

**Différence clé** — Le *conditionnement* est une propriété du **problème** (mathématique). La *stabilité* est une propriété de l'**algorithme** (implémentation).

**Beispiel 1.13** — $f(x) = \sqrt{x^2+1} - x$ : problème bien conditionné, mais algorithme naïf instable. Reformulation stable : $f(x) = 1/(\sqrt{x^2+1} + x)$.

**IEEE 754 double precision (Beispiel 1.12)** :
- 52 bits de mantisse → $\varepsilon_{mach} = 2^{-52} \approx 1.11 \times 10^{-16}$
- Overflow à $\approx 10^{308}$, underflow à $\approx 10^{-308}$

**Règle d'or (§1.4.4)** — Éviter les sous-problèmes mal conditionnés dans la décomposition d'un algorithme. En particulier : ne jamais soustraire des nombres presque égaux.

## Programmes associés

| Module | Contenu |
|---|---|
| `vector_norms.py` | p-normes, boules unitaires, équivalence |
| `cancellation.py` | 4 exemples d'Auslöschung + reformulations stables |
| `floating_point.py` | IEEE 754, ε_mach, Kahan summation |
| `condition_number.py` | cond_rel pour √, exp, ln, sin, soustraction |
