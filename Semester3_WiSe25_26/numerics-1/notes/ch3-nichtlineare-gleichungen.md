# Chapitre 3 — Nichtlineare Gleichungen

> **Résumé de révision** — Kröger, *Numerische Mathematik 1 für AMP*, §3.1–3.2

## 3.1 Équations scalaires $f(x) = 0$

### Conditionnement (§3.1.2)

$$\text{cond}_{abs} = \frac{1}{|f'(x^*)|}.$$

Si $f'(x^*) = 0$ (nullstelle multiple) → problème **mal conditionné**.

### Newton (§3.1.3, Satz 3.5)

$$x^+ = x - \frac{f(x)}{f'(x)}$$

**Convergence quadratique** ($\alpha = 2$) si :
1. $f'(x^*) \neq 0$ (nullstelle simple)
2. $f'$ Lipschitz-continue (Lemma 3.4)
3. $x_0$ suffisamment proche de $x^*$

**Ordre expérimental (§3.1.4)** :
$$\alpha \approx \frac{\ln(e_k / e_{k+1})}{\ln(e_{k-1} / e_k)}$$

### Critères d'arrêt (§3.1.5)

| Critère | Formule | Piège |
|---|---|---|
| Résidu | $|f(x^+)| \leq \tau_1$ | petit résidu $\neq$ solution précise |
| Relatif | $|x^+ - x| / |x^+| \leq \tau_2$ | $\tau_2 \approx \sqrt{\varepsilon_{mach}}$ |
| Absolu | $|x^+ - x| \leq \tau_3$ | utile si $x^* \approx 0$ |

**Recommandation** : combiner les 3.

### Bissection (§3.1.6)

Convergence **globale** garantie si $f(a) \cdot f(b) < 0$. Linéaire : 1 bit par itération ($\log_{10} 2 \approx 0.301$ chiffres/it.).

### Sécante (§3.1.7)

$$x^+ = x - f(x) \cdot \frac{x - x_{k-1}}{f(x) - f(x_{k-1})}$$

Ordre $\alpha = \phi = \frac{1+\sqrt{5}}{2} \approx 1.618$ (nombre d'or). Pas besoin de $f'$.

### Regula Falsi vs Illinois

- **Regula Falsi** : comme la sécante mais garde le changement de signe → stuck endpoint
- **Illinois** : divise $f$ du bord collé par 2 → convergence $\alpha \approx 1.442$

### Newton modifié (§3.1.8)

Pour nullstelle de multiplicité $m$ : $x^+ = x - m \cdot \frac{f(x)}{f'(x)}$. Retrouve $\alpha = 2$ si $m$ correct.

## 3.2 Systèmes $f(x) = 0$, $f : \mathbb{R}^n \to \mathbb{R}^n$

**Newton multivarié** — Résoudre le système linéaire (formule 3.13) :
$$J(x) \, s = -f(x), \qquad x^+ = x + s$$

**Ne jamais** calculer $J^{-1}$ explicitement — résoudre le système linéaire.

Si $J$ n'est pas disponible : approximation par **différences finies** ($n+1$ évaluations de $f$ par pas).

## Tableau comparatif

| Méthode | Ordre $\alpha$ | Coût/it. | Besoin de $f'$ | Convergence |
|---|---|---|---|---|
| Newton | 2 | $f + f'$ | oui | locale |
| Sécante | 1.618 | $f$ | non | locale |
| Regula Falsi | 1 (stuck) | $f$ | non | globale (encadrement) |
| Illinois | 1.442 | $f$ | non | globale |
| Bissection | 1 | $f$ | non | **globale** (garantie) |

## Programmes associés

| Module | Contenu |
|---|---|
| `newton_scalar.py` | Newton + sécante + bissection + ordre exp. |
| `newton_systems.py` | Newton multivarié, Jacobienne analytique/DF |
| `derivative_free.py` | Regula Falsi, Illinois, hybride |
| `stopping_criteria.py` | Comparaison des 3 critères |
