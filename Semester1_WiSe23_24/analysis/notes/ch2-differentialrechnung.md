# Chapitre 2 — Differentialrechnung

> **Résumé de révision** — Analysis / Mathematik 1 für AMP, S1

## 2.1 Dérivée

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Interprétation :** pente de la tangente, taux de variation instantané.

**Différences finies :**

| Méthode | Formule | Erreur |
|---|---|---|
| Progressive | $(f(x+h) - f(x))/h$ | $O(h)$ |
| Rétrograde | $(f(x) - f(x-h))/h$ | $O(h)$ |
| Centrée | $(f(x+h) - f(x-h))/(2h)$ | $O(h^2)$ |

## 2.2 Règles de dérivation

| Règle | Formule |
|---|---|
| Somme | $(f+g)' = f' + g'$ |
| Produit | $(fg)' = f'g + fg'$ |
| Quotient | $(f/g)' = (f'g - fg')/g^2$ |
| Chaîne | $(f \circ g)' = f'(g(x)) \cdot g'(x)$ |
| Inverse | $(f^{-1})'(y) = 1/f'(x)$ |

## 2.3 Kurvendiskussion

1. **Domaine** de définition
2. **Nullstellen** : $f(x) = 0$
3. **Extrema** : $f'(x_0) = 0$ et $f''(x_0) \neq 0$
   - $f''(x_0) > 0$ → minimum local
   - $f''(x_0) < 0$ → maximum local
4. **Wendepunkte** (inflexion) : $f''(x_0) = 0$ avec changement de signe
5. **Monotonie** : $f' > 0$ → croissante, $f' < 0$ → décroissante
6. **Convexité** : $f'' > 0$ → convexe, $f'' < 0$ → concave
7. **Asymptotes** : $\lim_{x \to \pm\infty} f(x)$

## 2.4 Optimisation

- **Newton pour $f'(x) = 0$** : $x^+ = x - f'(x)/f''(x)$ (convergence quadratique)
- **Descente de gradient** : $x^+ = x - \alpha f'(x)$ (convergence linéaire, $\alpha$ critique)
- **Section dorée** : sans dérivée, réduit l'intervalle par $\phi = (\sqrt{5}-1)/2$

## 2.5 Règle de L'Hôpital

Pour les formes $0/0$ ou $\infty/\infty$ :
$$\lim \frac{f(x)}{g(x)} = \lim \frac{f'(x)}{g'(x)}$$

**Transformations :** $0 \cdot \infty \to 0/0$, $1^\infty \to e^{(\cdot)}$, $0^0 \to e^{(\cdot)}$.

**Attention :** si $\lim f'/g'$ n'existe pas, L'Hôpital ne conclut rien.

## Programmes associés

| Module | Contenu |
|---|---|
| `derivatives.py` | Différences finies, vérification des règles |
| `curve_analysis.py` | Kurvendiskussion automatique |
| `optimization.py` | Newton, gradient, section dorée |
| `taylor_approximation.py` | Polynôme de Taylor, reste de Lagrange |
| `lhopital.py` | L'Hôpital, formes indéterminées |
