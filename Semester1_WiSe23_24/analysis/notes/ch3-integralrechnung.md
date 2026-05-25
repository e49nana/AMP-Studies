# Chapitre 3 — Integralrechnung

> **Résumé de révision** — Analysis / Mathematik 1 für AMP, S1

## 3.1 Intégrale de Riemann

$$\int_a^b f(x)\,dx = \lim_{n \to \infty} \sum_{i=1}^n f(x_i^*) \Delta x$$

**Sommes de Riemann :**

| Méthode | Point d'évaluation | Erreur |
|---|---|---|
| Gauche | $x_i$ | $O(1/n)$ |
| Droite | $x_{i+1}$ | $O(1/n)$ |
| Milieu | $(x_i + x_{i+1})/2$ | $O(1/n^2)$ |

## 3.2 Hauptsatz (théorème fondamental)

$$\frac{d}{dx} \int_a^x f(t)\,dt = f(x) \qquad \text{et} \qquad \int_a^b f(x)\,dx = F(b) - F(a)$$

## 3.3 Intégration numérique

| Méthode | Formule | Erreur | Exacte pour |
|---|---|---|---|
| Trapèzes | $\frac{h}{2}[f_0 + 2f_1 + \cdots + f_n]$ | $O(h^2)$ | deg $\leq 1$ |
| Simpson | $\frac{h}{3}[f_0 + 4f_1 + 2f_2 + \cdots + f_n]$ | $O(h^4)$ | deg $\leq 3$ |
| Gauss-Legendre ($n$ pts) | nœuds/poids optimaux | $O(h^{2n})$ | deg $\leq 2n-1$ |

## 3.4 Techniques d'intégration

- **Substitution :** $\int f(g(x))g'(x)\,dx = \int f(u)\,du$ avec $u = g(x)$
- **Par parties :** $\int u\,dv = uv - \int v\,du$
- **Fractions partielles :** décomposer $P(x)/Q(x)$ en somme de fractions simples

## 3.5 Intégrales impropres

- **Type 1 (borne infinie) :** $\int_a^\infty f = \lim_{b \to \infty} \int_a^b f$
- **Type 2 (singularité) :** $\int_a^b f = \lim_{\varepsilon \to 0^+} \int_{a+\varepsilon}^b f$
- **p-intégrale :** $\int_1^\infty x^{-p}\,dx$ converge ssi $p > 1$, valeur $= 1/(p-1)$

**Intégrales remarquables :**
- $\int_0^\infty e^{-x}\,dx = 1$
- $\int_0^\infty e^{-x^2}\,dx = \sqrt{\pi}/2$ (Gauss)
- $\Gamma(n) = \int_0^\infty t^{n-1}e^{-t}\,dt = (n-1)!$

## 3.6 Applications

- **Aire :** $A = \int_a^b |f(x) - g(x)|\,dx$
- **Volume de révolution :** $V = \pi \int_a^b f(x)^2\,dx$ (disques)
- **Longueur d'arc :** $L = \int_a^b \sqrt{1 + f'(x)^2}\,dx$

## Programmes associés

| Module | Contenu |
|---|---|
| `riemann_sums.py` | Sommes gauche/droite/milieu, visualisation |
| `numerical_integration.py` | Trapèzes, Simpson, Gauss-Legendre |
| `antiderivatives.py` | Hauptsatz, substitution, parties, fractions partielles |
| `improper_integrals.py` | Types 1 et 2, p-intégrale, Gamma |
| `applications_integrals.py` | Aires, volumes, longueurs d'arc |
