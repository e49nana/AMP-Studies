# Chapitre 4 — Statistik

> **Résumé de révision** — Stochastik für AMP, S3

## 4.1 Statistiques descriptives

**Position :** moyenne $\bar{x}$, médiane, mode, quantiles ($Q_1, Q_2, Q_3$)

**Dispersion :** variance $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$, écart-type $s$, IQR $= Q_3 - Q_1$

**Forme :**
- Asymétrie (skewness) : $> 0$ = queue à droite, $< 0$ = queue à gauche
- Aplatissement (kurtosis) : $> 0$ = queues lourdes, $< 0$ = queues légères (réf. normale = 0)

**Robustesse :** médiane et IQR résistent aux outliers, pas la moyenne et l'écart-type.

## 4.2 Estimation

**MLE (Maximum de vraisemblance) :**

| Distribution | $\hat{\theta}_{MLE}$ |
|---|---|
| $N(\mu, \sigma^2)$ | $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i-\bar{x})^2$ (biaisé !) |
| $\text{Exp}(\lambda)$ | $\hat{\lambda} = 1/\bar{x}$ |
| $\text{Bernoulli}(p)$ | $\hat{p} = \bar{x}$ |
| $\text{Poisson}(\lambda)$ | $\hat{\lambda} = \bar{x}$ |

**Biais :** $\hat{\sigma}^2_{MLE}$ sous-estime $\sigma^2$. Correction de Bessel : diviser par $n-1$ au lieu de $n$.

**Intervalles de confiance :**

| Cas | Formule | Distribution |
|---|---|---|
| $\mu$, $\sigma$ connu | $\bar{x} \pm z_{\alpha/2} \cdot \sigma/\sqrt{n}$ | $Z$ |
| $\mu$, $\sigma$ inconnu | $\bar{x} \pm t_{\alpha/2, n-1} \cdot s/\sqrt{n}$ | Student $t$ |
| proportion $p$ | $\hat{p} \pm z_{\alpha/2} \cdot \sqrt{\hat{p}(1-\hat{p})/n}$ | $Z$ (Wald) |

**Taille d'échantillon :** $n \geq (z_{\alpha/2} \cdot \sigma / \text{marge})^2$

## 4.3 Tests d'hypothèses

**Logique :** $H_0$ (hypothèse nulle) vs $H_1$ (alternative). On rejette $H_0$ si $p < \alpha$.

| Erreur | Définition | Probabilité |
|---|---|---|
| Type I (faux positif) | rejeter $H_0$ alors que $H_0$ est vraie | $\alpha$ |
| Type II (faux négatif) | ne pas rejeter $H_0$ alors que $H_1$ est vraie | $\beta$ |

**Puissance** $= 1 - \beta = P(\text{rejeter } H_0 | H_1 \text{ vraie})$. Augmente avec $n$, la taille d'effet, et $\alpha$.

**Tests classiques :**

| Test | Hypothèse | Statistique |
|---|---|---|
| Z-test | $\mu = \mu_0$ ($\sigma$ connu) | $Z = (\bar{x}-\mu_0)/(\sigma/\sqrt{n})$ |
| t-test (1 éch.) | $\mu = \mu_0$ ($\sigma$ inconnu) | $t = (\bar{x}-\mu_0)/(s/\sqrt{n})$ |
| Welch t-test | $\mu_1 = \mu_2$ | $t = (\bar{x}_1-\bar{x}_2)/\text{SE}$ |
| $\chi^2$ indépendance | $X \perp Y$ | $\sum (O-E)^2/E$ |

**p-value :** sous $H_0$, les p-values suivent $U(0,1)$. ~5% de faux positifs par hasard.

## 4.4 Régression linéaire

$$\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x, \qquad \hat{\beta}_1 = \frac{S_{xy}}{S_{xx}}, \qquad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

**$R^2$** $= 1 - SSE/SST$ : proportion de variance expliquée ($0$ à $1$).

**Test de significativité :** $H_0 : \beta_1 = 0$ (pas de relation). $t = \hat{\beta}_1 / \text{SE}(\hat{\beta}_1)$.

**Bandes :**
- IC pour $E[Y|x]$ (confiance) : plus étroit
- Bande de prédiction pour une nouvelle $Y$ : plus large (inclut $\sigma^2$)

**Résidus :** vérifier normalité (QQ-plot), homoscédasticité, indépendance.

## Programmes associés

| Module | Contenu |
|---|---|
| `descriptive_stats.py` | Moyenne, médiane, quantiles, skewness, boxplot |
| `estimation.py` | MLE, IC (Z/t), taille d'échantillon, biais |
| `hypothesis_testing.py` | Z/t/χ² tests, puissance, p-value |
| `regression_stats.py` | OLS, R², IC coefficients, bandes, résidus |
