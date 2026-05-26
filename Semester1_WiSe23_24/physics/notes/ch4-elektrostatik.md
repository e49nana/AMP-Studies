# Chapitre 4 — Elektrostatik

> **Résumé de révision** — Physik für AMP

## 4.1 Loi de Coulomb

$$F = k\frac{q_1 q_2}{r^2}, \qquad k = \frac{1}{4\pi\varepsilon_0} \approx 8.99 \times 10^9 \text{ N·m²/C²}$$

- Même signe → répulsion, signes opposés → attraction
- $F_{elec}/F_{grav} \approx 10^{39}$ (proton-électron)

## 4.2 Champ électrique

$$\vec{E} = \frac{\vec{F}}{q} = k\frac{Q}{r^2}\hat{r}$$

**Superposition :** $\vec{E}_{total} = \sum \vec{E}_i$ (les champs s'additionnent vectoriellement).

**Dipôle :** sur l'axe, $E \propto 1/r^3$ (décroît plus vite qu'une charge seule).

## 4.3 Loi de Gauss

$$\Phi_E = \oint \vec{E} \cdot d\vec{A} = \frac{Q_{enc}}{\varepsilon_0}$$

| Symétrie | Géométrie de Gauss | Champ |
|---|---|---|
| Sphérique | sphère | $E = kQ/r^2$ (extérieur), $E = kQr/R^3$ (intérieur) |
| Cylindrique | cylindre | $E = \lambda/(2\pi\varepsilon_0 r)$ |
| Plane | boîte | $E = \sigma/(2\varepsilon_0)$ (constant !) |

**Conducteur :** $E = 0$ à l'intérieur, charges sur la surface, $E = \sigma/\varepsilon_0$ juste dehors.

## 4.4 Potentiel électrique

$$V = k\frac{Q}{r}, \qquad \vec{E} = -\nabla V, \qquad U = qV$$

- Équipotentielles ⊥ lignes de champ
- Condensateur plan : $V$ linéaire, $E = V/d$ constant, $C = \varepsilon_0 A/d$

## Programmes associés

| Module | Contenu |
|---|---|
| `coulomb.py` | Force, superposition, équilibre |
| `electric_field.py` | Lignes de champ, dipôle, streamplot |
| `gauss_law.py` | Symétries, sphère/cylindre/plan |
| `electric_potential.py` | V, équipotentielles, E = -∇V |
