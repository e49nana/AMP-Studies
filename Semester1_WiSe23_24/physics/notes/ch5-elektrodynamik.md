# Chapitre 5 — Elektrodynamik

> **Résumé de révision** — Physik für AMP

## 5.1 Circuits

**Ohm :** $V = RI$. **Kirchhoff :** nœuds ($\sum I = 0$), mailles ($\sum V = 0$).

| Configuration | Formule |
|---|---|
| Série | $R_{tot} = R_1 + R_2 + \cdots$ |
| Parallèle | $1/R_{tot} = 1/R_1 + 1/R_2 + \cdots$ |

**RC :** $\tau = RC$, charge $V_C = V_s(1-e^{-t/\tau})$, à $t = \tau$ : 63%.

**RL :** $\tau = L/R$, établissement $I = (V/R)(1-e^{-t/\tau})$.

**RLC :** $\omega_0 = 1/\sqrt{LC}$, 3 régimes selon $R$ vs $2\sqrt{L/C}$.

## 5.2 Champ magnétique

$$\vec{F} = q\vec{v} \times \vec{B} \qquad \text{(force de Lorentz)}$$

| Source | Champ |
|---|---|
| Fil infini | $B = \mu_0 I/(2\pi r)$ |
| Solénoïde | $B = \mu_0 n I$ (uniforme) |
| Boucle (axe) | $B = \mu_0 IR^2/[2(R^2+z^2)^{3/2}]$ |

**Cyclotron :** $r = mv/(qB)$, $\omega_c = qB/m$ (indépendant de $v$ !).

**Force entre fils :** $F/L = \mu_0 I_1 I_2/(2\pi d)$.

## 5.3 Ondes EM

$$c = \frac{1}{\sqrt{\mu_0\varepsilon_0}} \approx 3 \times 10^8 \text{ m/s}$$

$\vec{E} \perp \vec{B} \perp \vec{k}$, $E/B = c$.

**Poynting :** $\vec{S} = \vec{E} \times \vec{B}/\mu_0$, intensité $I = \frac{1}{2}\varepsilon_0 c E_0^2$.

## 5.4 Induction

$$\varepsilon = -\frac{d\Phi_B}{dt} \qquad \text{(Faraday)}$$

**Lenz :** le courant induit s'oppose à la variation de flux.

| Application | Formule |
|---|---|
| Générateur AC | $\varepsilon = NBA\omega\sin(\omega t)$ |
| Inductance solénoïde | $L = \mu_0 n^2 V$ |
| Énergie magnétique | $U = \frac{1}{2}LI^2$ |
| Transformateur | $V_2/V_1 = N_2/N_1$ |

## Programmes associés

| Module | Contenu |
|---|---|
| `circuits.py` | Ohm, Kirchhoff, RC/RL/RLC |
| `magnetic_field.py` | Biot-Savart, Lorentz, cyclotron |
| `electromagnetic_waves.py` | Ondes EM, spectre, Poynting |
| `induction.py` | Faraday, générateur, transformateur |
