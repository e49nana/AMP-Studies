# Chapitre 1 — Kinematik

> **Résumé de révision** — Physik für AMP

## 1.1 Mouvement rectiligne

- **Position :** $x(t)$
- **Vitesse :** $v = dx/dt$
- **Accélération :** $a = dv/dt = d^2x/dt^2$

**MRUA** (mouvement rectiligne uniformément accéléré) :
$$x(t) = x_0 + v_0 t + \tfrac{1}{2}at^2, \qquad v(t) = v_0 + at, \qquad v^2 = v_0^2 + 2a\Delta x$$

## 1.2 Tir oblique (Wurfbewegung)

$$x(t) = v_0\cos\alpha \cdot t, \qquad y(t) = v_0\sin\alpha \cdot t - \tfrac{1}{2}gt^2$$

| Grandeur | Formule |
|---|---|
| Portée | $R = v_0^2\sin(2\alpha)/g$ |
| Hauteur max | $H = v_0^2\sin^2\alpha/(2g)$ |
| Temps de vol | $T = 2v_0\sin\alpha/g$ |
| Angle optimal | $45°$ (sans frottement) |

**Enveloppe de sécurité :** $y_{max}(x) = v_0^2/(2g) - gx^2/(2v_0^2)$

## 1.3 Mouvement circulaire (Kreisbewegung)

$$v = \omega R, \qquad a_c = \omega^2 R = v^2/R, \qquad T = 2\pi/\omega$$

- $a_c$ dirigée vers le centre (centripète)
- $F_c = ma_c = m\omega^2 R$
- Virage : $v_{max} = \sqrt{\mu g R}$
- Satellite : $v = \sqrt{GM/r}$, orbite géostationnaire à $h \approx 35\,786$ km

## 1.4 Référentiels et mouvement relatif

**Transformation de Galilée :** $\vec{v}_{abs} = \vec{v}_{rel} + \vec{v}_{ref}$

**Bateau dans une rivière :** correction d'angle $\alpha = \arcsin(v_{courant}/v_{bateau})$ pour traverser droit.

## 1.5 Courbes paramétriques

- **Cycloïde :** $x = R(t - \sin t)$, $y = R(1 - \cos t)$ (brachistochrone)
- **Lissajous :** $x = \sin(at)$, $y = \sin(bt + \delta)$ (rapport de fréquences)
- **Épicycloïde / Hypocycloïde :** spirographe, astroïde ($R/r = 4$)

## Programmes associés

| Module | Contenu |
|---|---|
| `projectile_motion.py` | Tir oblique, frottement, enveloppe |
| `circular_motion.py` | MCU, satellites, virages |
| `relative_motion.py` | Galilée, rivière, avion |
| `parametric_curves.py` | Cycloïde, Lissajous, spirographe |
