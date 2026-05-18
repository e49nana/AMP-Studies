# Physik für AMP — Python Simulations

**Course:** Physik für AMP  
**Semester:** WiSe 2023/2024 – SoSe 2024 (S1-S2)  
**Author:** Emmanuel Nanan — TH Nürnberg

Physics simulations and visualizations in Python.

## Structure

```
code/
├── ch1-kinematik/          # Projectile, circular motion, reference frames
├── ch2-dynamik/            # Newton's laws, oscillations, gravity, collisions (coming)
├── ch3-energie/            # Work-energy, conservation, power (coming)
├── ch4-elektrostatik/      # Coulomb, E-field, Gauss, potential (coming)
├── ch5-elektrodynamik/     # Circuits, Biot-Savart, Faraday (coming)
├── ch6-wellen-optik/       # Waves, interference, optics (coming)
└── requirements.txt
```

## Chapter 1 — Kinematik (4 modules)

| Module | Topics |
|---|---|
| `projectile_motion.py` | Parabolic trajectory, drag, angle optimization, safety envelope |
| `circular_motion.py` | UCM, centripetal acceleration, satellites, banked curves |
| `relative_motion.py` | Galilean transform, river crossing, wind correction |
| `parametric_curves.py` | Cycloid, Lissajous, spirograph, curvature |

## Quick Start

```bash
pip install numpy matplotlib scipy
cd ch1-kinematik
python projectile_motion.py
```
