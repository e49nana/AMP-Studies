# Analysis / Mathematik S1 — Python Implementations

**Course:** Mathematik 1 für AMP  
**Semester:** WiSe 2023/2024 (S1)  
**Author:** Emmanuel Nanan — TH Nürnberg

From-scratch implementations of analysis concepts in Python, with visualizations.

## Structure

```
code/
├── ch1-folgen-reihen/         # Sequences, series, Taylor, limits
├── ch2-differentialrechnung/  # Derivatives, curve analysis, optimization (coming)
├── ch3-integralrechnung/      # Riemann, numerical integration, applications (coming)
├── ch4-ode/                   # Euler, RK4, phase portraits (coming)
├── ch5-spezielle-funktionen/  # Fourier, power series (coming)
└── requirements.txt
```

## Chapter 1 — Folgen und Reihen (4 modules)

| Module | Topics |
|---|---|
| `sequences.py` | Arithmetic/geometric, Heron √2, (1+1/n)^n → e, convergence analysis |
| `series.py` | Geometric, harmonic, Riemann, d'Alembert/Cauchy/Leibniz criteria |
| `taylor_series.py` | Taylor expansions (exp, sin, cos, ln), remainder, radius of convergence |
| `limits.py` | Numerical limits, ε-δ, indeterminate forms, growth comparison |

## Quick Start

```bash
pip install numpy matplotlib
cd ch1-folgen-reihen
python sequences.py
```
