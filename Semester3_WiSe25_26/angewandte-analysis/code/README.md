# Angewandte Analysis — Python Implementations

**Course:** Angewandte Analysis für AMP  
**Semester:** WiSe 2025/2026 – SoSe 2026 (S3-S4)  
**Author:** Emmanuel Nanan — TH Nürnberg

Applied analysis: multivariable calculus, functional analysis, PDEs, transforms.

## Structure

```
code/
├── ch1-mehrdimensionale-analysis/  # Gradient, Hessian, optimization, vector calculus
├── ch2-funktionalanalysis/         # Metric/normed/Hilbert spaces (coming)
├── ch3-pde/                        # Heat, wave, Laplace equations (coming)
├── ch4-integraltransformationen/   # Fourier/Laplace transform, FFT (coming)
└── requirements.txt
```

## Chapter 1 — Mehrdimensionale Analysis (4 modules)

| Module | Topics |
|---|---|
| `partial_derivatives.py` | Partial derivatives, gradient, Hessian, Jacobian, critical point classification |
| `multivariable_optimization.py` | Gradient descent (with backtracking), Newton, Lagrange multipliers |
| `vector_calculus.py` | Divergence, curl, Laplacian, conservative fields, 2D/3D |
| `multiple_integrals.py` | Double/triple integrals, polar/spherical coordinates, center of mass |

## Quick Start

```bash
pip install numpy matplotlib scipy
cd ch1-mehrdimensionale-analysis
python partial_derivatives.py
```
