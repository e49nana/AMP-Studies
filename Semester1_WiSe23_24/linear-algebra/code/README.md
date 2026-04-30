# Lineare Algebra — Python Implementations

**Course:** Lineare Algebra für AMP  
**Semester:** WiSe 2023/2024 (S1)  
**Author:** Emmanuel Nanan — TH Nürnberg

From-scratch implementations of linear algebra algorithms in Python, compared with NumPy.

## Structure

```
code/
├── ch1-vektoren/           # Vectors, dot/cross product, lines, planes, subspaces
├── ch2-matrizen/           # Matrix operations, types, linear maps (coming)
├── ch3-gleichungssysteme/  # Gauss-Jordan, RREF, rank (coming)
├── ch4-determinanten/      # Leibniz, Sarrus, Laplace, Cramer (coming)
├── ch5-eigenwerte/         # Eigenvalues, diagonalization, spectral theorem (coming)
├── ch6-anwendungen/        # Markov, SVD, transformations (coming)
└── requirements.txt
```

## Chapter 1 — Vektoren (6 modules)

| Module | Topics |
|---|---|
| `vectors_2d_3d.py` | Vector class, operations, combinations, 2D/3D visualization |
| `dot_product.py` | Dot product, angles, orthogonal projection, Cauchy-Schwarz |
| `cross_product.py` | Cross product, areas, volumes, Spatprodukt |
| `lines_planes.py` | Lines/planes in R³, intersections, distances |
| `linear_independence.py` | Independence test, rank, basis extraction, coordinates |
| `subspaces.py` | Kern, Bild, RREF, Rangsatz (rank-nullity theorem) |

## Quick Start

```bash
pip install numpy matplotlib
cd ch1-vektoren
python vectors_2d_3d.py
```
