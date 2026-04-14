"""
Linear Algebra Essentials
==========================
Core operations for numerical computing and functional analysis.
Covers: Vector spaces, norms, inner products, matrix decompositions.

Author: Emmanuel Nana Nana
Repo: AMP-Studies / Scientific-Simulation-Project
"""

import numpy as np
from typing import Tuple, Optional, List
from numpy.linalg import norm, eig, svd, qr, cholesky, det, inv, matrix_rank


# =============================================================================
# VECTOR NORMS
# =============================================================================

def l1_norm(x: np.ndarray) -> float:
    """
    L1 norm (Manhattan/Taxicab norm).
    ||x||₁ = Σ|xᵢ|
    """
    return np.sum(np.abs(x))


def l2_norm(x: np.ndarray) -> float:
    """
    L2 norm (Euclidean norm).
    ||x||₂ = √(Σxᵢ²)
    """
    return np.sqrt(np.sum(x ** 2))


def linf_norm(x: np.ndarray) -> float:
    """
    L∞ norm (Maximum/Chebyshev norm).
    ||x||∞ = max|xᵢ|
    """
    return np.max(np.abs(x))


def lp_norm(x: np.ndarray, p: float) -> float:
    """
    General Lp norm.
    ||x||ₚ = (Σ|xᵢ|ᵖ)^(1/p)
    """
    return np.sum(np.abs(x) ** p) ** (1 / p)


def norm_comparison(x: np.ndarray) -> dict:
    """Compare all common norms for a vector."""
    return {
        'L1 (Manhattan)': l1_norm(x),
        'L2 (Euclidean)': l2_norm(x),
        'L∞ (Maximum)': linf_norm(x),
        'L3': lp_norm(x, 3),
        'L0.5 (quasi-norm)': lp_norm(x, 0.5)
    }


# =============================================================================
# MATRIX NORMS
# =============================================================================

def frobenius_norm(A: np.ndarray) -> float:
    """
    Frobenius norm (matrix L2 norm).
    ||A||_F = √(Σᵢⱼ aᵢⱼ²) = √(trace(AᵀA))
    """
    return np.sqrt(np.sum(A ** 2))


def spectral_norm(A: np.ndarray) -> float:
    """
    Spectral norm (largest singular value).
    ||A||₂ = σ_max(A)
    
    Also called induced 2-norm or operator norm.
    """
    singular_values = svd(A, compute_uv=False)
    return singular_values[0]


def matrix_1_norm(A: np.ndarray) -> float:
    """
    Matrix 1-norm (maximum absolute column sum).
    ||A||₁ = maxⱼ Σᵢ|aᵢⱼ|
    """
    return np.max(np.sum(np.abs(A), axis=0))


def matrix_inf_norm(A: np.ndarray) -> float:
    """
    Matrix ∞-norm (maximum absolute row sum).
    ||A||∞ = maxᵢ Σⱼ|aᵢⱼ|
    """
    return np.max(np.sum(np.abs(A), axis=1))


# =============================================================================
# INNER PRODUCTS
# =============================================================================

def inner_product(x: np.ndarray, y: np.ndarray) -> float:
    """
    Standard inner product (dot product).
    ⟨x, y⟩ = Σ xᵢyᵢ = xᵀy
    """
    return np.dot(x, y)


def weighted_inner_product(x: np.ndarray, y: np.ndarray, W: np.ndarray) -> float:
    """
    Weighted inner product.
    ⟨x, y⟩_W = xᵀWy
    
    W must be symmetric positive definite.
    """
    return x @ W @ y


def angle_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    """
    Angle between two vectors in radians.
    θ = arccos(⟨x,y⟩ / (||x|| ||y||))
    """
    cos_theta = np.dot(x, y) / (l2_norm(x) * l2_norm(y))
    # Clip to handle numerical errors
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.arccos(cos_theta)


def are_orthogonal(x: np.ndarray, y: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if two vectors are orthogonal."""
    return np.abs(np.dot(x, y)) < tol


# =============================================================================
# PROJECTIONS
# =============================================================================

def project_onto_vector(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Project vector v onto vector u.
    proj_u(v) = (⟨v,u⟩ / ⟨u,u⟩) · u
    """
    return (np.dot(v, u) / np.dot(u, u)) * u


def project_onto_subspace(v: np.ndarray, basis: List[np.ndarray]) -> np.ndarray:
    """
    Project vector v onto subspace spanned by orthonormal basis.
    
    Parameters
    ----------
    v : vector to project
    basis : list of orthonormal basis vectors
    """
    projection = np.zeros_like(v, dtype=float)
    for u in basis:
        projection += np.dot(v, u) * u
    return projection


def gram_schmidt(vectors: List[np.ndarray]) -> List[np.ndarray]:
    """
    Gram-Schmidt orthonormalization.
    
    Transforms a set of linearly independent vectors into
    an orthonormal basis.
    
    Parameters
    ----------
    vectors : list of linearly independent vectors
    
    Returns
    -------
    list of orthonormal vectors
    """
    orthonormal = []
    
    for v in vectors:
        # Subtract projections onto previous vectors
        w = v.copy().astype(float)
        for u in orthonormal:
            w -= np.dot(v, u) * u
        
        # Normalize
        w_norm = l2_norm(w)
        if w_norm > 1e-10:
            orthonormal.append(w / w_norm)
    
    return orthonormal


# =============================================================================
# MATRIX DECOMPOSITIONS
# =============================================================================

def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LU decomposition with partial pivoting.
    PA = LU
    
    Returns
    -------
    P : permutation matrix
    L : lower triangular (unit diagonal)
    U : upper triangular
    """
    from scipy.linalg import lu
    P, L, U = lu(A)
    return P, L, U


def qr_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    QR decomposition.
    A = QR
    
    Q : orthogonal matrix (Qᵀ = Q⁻¹)
    R : upper triangular
    
    Applications: solving least squares, eigenvalue algorithms
    """
    return qr(A)


def svd_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Singular Value Decomposition.
    A = UΣVᵀ
    
    U : left singular vectors (orthogonal)
    Σ : diagonal matrix of singular values
    Vᵀ : right singular vectors (orthogonal)
    
    Applications: dimensionality reduction, pseudoinverse, rank
    """
    U, s, Vt = svd(A)
    return U, s, Vt


def eigendecomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigendecomposition.
    A = VΛV⁻¹
    
    Returns
    -------
    eigenvalues : array of eigenvalues
    eigenvectors : matrix where column i is eigenvector for eigenvalue i
    """
    eigenvalues, eigenvectors = eig(A)
    return eigenvalues, eigenvectors


def cholesky_decomposition(A: np.ndarray) -> np.ndarray:
    """
    Cholesky decomposition for symmetric positive definite matrices.
    A = LLᵀ
    
    L : lower triangular
    
    More efficient than LU for SPD matrices.
    """
    return cholesky(A)


# =============================================================================
# MATRIX PROPERTIES
# =============================================================================

def condition_number(A: np.ndarray, p: int = 2) -> float:
    """
    Condition number of a matrix.
    κ(A) = ||A|| · ||A⁻¹||
    
    For p=2: κ(A) = σ_max / σ_min
    
    Large condition number → ill-conditioned (sensitive to errors)
    """
    return np.linalg.cond(A, p)


def is_symmetric(A: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if matrix is symmetric: A = Aᵀ"""
    return np.allclose(A, A.T, atol=tol)


def is_positive_definite(A: np.ndarray) -> bool:
    """
    Check if matrix is positive definite.
    A is PD iff all eigenvalues are positive.
    """
    if not is_symmetric(A):
        return False
    eigenvalues = np.linalg.eigvalsh(A)
    return np.all(eigenvalues > 0)


def is_orthogonal(A: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if matrix is orthogonal: AᵀA = I"""
    n = A.shape[0]
    return np.allclose(A.T @ A, np.eye(n), atol=tol)


def spectral_radius(A: np.ndarray) -> float:
    """
    Spectral radius: maximum absolute eigenvalue.
    ρ(A) = max|λᵢ|
    
    Important for iterative method convergence.
    """
    eigenvalues = np.linalg.eigvals(A)
    return np.max(np.abs(eigenvalues))


def nullspace(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Find orthonormal basis for null space of A.
    Returns vectors x such that Ax = 0.
    """
    U, s, Vt = svd(A)
    null_mask = s < tol
    null_space = Vt[null_mask].T
    return null_space


def column_space(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Find orthonormal basis for column space (range) of A.
    """
    U, s, Vt = svd(A)
    rank = np.sum(s > tol)
    return U[:, :rank]


# =============================================================================
# APPLICATIONS
# =============================================================================

def solve_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve overdetermined system Ax ≈ b in least squares sense.
    Minimizes ||Ax - b||₂
    
    Solution: x = (AᵀA)⁻¹Aᵀb = A⁺b (pseudoinverse)
    """
    return np.linalg.lstsq(A, b, rcond=None)[0]


def low_rank_approximation(A: np.ndarray, k: int) -> np.ndarray:
    """
    Best rank-k approximation using SVD (Eckart-Young theorem).
    
    Parameters
    ----------
    A : matrix to approximate
    k : target rank
    
    Returns
    -------
    A_k : best rank-k approximation minimizing ||A - A_k||
    """
    U, s, Vt = svd(A)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def power_iteration(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-10) -> Tuple[float, np.ndarray]:
    """
    Power iteration to find dominant eigenvalue/eigenvector.
    
    Returns
    -------
    eigenvalue : dominant eigenvalue
    eigenvector : corresponding eigenvector (normalized)
    """
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / l2_norm(v)
    
    for _ in range(max_iter):
        v_new = A @ v
        v_new = v_new / l2_norm(v_new)
        
        if l2_norm(v_new - v) < tol:
            break
        v = v_new
    
    eigenvalue = v @ A @ v
    return eigenvalue, v


# =============================================================================
# BANACH & HILBERT SPACE CONCEPTS
# =============================================================================

def is_cauchy_sequence(sequence: List[np.ndarray], tol: float = 1e-6) -> bool:
    """
    Check if a sequence of vectors is Cauchy.
    A sequence is Cauchy if ||xₘ - xₙ|| → 0 as m,n → ∞
    """
    n = len(sequence)
    if n < 2:
        return True
    
    # Check last few terms
    for i in range(max(0, n-10), n):
        for j in range(i+1, n):
            if l2_norm(sequence[i] - sequence[j]) > tol:
                return False
    return True


def check_parallelogram_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Verify parallelogram law (characterizes inner product spaces).
    ||x + y||² + ||x - y||² = 2(||x||² + ||y||²)
    
    Returns (LHS, RHS) - should be equal for Hilbert spaces.
    """
    lhs = l2_norm(x + y)**2 + l2_norm(x - y)**2
    rhs = 2 * (l2_norm(x)**2 + l2_norm(y)**2)
    return lhs, rhs


def check_cauchy_schwarz(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Verify Cauchy-Schwarz inequality.
    |⟨x, y⟩| ≤ ||x|| · ||y||
    
    Returns (LHS, RHS) - LHS should be ≤ RHS.
    """
    lhs = np.abs(inner_product(x, y))
    rhs = l2_norm(x) * l2_norm(y)
    return lhs, rhs


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LINEAR ALGEBRA ESSENTIALS DEMO")
    print("=" * 60)
    
    # --- Vector Norms ---
    print("\n1. Vector Norms")
    print("-" * 40)
    x = np.array([3, -4, 0, 2])
    print(f"   x = {x}")
    for name, value in norm_comparison(x).items():
        print(f"   {name}: {value:.4f}")
    
    # --- Inner Products ---
    print("\n2. Inner Products & Angles")
    print("-" * 40)
    a = np.array([1, 0, 0])
    b = np.array([1, 1, 0])
    print(f"   a = {a}, b = {b}")
    print(f"   ⟨a, b⟩ = {inner_product(a, b)}")
    print(f"   Angle: {np.degrees(angle_between_vectors(a, b)):.2f}°")
    print(f"   Orthogonal: {are_orthogonal(a, b)}")
    
    # --- Gram-Schmidt ---
    print("\n3. Gram-Schmidt Orthonormalization")
    print("-" * 40)
    v1 = np.array([1, 1, 0])
    v2 = np.array([1, 0, 1])
    v3 = np.array([0, 1, 1])
    orthonormal = gram_schmidt([v1, v2, v3])
    print(f"   Input: {[v1.tolist(), v2.tolist(), v3.tolist()]}")
    print(f"   Orthonormal basis:")
    for i, u in enumerate(orthonormal):
        print(f"     u{i+1} = [{', '.join(f'{x:.4f}' for x in u)}]")
    
    # --- Matrix Properties ---
    print("\n4. Matrix Properties")
    print("-" * 40)
    A = np.array([[4, 2], [2, 3]])
    print(f"   A = \n{A}")
    print(f"   Symmetric: {is_symmetric(A)}")
    print(f"   Positive definite: {is_positive_definite(A)}")
    print(f"   Condition number: {condition_number(A):.4f}")
    print(f"   Spectral radius: {spectral_radius(A):.4f}")
    
    # --- SVD ---
    print("\n5. SVD Decomposition")
    print("-" * 40)
    B = np.array([[1, 2], [3, 4], [5, 6]])
    U, s, Vt = svd_decomposition(B)
    print(f"   B (3x2) = \n{B}")
    print(f"   Singular values: {s}")
    print(f"   Rank: {len(s[s > 1e-10])}")
    
    # --- Low Rank Approximation ---
    print("\n6. Low Rank Approximation")
    print("-" * 40)
    C = np.random.randn(5, 5)
    C_rank2 = low_rank_approximation(C, k=2)
    error = frobenius_norm(C - C_rank2)
    print(f"   Original rank: {matrix_rank(C)}")
    print(f"   Approximation rank: {matrix_rank(C_rank2)}")
    print(f"   Frobenius error: {error:.4f}")
    
    # --- Hilbert Space Properties ---
    print("\n7. Hilbert Space Properties")
    print("-" * 40)
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    
    lhs, rhs = check_parallelogram_law(x, y)
    print(f"   Parallelogram law: {lhs:.4f} = {rhs:.4f} ✓")
    
    lhs, rhs = check_cauchy_schwarz(x, y)
    print(f"   Cauchy-Schwarz: {lhs:.4f} ≤ {rhs:.4f} ✓")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
