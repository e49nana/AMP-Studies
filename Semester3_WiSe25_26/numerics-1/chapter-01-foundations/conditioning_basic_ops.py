#!/usr/bin/env python3
"""
Program 2: Conditioning of basic operations
- Asks for x1, x2
- Computes relative conditioning (∞-norm style) for:
  * addition: f(x1, x2) = x1 + x2
  * multiplication: f(x1, x2) = x1 * x2
  * division: f(x1, x2) = x1 / x2

For addition:
  cond_rel,∞ = (|x1| + |x2|) / |x1 + x2|
For multiplication:
  cond_rel,∞ = 2  (from the script)
For division:
  computed via the error amplification factors φ1, φ2
"""

import math


def cond_add(x1: float, x2: float) -> float:
    """Relative condition number (∞-style) of addition."""
    f = x1 + x2
    if f == 0.0:
        # Catastrophic cancellation: extremely ill-conditioned
        return math.inf
    return (abs(x1) + abs(x2)) / abs(f)


def cond_mult(x1: float, x2: float) -> float:
    """Relative condition number (∞-style) of multiplication."""
    # As derived in the notes: cond_rel,∞ = 2
    return 2.0


def cond_div(x1: float, x2: float) -> float:
    """Relative condition number (∞-style) of division f(x1, x2) = x1 / x2."""
    if x2 == 0.0:
        return math.inf
    f = x1 / x2
    # partial derivatives:
    d1 = 1.0 / x2
    d2 = -x1 / (x2 ** 2)
    # error amplification factors φ_i = (∂f/∂x_i) * (x_i / f(x))
    phi1 = d1 * (x1 / f)
    phi2 = d2 * (x2 / f)
    return abs(phi1) + abs(phi2)


def main() -> None:
    print("=== Program 2: Conditioning of basic operations ===")

    try:
        x1 = float(input("x1 = "))
        x2 = float(input("x2 = "))
    except ValueError:
        print("Invalid input: please enter only numbers.")
        return

    print("\n--- Addition f(x1, x2) = x1 + x2 ---")
    k_add = cond_add(x1, x2)
    print(f"cond_rel,∞(addition) = {k_add}")

    print("\n--- Multiplication f(x1, x2) = x1 * x2 ---")
    k_mult = cond_mult(x1, x2)
    print(f"cond_rel,∞(multiplication) = {k_mult}")

    print("\n--- Division f(x1, x2) = x1 / x2 ---")
    if x2 == 0.0:
        print("Division is undefined (x2 = 0).")
    else:
        k_div = cond_div(x1, x2)
        print(f"cond_rel,∞(division) = {k_div}")

    print("\nNote: if x1 ≈ -x2, addition becomes very ill-conditioned")
    print("      because of cancellation (Auslöschung).")


if __name__ == "__main__":
    main()
