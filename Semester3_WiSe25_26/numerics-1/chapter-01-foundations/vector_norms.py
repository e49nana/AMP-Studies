#!/usr/bin/env python3
"""
Program 1: Vector norms (‖·‖₁, ‖·‖₂, ‖·‖∞)
- Reads a vector from user input
- Computes 1-, 2- and infinity-norm
- Checks the inequalities: ‖x‖∞ ≤ ‖x‖₁ ≤ n‖x‖∞
"""

import math
from typing import List


def norm_1(x: List[float]) -> float:
    """Return the 1-norm of vector x."""
    return sum(abs(xi) for xi in x)


def norm_2(x: List[float]) -> float:
    """Return the 2-norm (Euclidean norm) of vector x."""
    return math.sqrt(sum(abs(xi) ** 2 for xi in x))


def norm_inf(x: List[float]) -> float:
    """Return the infinity norm of vector x."""
    return max(abs(xi) for xi in x)


def main() -> None:
    print("=== Program 1: Vector norms ===")
    print("Enter the components of the vector separated by spaces (e.g. 1 2 -3):")
    line = input("> ").strip()

    if not line:
        print("No input given. Exiting.")
        return

    try:
        x = [float(s) for s in line.split()]
    except ValueError:
        print("Invalid input: please enter only numbers.")
        return

    n = len(x)
    n1 = norm_1(x)
    n2 = norm_2(x)
    ninf = norm_inf(x)

    print(f"\nVector x = {x}")
    print(f"‖x‖₁  = {n1:.6g}")
    print(f"‖x‖₂  = {n2:.6g}")
    print(f"‖x‖∞  = {ninf:.6g}")

    print("\nChecking the inequalities from the script (example 1.6):")
    print(f"‖x‖∞ ≤ ‖x‖₁ ?     {ninf <= n1}")
    print(f"‖x‖₁ ≤ n·‖x‖∞ ?   {n1 <= n * ninf}")
    print(f"n = {n},   n·‖x‖∞ = {n * ninf:.6g}")


if __name__ == "__main__":
    main()
