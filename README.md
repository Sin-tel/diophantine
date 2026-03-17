# Diophantine
Tools for calculations over integer lattices and solving systems of linear Diophantine equations.

This crate provides exact algorithms for integer linear algebra. It is designed
for applications in algebra, number theory, and lattice problems. It is a pure
Rust implementation and does not depend on external libraries such as FLINT.

## Features
* Solving Diophantine equations: Find exact integer matrix solutions to AX = B.
* Exact determinants & inverses.
* Hermite Normal Form (HNF): Compute the HNF and extended HNF of integer matrices.
* Lattice reduction: Lenstra–Lenstra–Lovász (LLL) basis reduction.
* Closest Vector Problem (CVP): Babai's Nearest Plane algorithm for approximate CVP.
* Nullspaces: Compute exact left and right integer kernels of matrices.

## Example: Solving a system of linear Diophantine equations
```
use diophantine::{solve_diophantine, Matrix};

// 2x + 3y = 8
// 4x +  y = 6
let a: Matrix<i64> = vec![vec![2, 3], vec![4, 1]];
let b: Matrix<i64> = vec![vec![8], vec![6]];

// Find the exact integer solution
let x = solve_diophantine(&a, &b).unwrap();
assert_eq!(x, vec![vec![1], vec![2]]); // x = 1, y = 2
```

## Limitations & Safety
This crate is **not** intended for cryptographic applications.

We use `i64` internally for all calculations. Lattice algorithms (particularly HNF and
Bareiss) are prone to explosive intermediate coefficient growth. Therefore, this crate
should only be used for relatively small numbers and dimensions (e.g. matrices smaller
than 10x10 with small initial values).

All internal operations use checked arithmetic. Any intermediate overflow will be safely
caught and returned as a `DiophantineError::Overflow`, ensuring no silent incorrect
results are produced in release mode. Extensive property testing is used to guarantee
the algorithms are correct and free of panics.
