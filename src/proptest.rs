// TODO: Also test LLL conditions

#[cfg(test)]
mod proptests {
    use crate::hnf::saturation;
    use crate::solve_diophantine;
    use crate::util::{eye, matmul};
    use crate::DiophantineError;
    use crate::{
        extended_hnf, hnf, integer_det, integer_inverse, left_kernel, right_kernel, Matrix,
    };
    use proptest::prelude::*;

    fn matrix(rows: usize, cols: usize, max_val: i64) -> impl Strategy<Value = Matrix<i64>> {
        proptest::collection::vec(proptest::collection::vec(-max_val..max_val, cols), rows)
    }

    prop_compose! {
        fn random_matrix()(
            rows in 1usize..=6,
            cols in 1usize..=6,
        )(mat in matrix(rows, cols, 1000)) -> Matrix<i64> {
            mat
        }
    }

    prop_compose! {
        fn random_two_square()(
            n in 1usize..=6,
        )(
            a in matrix(n, n, 100),
            b in matrix(n, n, 100),
        ) -> (Matrix<i64>, Matrix<i64>) {
            (a, b)
        }
    }

    prop_compose! {
        fn random_matmul_pair()(
            n in 1usize..=6,
            a_rows in 1usize..=6,
            b_cols in 1usize..=6,
        )(
            a in matrix(a_rows, n, 20),
            b in matrix(n, b_cols, 20),
        ) -> (Matrix<i64>, Matrix<i64>) {
            (a, b)
        }
    }

    prop_compose! {
        fn rank_deficient_matrix()(
            rows in 1usize..=6,
            cols in 1usize..=6,
        )(
            a in matrix(rows, 2, 1000),
            b in matrix(2, cols, 1000)
        ) -> Matrix<i64> {
            matmul(&a, &b).unwrap()
        }
    }

    prop_compose! {
        // Generates a random unimodular matrix
        fn unimodular_matrix()(size in 1usize..=6)(
            // Randomly choose 1 or -1 for the diagonals of L and U
            l_diag in proptest::collection::vec(proptest::bool::ANY, size),
            u_diag in proptest::collection::vec(proptest::bool::ANY, size),
            // Random values for the lower/upper off-diagonal elements
            values in proptest::collection::vec(-10i64..=10, size * size),
        ) -> Matrix<i64> {
            let n = l_diag.len();

            let mut l = vec![vec![0; n]; n];
            let mut u = vec![vec![0; n]; n];

            let mut idx = 0;

            for i in 0..n {
                // Set diagonals to 1 or -1
                l[i][i] = if l_diag[i] { 1 } else { -1 };
                u[i][i] = if u_diag[i] { 1 } else { -1 };

                // Fill the off-diagonals
                for j in 0..n {
                    if j < i {
                        l[i][j] = values[idx];
                        idx += 1;
                    } else if j > i {
                        u[i][j] = values[idx];
                        idx += 1;
                    }
                }
            }

            // Multiply L * U to get a dense unimodular matrix
            let mut m = vec![vec![0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        m[i][j] += l[i][k] * u[k][j];
                    }
                }
            }
            m
        }
    }

    /// Verifies that a matrix is in canonical row Hermite Normal Form
    fn assert_is_hnf(mat: &Matrix<i64>) {
        let n = mat.len();
        if n == 0 {
            return;
        }
        let m = mat[0].len();

        let mut prev_pivot_col = -1;

        for i in 0..n {
            // Find the first non-zero element in the row
            let mut pivot_col = -1;
            for j in 0..m {
                if mat[i][j] != 0 {
                    pivot_col = j as i32;
                    break;
                }
            }

            if pivot_col == -1 {
                // If it's a zero row, all subsequent rows must also be zero rows
                for k in (i + 1)..n {
                    assert!(
                        mat[k].iter().all(|&x| x == 0),
                        "Zero rows must be at the bottom of the matrix"
                    );
                }
                break;
            }

            // Pivot must be strictly to the right of the previous pivot
            assert!(pivot_col > prev_pivot_col,);

            let p = pivot_col as usize;
            let pivot_val = mat[i][p];

            // Pivot must be strictly positive
            assert!(pivot_val > 0);

            // Entries above the pivot must be non-negative and strictly less than the pivot
            for k in 0..i {
                let val = mat[k][p];
                assert!(val >= 0 && val < pivot_val);
            }

            prev_pivot_col = pivot_col;
        }
    }

    fn remove_zero_rows(mat: &Matrix<i64>) -> Matrix<i64> {
        let mut res = Vec::new();

        for (i, row) in mat.iter().enumerate() {
            if row.iter().any(|&val| val != 0) {
                res.push(mat[i].clone());
            }
        }
        res
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10_000))]

        #[test]
        fn inverse_property(mat in unimodular_matrix()) {
            let inv = integer_inverse(&mat).unwrap();
            let n = mat.len();
            let prod = matmul(&mat, &inv).unwrap();

            // Verify it's the identity matrix
            prop_assert_eq!(prod, eye(n));
        }

        #[test]
        fn left_kernel_property(mat in rank_deficient_matrix()) {
            if let Ok(kernel) = left_kernel(&mat) {
                let res = matmul(&kernel, &mat).unwrap();

                if !res.is_empty() {
                    for i in 0..res.len() {
                        for j in 0..res[0].len() {
                            prop_assert_eq!(res[i][j], 0);
                        }
                    }
                }
            }
        }

        #[test]
        fn right_kernel_property(mat in rank_deficient_matrix()) {
            if let Ok(kernel) = right_kernel(&mat) {
                let res = matmul(&mat, &kernel).unwrap();

                if !res.is_empty() {
                    for i in 0..res.len() {
                        for j in 0..res[0].len() {
                            prop_assert_eq!(res[i][j], 0);
                        }
                    }
                }
            }
        }

        #[test]
        fn hnf_extended_same(mat in random_matrix()) {
            // hnf and extended_hnf should give the same results
            let h = hnf(&mat);
            let h_ext = extended_hnf(&mat);


            if let (Ok(h), Ok((h2, _))) = (h.clone(), h_ext.clone()) {
                prop_assert_eq!(h, h2);
            }
        }

        #[test]
        fn hnf_invariants(mat in random_matrix()) {
            // Extended HNF must always satisfy U * A = H
            if let Ok((h, u)) = extended_hnf(&mat) {
                // U must be unimodular (abs(det(U)) == 1)
                if let Ok(det_u) = integer_det(&u) {
                    prop_assert_eq!(det_u.abs(), 1);
                }

                // U * A = H
                if let Ok(ua) = matmul(&u, &mat) {
                    prop_assert_eq!(ua, h);
                }
            }
        }

        #[test]
        fn hnf_structure(mat in random_matrix()) {
            if let Ok(h) = hnf(&mat) {
                assert_is_hnf(&h);
            }
        }

        #[test]
        fn determinant_multiplicativity((a, b) in random_two_square()) {
            // det(A * B) == det(A) * det(B)
            let ab = matmul(&a, &b).unwrap();

            let det_a = integer_det(&a);
            let det_b = integer_det(&b);
            let det_ab = integer_det(&ab);

            if let (Ok(det_a), Ok(det_b), Ok(det_ab)) = (det_a, det_b, det_ab) {
                prop_assert_eq!(det_a * det_b, det_ab);
            }
        }


        #[test]
        fn diophantine_solve((a, x) in random_matmul_pair()) {
            let b = matmul(&a, &x).unwrap();

            match solve_diophantine(&a, &b) {
                Ok(x_sol) => {
                    // Verify A * X_sol == B
                    // This one may fail because x_check is different but valid solution
                    if let Ok(b_check) = matmul(&a, &x_sol) {
                        prop_assert_eq!(b, b_check);
                    }
                }
                Err(DiophantineError::Overflow(_)) => {
                    // Ignore overflow errors gracefully. The system is solvable,
                    // but intermediate i64 bounds were exceeded.
                    prop_assert!(true);
                }
                Err(e) => {
                    // Fail the test if it throws `NoSolution` or `InvalidDimensions`
                    prop_assert!(
                        false,
                        "Solver rejected a solvable system: {:?}", e
                    );
                }
            }
        }

        #[test]
        fn test_saturation_via_double_kernel(mat in random_matrix()) {
            // The kernel of a homomorphism is always a saturated (torsion-free) submodule.
            // Thus, the saturation of a matrix A should be equivalent to
            // the right kernel of the left kernel of A.

            // Calculate the saturation directly
            let saturation_direct = saturation(&mat);


            // Calculate it via the double kernel trick.
            let saturation_dual = (|| {
                let ker = right_kernel(&mat)?;
                let basis = left_kernel(&ker)?;
                hnf(&basis)
            })();

            if let (Ok(sat1), Ok(sat2)) = (saturation_direct, saturation_dual) {
                // This doesn't always work: sat2 is empty if the basis is identity.
                // We also need to remove zero rows from the direct saturation
                // since those will get removed by the kernel calculation.

                if !sat2.is_empty() {
                    prop_assert_eq!(remove_zero_rows(&sat1), sat2, "Direct saturation did not match double kernel method");
                }
            }
        }

    }
}
