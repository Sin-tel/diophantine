//! Lattice basis reduction, CVP, SVP.
//!
//! Uses `f64` for calculations, so the results are not exact.

use crate::{DiophantineError, Matrix};
use std::ops::SubAssign;

/// Subtracts `scale * other` from `target` in place.
fn vec_sub_assign<T>(target: &mut [T], other: &[T], scale: T)
where
    T: Copy + SubAssign + std::ops::Mul<Output = T>,
{
    for (t, &o) in target.iter_mut().zip(other.iter()) {
        *t -= scale * o;
    }
}

fn inner_prod(x: &[f64], y: &[f64], w: &Matrix<f64>) -> f64 {
    let n = w.len();
    if n == 0 {
        return 0.0;
    }
    let m = w[0].len();
    assert_eq!(x.len(), n);
    assert_eq!(y.len(), m);

    // temp = W * y
    let mut temp = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..m {
            sum += w[i][j] * y[j];
        }
        temp[i] = sum;
    }

    // result = x dot temp
    let mut res = 0.0;
    for i in 0..n {
        res += x[i] * temp[i];
    }
    res
}

/// Gram-Schmidt Orthogonalization
fn gramschmidt(v: &Matrix<i64>, w: &Matrix<f64>) -> Matrix<f64> {
    let nrows = v.len();
    if nrows == 0 {
        return vec![];
    }

    // Convert input integer basis to f64
    let v_f64: Matrix<f64> = v
        .iter()
        .map(|row| row.iter().map(|&x| x as f64).collect())
        .collect();

    let mut u = v_f64.clone();

    for i in 1..nrows {
        // Clone current row to work on it, then assign back
        let mut ui = u[i].clone();
        let v_i = &v_f64[i];

        for j in 0..i {
            let uj = &u[j];

            let num = inner_prod(uj, v_i, w);
            let den = inner_prod(uj, uj, w);

            // Handle zero-norm vectors if necessary (though rare in basis)
            let proj_coeff = if den.abs() < 1e-9 { 0.0 } else { num / den };

            vec_sub_assign(&mut ui, uj, proj_coeff);
        }
        u[i] = ui;
    }
    u
}

// Helper for LLL: Calculate mu coefficient
fn mu(basis: &Matrix<i64>, ortho: &Matrix<f64>, w: &Matrix<f64>, i: usize, j: usize) -> f64 {
    let a = &ortho[j];
    // Convert basis row to f64 on the fly for calculation
    let b: Vec<f64> = basis[i].iter().map(|&x| x as f64).collect();

    let num = inner_prod(a, &b, w);
    let den = inner_prod(a, a, w);

    if den.abs() < 1e-9 { 0.0 } else { num / den }
}

/// Compute the LLL reduction of a basis.
///
/// Returns an error if dimensions do not match.
///
/// # Arguments
/// * `basis` - The lattice basis (row vectors).
/// * `delta` - The reduction parameter (typically 0.75 or 0.99).
/// * `w` - The quadratic form matrix (weights). Pass Identity matrix for standard Euclidean.
pub fn lll(
    basis: &Matrix<i64>,
    delta: f64,
    w: &Matrix<f64>,
) -> Result<Matrix<i64>, DiophantineError> {
    let mut basis = basis.clone();

    let n = basis.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let m = basis[0].len();

    if w.len() != m {
        return Err(DiophantineError::InvalidDimensions(
            "Basis should have same number of columns as w".to_string(),
        ));
    }

    if w.len() != w[0].len() {
        return Err(DiophantineError::InvalidDimensions(
            "W must be square".to_string(),
        ));
    }

    let mut ortho = gramschmidt(&basis, w);
    let mut k = 1;

    while k < n {
        // Size reduction step
        for j in (0..k).rev() {
            let mu_kj = mu(&basis, &ortho, w, k, j);
            if mu_kj.abs() > 0.5 {
                let mu_int = mu_kj.round_ties_even() as i64;

                // basis[k] -= mu_int * basis[j]
                let basis_j = basis[j].clone();
                vec_sub_assign(&mut basis[k], &basis_j, mu_int);

                // Update GS
                ortho = gramschmidt(&basis, w);
            }
        }

        // LLL condition check
        let mu_k_k1 = mu(&basis, &ortho, w, k, k - 1);
        let norm_ortho_k1 = inner_prod(&ortho[k - 1], &ortho[k - 1], w);
        let norm_ortho_k = inner_prod(&ortho[k], &ortho[k], w);

        let l_condition = (delta - mu_k_k1.powi(2)) * norm_ortho_k1;

        if norm_ortho_k >= l_condition {
            k += 1;
        } else {
            // Swap rows k and k-1
            basis.swap(k, k - 1);

            // Recompute GS
            ortho = gramschmidt(&basis, w);

            k = k.saturating_sub(1).max(1);
        }
    }

    Ok(basis)
}

/// Babai's Nearest Plane Algorithm for approximate CVP.
///
/// Returns an error if dimensions do not match.
/// The basis should be LLL-reduced first.
///
/// # Arguments
/// * `v`     - The query vector.
/// * `basis` - The lattice basis (row vectors).
/// * `w` - The quadratic form matrix (weights). Pass Identity matrix for standard Euclidean.
pub fn nearest_plane(
    v: &[i64],
    basis: &Matrix<i64>,
    w: &Matrix<f64>,
) -> Result<Vec<i64>, DiophantineError> {
    let mut b = v.to_vec();

    let n = basis.len(); // number of rows
    if n == 0 {
        return Ok(vec![]);
    }
    let m = basis[0].len(); // number of cols

    if v.len() != m {
        return Err(DiophantineError::InvalidDimensions(
            "Target vector should have same length as basis".to_string(),
        ));
    }
    if w.len() != m {
        return Err(DiophantineError::InvalidDimensions(
            "Basis should have same number of columns as w".to_string(),
        ));
    }
    if w.len() != w[0].len() {
        return Err(DiophantineError::InvalidDimensions(
            "W must be square".to_string(),
        ));
    }

    let ortho = gramschmidt(basis, w);

    for j in (0..n).rev() {
        let a = &ortho[j];
        let b_f64: Vec<f64> = b.iter().map(|&x| x as f64).collect();

        let num = inner_prod(a, &b_f64, w);
        let den = inner_prod(a, a, w);
        let mu = if den.abs() < 1e-9 { 0.0 } else { num / den };

        let mu_int = mu.round_ties_even() as i64;

        // b -= mu_int * basis[j]
        let basis_j = &basis[j];
        vec_sub_assign(&mut b, basis_j, mu_int);
    }

    let mut result = Vec::with_capacity(v.len());
    for (orig_val, residue_val) in v.iter().zip(b.iter()) {
        result.push(orig_val - residue_val);
    }
    Ok(result)
}

/// Slack on an enumeration radius, so that points exactly on the bound survive rounding.
fn with_tolerance(radius_sq: f64) -> f64 {
    radius_sq * (1.0 + 1e-9) + 1e-9
}

/// How far a squared Gram-Schmidt norm may fall below the squared norm of its own row before
/// the row counts as linearly dependent on the earlier ones. Gram-Schmidt here is classical
/// and in `f64`, so the computed norm of a dependent row is noise of relative size around
/// `1e-16` times the conditioning; this leaves several orders of magnitude of headroom.
const DEGENERATE_REL_EPS: f64 = 1e-12;

/// How large the squared distance from a target to the span of a basis may be, relative to
/// the squared norm of the target, and still count as the target lying in that span.
const PROJECTION_EPS: f64 = 1e-12;

/// How large the squared distance from a target to the span may grow before squared
/// distances measured from the target stop being a usable way to rank lattice points.
/// Integer coordinates and an integer form make those distances whole numbers, and an `f64`
/// spaces whole numbers exactly one apart up to `2^52`, so below this they are still exact
/// and ties between equidistant points are still ties.
const EXACT_SCORE_LIMIT: f64 = (1u64 << 52) as f64;

/// Greatest common divisor, non-negative, with `gcd(0, x) = |x|`.
fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.unsigned_abs(), b.unsigned_abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    // Only `gcd(i64::MIN, 0)` lands on an absolute value an i64 cannot hold; a gcd of
    // `i64::MIN` with anything else divides that other value and so fits. Capping it there
    // understates the step, which only ever understates how far a coordinate has to miss by.
    a.min(i64::MAX as u64) as i64
}

/// Compute d^T W d, without allocating.
fn quad_form(d: &[f64], w: &Matrix<f64>) -> f64 {
    let mut res = 0.0;
    for i in 0..d.len() {
        let mut row = 0.0;
        for j in 0..d.len() {
            row += w[i][j] * d[j];
        }
        res += d[i] * row;
    }
    res
}

/// Checks that `v`, the rows of `basis` and `w` agree on the ambient dimension.
fn check_dims(v: &[i64], basis: &Matrix<i64>, w: &Matrix<f64>) -> Result<(), DiophantineError> {
    let m = v.len();
    if basis.iter().any(|row| row.len() != m) {
        return Err(DiophantineError::InvalidDimensions(
            "Target vector should have same length as basis columns".to_string(),
        ));
    }
    if w.len() != m || w.iter().any(|row| row.len() != m) {
        return Err(DiophantineError::InvalidDimensions(
            "W must be square and match basis columns".to_string(),
        ));
    }
    Ok(())
}

/// Schnorr-Euchner enumeration.
///
/// Distances here are measured from `p`, the projection of `v` onto the span of `basis`,
/// rather than from `v` itself. The two differ by a fixed vector that no choice of
/// coefficients can touch, and holding it out keeps every quantity the search compares at
/// the scale of the distances that actually distinguish lattice points. A target far off the
/// span would otherwise swamp them: once `|v - p|^2_w` passes the `f64` mantissa, the
/// relative slack in `with_tolerance` opens the search out over a huge radius, and the sums
/// being compared lose the low bits that tell candidates apart.
///
/// Visits the vectors `x` of the lattice with `|p - x|^2_w <= radius_sq` (up to a small
/// tolerance), in Schnorr-Euchner order: near first, but not sorted. `visit` gets the
/// coefficients of `x`, `x` itself, `|p - x|^2_w` as computed during enumeration, and `p`,
/// and returns the squared radius to continue with, measured the same way: the same to go
/// on, smaller to tighten, negative to stop.
///
/// Stops early once `max_nodes` nodes of the search tree have been visited, except that the
/// first lattice point is always reached, so `visit` is called at least once whenever the
/// lattice is non-empty. Returns whether the search ran to completion: `false` means the
/// budget ran out and points outside the visited part may have been missed.
///
/// `w` must be definite on the span of `basis`, and the rows of `basis` must be linearly
/// independent; both are checked. The basis should be reduced for speed.
pub(crate) fn enumerate<F>(
    v: &[i64],
    basis: &Matrix<i64>,
    w: &Matrix<f64>,
    radius_sq: f64,
    max_nodes: Option<u64>,
    visit: F,
) -> Result<bool, DiophantineError>
where
    F: FnMut(&[i64], &[i64], f64, &[f64]) -> Result<f64, DiophantineError>,
{
    enumerate_inner::<false, _, _>(v, basis, w, radius_sq, max_nodes, |_, _, _| false, visit)
}

/// [`enumerate`], with a second chance to rule out a branch.
///
/// Pruning on the radius alone asks whether a branch can hold a point close enough under
/// `w`. A caller ranking by something other than `w` gets a weaker question than the one it
/// cares about, and pays for the gap in nodes. `filter` is asked the caller's own question
/// at every node, before the branch below it is walked: it gets the level whose coefficients
/// are now fixed, the part of the target no remaining coefficient can change, and that
/// part's squared `w`-norm, and returns whether nothing below can be good enough.
///
/// Level `k` means the coefficients of rows `k..` are settled and rows `..k` are still free,
/// so the residual is what is left of `v` after taking off the settled rows and projecting
/// away the span of the free ones. At a leaf it is the whole difference `v - x`. Returning
/// `true` drops the branch and the search moves on to the next coefficient at that level; it
/// never ends the level, which stays the radius' job, so `filter` need not be monotone.
pub(crate) fn enumerate_filtered<F, P>(
    v: &[i64],
    basis: &Matrix<i64>,
    w: &Matrix<f64>,
    radius_sq: f64,
    max_nodes: Option<u64>,
    filter: P,
    visit: F,
) -> Result<bool, DiophantineError>
where
    F: FnMut(&[i64], &[i64], f64, &[f64]) -> Result<f64, DiophantineError>,
    P: FnMut(usize, &[f64], f64) -> bool,
{
    enumerate_inner::<true, _, _>(v, basis, w, radius_sq, max_nodes, filter, visit)
}

/// The body of [`enumerate`] and [`enumerate_filtered`]. `FILTER` says whether the residual
/// each node is judged on gets maintained at all, which costs a vector update per node.
fn enumerate_inner<const FILTER: bool, F, P>(
    v: &[i64],
    basis: &Matrix<i64>,
    w: &Matrix<f64>,
    radius_sq: f64,
    max_nodes: Option<u64>,
    mut filter: P,
    mut visit: F,
) -> Result<bool, DiophantineError>
where
    F: FnMut(&[i64], &[i64], f64, &[f64]) -> Result<f64, DiophantineError>,
    P: FnMut(usize, &[f64], f64) -> bool,
{
    check_dims(v, basis, w)?;
    if radius_sq.is_nan() || radius_sq < 0.0 {
        return Ok(true);
    }

    let n = basis.len();
    let m = v.len();
    let v_f64: Vec<f64> = v.iter().map(|&x| x as f64).collect();

    if n == 0 {
        // The lattice is just the origin, and so is the span it is measured in
        visit(&[], &vec![0; m], 0.0, &vec![0.0; m])?;
        return Ok(true);
    }

    // Cache floating point representation of the basis to avoid inner loop allocations
    let mut basis_f64 = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            basis_f64[i][j] = basis[i][j] as f64;
        }
    }

    // Compute Gram-Schmidt orthogonalization
    let ortho = gramschmidt(basis, w);

    // A Gram-Schmidt norm is what makes the distance grow as the search descends through its
    // level, so a vanishing one lets the search wander that level for free. That happens
    // exactly when the row is (numerically) in the span of the earlier ones, or when `w` is
    // not definite on the span, both of which are preconditions rather than something to
    // paper over: reject them instead of flooring the norm and enumerating forever.
    let mut b_star_norms = vec![0.0; n];
    for i in 0..n {
        let norm = inner_prod(&ortho[i], &ortho[i], w);
        let row_norm = inner_prod(&basis_f64[i], &basis_f64[i], w);
        // Asked this way round so that a norm that does not compare at all fails too
        let floor = DEGENERATE_REL_EPS * row_norm;
        if !matches!(norm.partial_cmp(&floor), Some(std::cmp::Ordering::Greater)) {
            return Err(DiophantineError::InvalidArgument(
                "Basis rows must be linearly independent and w definite on their span".to_string(),
            ));
        }
        b_star_norms[i] = norm;
    }

    // Compute GS mu coefficients
    let mut mu_mat = vec![vec![0.0; n]; n];
    for i in 0..n {
        mu_mat[i][i] = 1.0;
        for j in 0..i {
            let num = inner_prod(&basis_f64[i], &ortho[j], w);
            mu_mat[i][j] = num / b_star_norms[j];
        }
    }

    // Project target vector into the GS basis (theta)
    let mut theta = vec![0.0; n];
    for j in 0..n {
        let num = inner_prod(&v_f64, &ortho[j], w);
        theta[j] = num / b_star_norms[j];
    }

    // The projection of v onto the span, which is what the search measures distances from.
    let mut outside = v_f64.clone();
    for j in 0..n {
        vec_sub_assign(&mut outside, &ortho[j], theta[j]);
    }
    // Gram-Schmidt ran in f64, so a target that lies in the span still comes back with an
    // out-of-span part, at the noise level of that pass. Reading it as real would leave the
    // projection a hair off the integer point it should be, and distances computed from it a
    // hair off each other, which is enough to break ties between equidistant lattice points
    // differently for different bases. Below that level, take the target to be in the span.
    let v_norm_sq = quad_form(&v_f64, w);
    let mut outside_dist = quad_form(&outside, w).max(0.0);
    let projection: Vec<f64> = if outside_dist <= PROJECTION_EPS * v_norm_sq {
        outside.iter_mut().for_each(|o| *o = 0.0);
        outside_dist = 0.0;
        v_f64.clone()
    } else {
        v_f64.iter().zip(&outside).map(|(a, b)| a - b).collect()
    };

    let mut bound = with_tolerance(radius_sq);

    let mut x = vec![0i64; n];
    let mut c = vec![0.0; n];
    let mut p = vec![0.0; n + 1];
    let mut d = vec![0i64; n];
    let mut step = vec![0i64; n];

    // partial[k] = sum of x[i] * basis[i] for i >= k, filled in on the way down
    let mut partial = vec![vec![0i64; m]; n + 1];
    let mut point = vec![0i64; m];

    // residual[k] = the part of v - x that the coefficients below level k cannot change,
    // which is the out-of-span part plus what levels k.. have already committed to. Its
    // squared w-norm is p[k] + outside_dist, so it is the vector behind the radius test, and
    // it is what `filter` judges a branch on. Only maintained when there is a filter to use
    // it: it costs a vector update at every node, where the radius test costs nothing.
    let residual_depth = if FILTER { n + 1 } else { 0 };
    let mut residual = vec![vec![0.0; m]; residual_depth];
    if FILTER {
        residual[n].copy_from_slice(&outside);
    }

    // Initialize the root node at level n - 1
    let mut k = n - 1;
    c[k] = theta[k];
    x[k] = c[k].round_ties_even() as i64;
    let y = c[k] - x[k] as f64;
    step[k] = if y >= 0.0 { 1 } else { -1 };
    d[k] = 1;
    p[n] = 0.0;
    p[k] = p[k + 1] + y * y * b_star_norms[k];
    if FILTER {
        let (below, above) = residual.split_at_mut(k + 1);
        for j in 0..m {
            below[k][j] = above[0][j] + y * ortho[k][j];
        }
    }

    // Nodes spent so far, and whether the first lattice point is in hand. The budget only
    // applies from then on, so that a caller that asks for very little still gets the
    // nearest plane point rather than nothing at all.
    let mut nodes: u64 = 0;
    let mut reached_leaf = false;

    // Depth-first search
    loop {
        if reached_leaf && max_nodes.is_some_and(|max| nodes >= max) {
            return Ok(false);
        }
        nodes += 1;

        // Asking the radius first keeps the cheaper test in front of the more expensive one
        if p[k] <= bound && !(FILTER && filter(k, &residual[k], p[k] + outside_dist)) {
            if k == 0 {
                // Reached a leaf node (a complete lattice point)
                for j in 0..m {
                    point[j] = x[0]
                        .checked_mul(basis[0][j])
                        .and_then(|t| t.checked_add(partial[1][j]))
                        .ok_or(DiophantineError::Overflow("enumerate: lattice vector"))?;
                }
                reached_leaf = true;
                let r = visit(&x, &point, p[0], &projection)?;
                if r.is_nan() || r < 0.0 {
                    return Ok(true);
                }
                bound = with_tolerance(r);
            } else {
                // Internal node: step down to level k - 1
                for j in 0..m {
                    partial[k][j] = x[k]
                        .checked_mul(basis[k][j])
                        .and_then(|t| t.checked_add(partial[k + 1][j]))
                        .ok_or(DiophantineError::Overflow("enumerate: lattice vector"))?;
                }

                k -= 1;
                let mut sum = 0.0;
                for i in (k + 1)..n {
                    sum += x[i] as f64 * mu_mat[i][k];
                }
                c[k] = theta[k] - sum;
                x[k] = c[k].round_ties_even() as i64;

                let y = c[k] - x[k] as f64;
                step[k] = if y >= 0.0 { 1 } else { -1 };
                d[k] = 1;
                p[k] = p[k + 1] + y * y * b_star_norms[k];
                if FILTER {
                    let (below, above) = residual.split_at_mut(k + 1);
                    for j in 0..m {
                        below[k][j] = above[0][j] + y * ortho[k][j];
                    }
                }
                continue;
            }
        } else if p[k] > bound {
            // Prune current branch: step back up to level k + 1
            k += 1;
            if k == n {
                return Ok(true);
            }
        }

        // Advance to the next integer coefficient at level k, in alternating SE order
        x[k] += step[k] * d[k];
        step[k] = -step[k];
        d[k] += 1;
        let y = c[k] - x[k] as f64;
        p[k] = p[k + 1] + y * y * b_star_norms[k];
        if FILTER {
            let (below, above) = residual.split_at_mut(k + 1);
            for j in 0..m {
                below[k][j] = above[0][j] + y * ortho[k][j];
            }
        }
    }
}

/// A score to rank lattice points by, carrying the squared radius that goes with it.
///
/// The two are the same distance seen from different places: `order` from the target, which
/// is what callers asked to be close to, and `radius_sq` from its projection onto the span,
/// which is what the enumeration measures and prunes against. Only `order` takes part in
/// comparisons, so equal distances stay equal and [`TopK`] breaks the tie on the vector.
#[derive(Clone, Copy)]
struct Scored {
    order: f64,
    radius_sq: f64,
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order
    }
}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.order.partial_cmp(&other.order)
    }
}

/// The `k` best vectors found so far, best first, ordered by score and then by the vector.
struct TopK<S> {
    k: usize,
    items: Vec<(S, Vec<i64>)>,
}

impl<S: PartialOrd + Copy> TopK<S> {
    fn new(k: usize) -> Self {
        TopK {
            k,
            items: Vec::with_capacity(k),
        }
    }

    fn insert(&mut self, x: &[i64], score: S) {
        // How a kept item compares to the new one
        let cmp = |(s, y): &(S, Vec<i64>)| {
            s.partial_cmp(&score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| y.as_slice().cmp(x))
        };
        if self.items.len() >= self.k && self.items.last().is_none_or(|last| cmp(last).is_le()) {
            return;
        }
        let pos = self.items.partition_point(|item| cmp(item).is_lt());
        self.items.insert(pos, (score, x.to_vec()));
        self.items.truncate(self.k);
    }

    /// The squared radius that still holds every candidate for the list, computed from the
    /// worst kept score by `radius_sq`. Infinite until the list is full.
    fn radius_sq(&self, radius_sq: impl Fn(S) -> f64) -> f64 {
        match self.items.last() {
            Some((s, _)) if self.items.len() >= self.k => radius_sq(*s),
            _ => f64::INFINITY,
        }
    }

    fn into_vecs(self) -> Matrix<i64> {
        self.items.into_iter().map(|(_, x)| x).collect()
    }
}

/// Closest Vector Problem (CVP) using Schnorr-Euchner enumeration.
///
/// Returns the closest vector in the lattice to the target vector `v`, and whether the
/// search completed. If it did, the vector is exactly the closest one, with ties broken by
/// the vector itself (lexicographically smallest); if `max_nodes` ran out first, it is the
/// closest one found so far, starting from the point Babai's nearest plane arrives at, which
/// the search always reaches. That is a point [`nearest_plane`] could return rather than
/// always the one it does return: where rounding a coefficient is an exact tie the two may
/// split it differently, and the vectors that follow are equally good under `w` but need not
/// be equally good under anything else.
///
/// For reasonable performance, `basis` MUST be highly reduced (e.g., LLL or BKZ) before calling.
///
/// # Arguments
/// * `v` - The target vector (should match the number of columns in the basis).
/// * `basis` - The lattice basis (row vectors), linearly independent.
/// * `w` - The metric quadratic form matrix (weights).
/// * `max_nodes` - Search budget, see [`cvp_top_k`]. `None` searches until done.
pub fn cvp_exact(
    v: &[i64],
    basis: &Matrix<i64>,
    w: &Matrix<f64>,
    max_nodes: Option<u64>,
) -> Result<(Vec<i64>, bool), DiophantineError> {
    let (mut vecs, complete) = cvp_top_k(v, basis, w, 1, max_nodes)?;
    Ok((vecs.swap_remove(0), complete))
}

/// The `k` vectors of the lattice closest to the target `v` under the quadratic form `w`,
/// closest first, and whether the search completed. Ties are broken by the vector itself
/// (lexicographically smallest first), so a completed search does not depend on the choice
/// of basis.
///
/// The exception is a target whose distance to the span of `basis` is past what an `f64`
/// holds exactly, around `2^52`, which needs a basis that does not span the whole space to
/// arise at all. Ranking there is done from the projection of the target onto the span, so
/// it stays correct between points at different distances but no longer separates points at
/// equal ones reliably, and which of several equally close vectors comes back may depend on
/// the basis.
///
/// For reasonable performance, `basis` should be reduced (e.g. LLL) under `w`, and `k` small.
///
/// # Search budget
/// `max_nodes` caps the nodes of the enumeration tree the search may visit, which is what
/// its running time is proportional to. `None` means no cap, and the result is then always
/// exact. Otherwise the second return value says whether the tree was exhausted within the
/// budget: `false` means the vectors are the best found so far rather than provably the
/// closest, and fewer than `k` of them may be returned. The first lattice point is always
/// reached, however small the budget, so at least one vector always comes back.
///
/// The budget is on work, not on quality, so which vectors a search that ran out returns
/// depends on the basis. For an exact answer, retry with a larger budget.
///
/// # Arguments
/// * `v` - The target vector (should match the number of columns in the basis).
/// * `basis` - The lattice basis (row vectors), linearly independent.
/// * `w` - The metric quadratic form matrix (weights), definite on the span of `basis`.
/// * `k` - The number of vectors to return.
/// * `max_nodes` - Search budget, or `None` to search until done.
pub fn cvp_top_k(
    v: &[i64],
    basis: &Matrix<i64>,
    w: &Matrix<f64>,
    k: usize,
    max_nodes: Option<u64>,
) -> Result<(Matrix<i64>, bool), DiophantineError> {
    check_dims(v, basis, w)?;
    if k == 0 {
        return Ok((vec![], true));
    }

    let mut top: TopK<Scored> = TopK::new(k);
    let mut diff = vec![0.0; v.len()];

    // Ranking by the squared distance to the target is exact, which is what makes ties
    // between equidistant points break on the vector and so come out the same whatever the
    // basis. That holds only while the distance stays inside the `f64` mantissa: a target
    // far off the span pushes it past, and the part that tells lattice points apart is the
    // first thing rounded away, leaving every candidate looking equally good. Past that,
    // rank by the distance to the projection of the target instead, which is the same order
    // measured from a point the lattice can actually reach, and so stays at the scale of the
    // differences being ranked. Decided once, on the first point, and then kept.
    let mut score_is_exact: Option<bool> = None;

    let complete = enumerate(
        v,
        basis,
        w,
        f64::INFINITY,
        max_nodes,
        |_, x, dist_sq, proj| {
            let exact = *score_is_exact.get_or_insert_with(|| {
                let outside: Vec<f64> = v
                    .iter()
                    .zip(proj)
                    .map(|(&vi, &pj)| vi as f64 - pj)
                    .collect();
                quad_form(&outside, w) < EXACT_SCORE_LIMIT
            });

            if exact {
                for j in 0..v.len() {
                    let dj = v[j]
                        .checked_sub(x[j])
                        .ok_or(DiophantineError::Overflow("cvp_top_k: difference"))?;
                    diff[j] = dj as f64;
                }
            } else {
                for j in 0..v.len() {
                    diff[j] = proj[j] - x[j] as f64;
                }
            }

            top.insert(
                x,
                Scored {
                    order: quad_form(&diff, w),
                    radius_sq: dist_sq,
                },
            );
            Ok(top.radius_sq(|s| s.radius_sq))
        },
    )?;

    Ok((top.into_vecs(), complete))
}

/// The `k` vectors `x` of the lattice for which `v - x` is smallest under the weighted
/// L1 norm `sum weights[i] * |(v - x)[i]|`, best first. Ties are broken by the weighted
/// L2 norm `sum (weights[i] * (v - x)[i])^2`, then by `x` itself (lexicographically smallest
/// first), so a completed search does not depend on the choice of basis.
///
/// Weights must be non-negative and may be zero, as long as no nonzero lattice vector has
/// zero weighted norm. For reasonable performance, `basis` should be reduced (e.g. LLL)
/// under `diag(weights^2)`, and `k` small.
///
/// Returns whether the search completed as well; see [`cvp_top_k`] for what `max_nodes`
/// does and what an incomplete search means.
///
/// # Cost
/// Pruning happens under `diag(weights^2)`, so the search walks an L2 ball wide enough to
/// hold the L1 ball it wants. No ellipsoid holds that ball more tightly, so some of the gap
/// is not removable, and this visits more nodes than [`cvp_top_k`] does at the same
/// dimension: around 20 times as many at dimension 9, growing with dimension.
///
/// One case is worse than that. A target far outside the span of `basis` is a large
/// distance that every lattice point pays alike, and squaring it to reach a radius turns a
/// small difference between points into a large one. Where the lattice cannot reach a
/// coordinate at all, that part is recognised and taken out, which is the usual shape of a
/// target with an extra coordinate the lattice does not span. Where the direction out of
/// the span is not a coordinate, it is not, and the cost still grows with how far out the
/// target sits; give such a search a `max_nodes` rather than letting it run.
///
/// # Arguments
/// * `v` - The target vector (should match the number of columns in the basis).
/// * `basis` - The lattice basis (row vectors), linearly independent.
/// * `weights` - The weight of each coordinate.
/// * `k` - The number of vectors to return.
/// * `max_nodes` - Search budget, or `None` to search until done.
pub fn cvp_l1_top_k(
    v: &[i64],
    basis: &Matrix<i64>,
    weights: &[i64],
    k: usize,
    max_nodes: Option<u64>,
) -> Result<(Matrix<i64>, bool), DiophantineError> {
    let m = v.len();
    if weights.len() != m {
        return Err(DiophantineError::InvalidDimensions(
            "Weights should have same length as target vector".to_string(),
        ));
    }
    if weights.iter().any(|&wi| wi < 0) {
        return Err(DiophantineError::InvalidArgument(
            "Weights must be non-negative".to_string(),
        ));
    }

    // The weighted L2 norm is at most the weighted L1 norm, so every candidate at least as
    // good as the current k-th best lies in the L2 ball of that radius under diag(weights^2).
    let mut w = vec![vec![0.0; m]; m];
    for i in 0..m {
        w[i][i] = (weights[i] as f64).powi(2);
    }
    check_dims(v, basis, &w)?;
    if k == 0 {
        return Ok((vec![], true));
    }

    // In coordinate `j` a lattice point can only land on a multiple of the gcd of column `j`,
    // so `floor[j]` is how far off the target it has to be there whatever the search does,
    // and no lattice point scores better than `unavoidable`. A column of zeroes means the
    // coordinate is out of reach entirely and its miss is the same for every lattice point.
    let mut floor_dist = vec![0i64; m];
    let mut reachable = vec![false; m];
    for j in 0..m {
        let step = basis.iter().fold(0i64, |g, row| gcd(g, row[j]));
        reachable[j] = step != 0;
        floor_dist[j] = if step == 0 {
            v[j].checked_abs()
                .ok_or(DiophantineError::Overflow("cvp_l1_top_k: target"))?
        } else {
            let rem = v[j].rem_euclid(step.abs());
            rem.min(step.abs() - rem)
        };
    }
    // All in the weighted norms the search works in, and in f64 because they only ever feed
    // a radius. `unavoidable` is the weighted L1 of the misses and no point scores below it,
    // `reach_floor_sq` and `stuck_floor_sq` split their weighted squared L2 by whether the
    // coordinate can be improved, and `worst_reach` is the largest weighted miss that can.
    let weighted = |j: usize| weights[j] as f64 * floor_dist[j] as f64;
    let weighted_sq = |j: usize| weighted(j).powi(2);
    let unavoidable: f64 = (0..m).map(weighted).sum();
    let reach_floor_sq: f64 = (0..m).filter(|&j| reachable[j]).map(weighted_sq).sum();
    let stuck_floor_sq: f64 = (0..m).filter(|&j| !reachable[j]).map(weighted_sq).sum();
    let worst_reach = (0..m)
        .filter(|&j| reachable[j])
        .map(weighted)
        .fold(0.0, f64::max);

    // Turns a weighted L1 score into the squared radius the enumeration prunes on, which it
    // measures inside the span. Writing each coordinate's miss as its floor plus a surplus,
    // the surpluses are non-negative, vanish where the coordinate is out of reach, and have
    // weighted L1 at most `slack`, which caps both the sum of their squares and how much
    // they can cross with the floors.
    let in_span_radius = |l1: f64, outside: f64| {
        let slack = (l1 - unavoidable).max(0.0);
        let surplus = 2.0 * worst_reach * slack + slack * slack;

        // An out-of-reach coordinate misses by the same amount for every lattice point, and
        // the span cannot lean that way at all, so that miss is already part of the distance
        // to the span: `stuck_floor_sq <= outside` holds exactly. Dropping the pair rather
        // than subtracting keeps the result at the scale of what survives, which matters
        // because both are enormous when the target is far off the span, where subtracting
        // would leave nothing but rounding. Clamping covers that rounding in the near case.
        let from_floors = reach_floor_sq + (stuck_floor_sq - outside).max(0.0) + surplus;

        // The plain `L2 <= L1` route, for when the floors say nothing. This one does have to
        // subtract, so give back the low bits it drops.
        let lost = 8.0 * f64::EPSILON * (l1 * l1 + outside);
        from_floors.min(l1 * l1 - outside + lost)
    };

    let mut top = TopK::new(k);
    // How far the target is from the span, which the enumeration does not count but the L1
    // score does. Worked out once, the first time a lattice point comes back.
    let mut outside_dist: Option<f64> = None;
    // The k-th best score so far, which is what a branch has to beat, shared with the filter
    let best_l1 = std::cell::Cell::new(f64::INFINITY);

    // A branch is judged on the part of the target its remaining coefficients cannot reach.
    // Since `w` is `diag(weights^2)`, scaling that residual by `w` gives a vector `u` with
    // `|u_j| <= weights[j]` once divided through by its largest weighted coordinate, and `u`
    // is orthogonal to everything the branch can still add. For any such `u` the weighted L1
    // score of every point below is at least `<u, residual>`, which works out as the squared
    // norm over that largest weighted coordinate. That beats reading the radius off the
    // squared norm alone by the ratio between the residual's weighted L2 and L-infinity
    // norms, up to a factor of `sqrt(m)` in radius.
    let filter = |_k: usize, residual: &[f64], norm_sq: f64| {
        let limit = best_l1.get();
        if !limit.is_finite() {
            return false;
        }
        let peak = (0..m)
            .map(|j| (weights[j] as f64 * residual[j]).abs())
            .fold(0.0, f64::max);
        peak > 0.0 && norm_sq > with_tolerance(limit * peak)
    };

    let complete = enumerate_filtered(
        v,
        basis,
        &w,
        f64::INFINITY,
        max_nodes,
        filter,
        |_, x, _, proj| {
            let mut l1: i64 = 0;
            let mut l2: i64 = 0;
            for j in 0..m {
                let wd = v[j]
                    .checked_sub(x[j])
                    .and_then(|dj| dj.checked_mul(weights[j]))
                    .ok_or(DiophantineError::Overflow("cvp_l1_top_k: norm"))?;
                l1 = wd
                    .checked_abs()
                    .and_then(|a| l1.checked_add(a))
                    .ok_or(DiophantineError::Overflow("cvp_l1_top_k: norm"))?;
                l2 = wd
                    .checked_mul(wd)
                    .and_then(|s| l2.checked_add(s))
                    .ok_or(DiophantineError::Overflow("cvp_l1_top_k: norm"))?;
            }
            top.insert(x, (l1, l2));
            best_l1.set(top.radius_sq(|(l1, _)| l1 as f64));

            // What is left of the radius once the part outside the span is taken off bounds the
            // part inside, which is what the enumeration measures.
            let outside = *outside_dist.get_or_insert_with(|| {
                let d: Vec<f64> = v
                    .iter()
                    .zip(proj)
                    .map(|(&vi, &pj)| vi as f64 - pj)
                    .collect();
                quad_form(&d, &w)
            });
            Ok(top.radius_sq(|(l1, _)| in_span_radius(l1 as f64, outside)))
        },
    )?;

    Ok((top.into_vecs(), complete))
}

/// Shortest Vector Problem (SVP) using Schnorr-Euchner enumeration.
///
/// Returns the shortest **non-zero** vector in the lattice, and whether the search
/// completed. If it did, the vector is exactly the shortest one, with ties broken by the
/// vector itself (lexicographically smallest); if `max_nodes` ran out first, it is the
/// shortest one found so far, which is never worse than the shortest row of `basis`.
///
/// For reasonable performance, `basis` MUST be highly reduced (e.g., LLL or BKZ) before calling.
///
/// # Arguments
/// * `basis` - The lattice basis (row vectors), linearly independent.
/// * `w` - The metric quadratic form matrix (weights).
/// * `max_nodes` - Search budget, see [`cvp_top_k`]. `None` searches until done.
pub fn svp_exact(
    basis: &Matrix<i64>,
    w: &Matrix<f64>,
    max_nodes: Option<u64>,
) -> Result<(Vec<i64>, bool), DiophantineError> {
    let Some(first) = basis.first() else {
        return Ok((vec![], true));
    };
    let m = first.len();
    let origin = vec![0; m];
    check_dims(&origin, basis, w)?;

    let mut top = TopK::new(1);
    let mut x_f64 = vec![0.0; m];

    // Seed with the shortest row. Unlike the CVP searches, the first lattice point the
    // enumeration reaches is the origin, which is not a candidate, so without a seed a
    // budget could run out leaving nothing to return. It also starts the search at a finite
    // radius rather than an infinite one, which can only prune more.
    for row in basis {
        for (xf, &xi) in x_f64.iter_mut().zip(row) {
            *xf = xi as f64;
        }
        top.insert(row, quad_form(&x_f64, w));
    }

    let complete = enumerate(
        &origin,
        basis,
        w,
        top.radius_sq(|s| s),
        max_nodes,
        |coeffs, x, _, _| {
            if coeffs.iter().any(|&c| c != 0) {
                for (xf, &xi) in x_f64.iter_mut().zip(x) {
                    *xf = xi as f64;
                }
                top.insert(x, quad_form(&x_f64, w));
            }
            Ok(top.radius_sq(|s| s))
        },
    )?;

    Ok((top.into_vecs().pop().unwrap_or_default(), complete))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{eye, integer_det};

    fn norm_sq(v: &[i64]) -> i64 {
        v.iter().map(|x| x * x).sum()
    }

    /// Unwraps the result of a search run without a budget, asserting that it completed.
    #[track_caller]
    fn exhaustive<T>((value, complete): (T, bool)) -> T {
        assert!(complete, "a search without a budget must always complete");
        value
    }

    #[test]
    fn lll_identity() {
        // LLL of identity is identity
        let basis = vec![vec![1, 0], vec![0, 1]];
        let w = eye(2);
        let reduced = lll(&basis, 0.99, &w).unwrap();
        assert_eq!(basis, reduced);
    }

    #[test]
    fn lll_eye() {
        // Should reduce to identity
        let basis = vec![
            vec![1, 0, 0, 0],
            vec![10, 1, 0, 0],
            vec![10, 0, 1, 0],
            vec![10, 0, 0, 1],
        ];
        let w = eye(4);

        let reduced = lll(&basis, 0.99, &w).unwrap();

        let det_in = integer_det(&basis).unwrap().abs();
        let det_out = integer_det(&reduced).unwrap().abs();
        assert_eq!(det_in, det_out, "LLL must preserve lattice determinant");

        assert!(
            reduced
                == vec![
                    vec![1, 0, 0, 0],
                    vec![0, 1, 0, 0],
                    vec![0, 0, 1, 0],
                    vec![0, 0, 0, 1],
                ]
        );
    }

    #[test]
    fn lll_det() {
        // Some random 4x4 matrix
        let basis = vec![
            vec![12, 10, 54, 46],
            vec![23, 23, -56, 23],
            vec![43, -8, 53, 20],
            vec![10, 8, -89, 1],
        ];
        let w = eye(4);

        let reduced = lll(&basis, 0.99, &w).unwrap();

        let det_in = integer_det(&basis).unwrap().abs();
        let det_out = integer_det(&reduced).unwrap().abs();

        assert_eq!(det_in, det_out, "LLL must preserve determinant");

        let max_input_norm = basis.iter().map(|r| norm_sq(r)).max().unwrap();
        let first_reduced_norm = norm_sq(&reduced[0]);

        assert!(
            first_reduced_norm < max_input_norm,
            "First vector should be reduced"
        );
    }

    #[test]
    fn phi_lll() {
        // Find an integer polynomial for the golden ratio
        // Last row:
        //   round(10_000 * phi^2)
        //   round(10_000 * phi)
        //   10_000
        //
        // First vector should be [1 -1 -1 0]
        // Since it is the root of x^2 - x - 1

        let basis = vec![
            vec![1, 0, 0, 26_180],
            vec![0, 1, 0, 16_180],
            vec![0, 0, 1, 10_000],
        ];

        let w = eye(4);
        let reduced = lll(&basis, 0.99, &w).unwrap();

        // Don't know what sign it is going to give
        assert!(reduced[0] == vec![1, -1, -1, 0] || reduced[0] == vec![-1, 1, 1, 0]);
    }

    #[test]
    fn phi_exact() {
        let basis = vec![
            vec![1, 0, 0, 26_180],
            vec![0, 1, 0, 16_180],
            vec![0, 0, 1, 10_000],
        ];

        let w = eye(4);
        let sv = exhaustive(svp_exact(&basis, &w, None).unwrap());
        assert!(sv == vec![1, -1, -1, 0] || sv == vec![-1, 1, 1, 0]);
    }

    #[test]
    fn nearest_plane_simple_grid() {
        // If the lattice is Z^2, this should always return the same vector
        let basis = vec![vec![1, 0], vec![0, 1]];
        let w = eye(2);

        let target = vec![10, 10];
        let res = nearest_plane(&target, &basis, &w).unwrap();
        assert_eq!(res, vec![10, 10]);

        let target = vec![123, 456];
        let res = nearest_plane(&target, &basis, &w).unwrap();
        assert_eq!(res, vec![123, 456]);
    }

    #[test]
    fn nearest_plane_scaled_lattice() {
        // Lattice 2*Z^2
        let basis = vec![vec![2, 0], vec![0, 2]];
        let w = eye(2);

        let target = vec![3, 3];
        let res = nearest_plane(&target, &basis, &w).unwrap();
        assert_eq!(res, vec![4, 4]);

        let target = vec![5, 1];
        let res = nearest_plane(&target, &basis, &w).unwrap();
        assert_eq!(res, vec![4, 0]);
    }

    #[test]
    fn test_lll_dims() {
        let basis = vec![vec![1, 0], vec![0, 1], vec![0, 1]];
        let w = eye(3);
        let res = lll(&basis, 0.75, &w);
        assert!(matches!(res, Err(DiophantineError::InvalidDimensions(_))));

        let basis = eye(3);
        let w = vec![vec![1., 0.], vec![0., 1.], vec![0., 1.]];
        let res = lll(&basis, 0.75, &w);
        assert!(matches!(res, Err(DiophantineError::InvalidDimensions(_))));
    }

    #[test]
    fn test_nearest_plane_dims() {
        let target = vec![1, 2, 3, 4];
        let basis = eye(3);
        let w = eye(3);
        let res = nearest_plane(&target, &basis, &w);
        assert!(matches!(res, Err(DiophantineError::InvalidDimensions(_))));

        let target = vec![1, 2, 3];
        let basis = eye(3);
        let w = eye(4);
        let res = nearest_plane(&target, &basis, &w);
        assert!(matches!(res, Err(DiophantineError::InvalidDimensions(_))));
    }

    #[test]
    fn svp_identity() {
        let basis = vec![vec![1, 0], vec![0, 1]];
        let w = eye(2);
        let sv = exhaustive(svp_exact(&basis, &w, None).unwrap());

        let norm = norm_sq(&sv);
        assert_eq!(norm, 1, "Shortest vector in Z^2 should have norm 1");
        assert!(sv == vec![1, 0] || sv == vec![0, 1] || sv == vec![-1, 0] || sv == vec![0, -1]);
    }

    #[test]
    fn svp_known_lattice() {
        let basis = vec![vec![1, 13, 14], vec![0, 12, 13]];
        let w = eye(3);

        let sv = exhaustive(svp_exact(&basis, &w, None).unwrap());

        // Should be [1, 1, 1]
        assert_eq!(norm_sq(&sv), 3);
    }

    #[test]
    fn test_cvp_exact_in_lattice() {
        // If the target is exactly a lattice point, the distance should be 0
        let basis = vec![vec![2, 0], vec![0, 2]];
        let w = eye(2);
        let target = vec![4, 6];

        let closest = exhaustive(cvp_exact(&target, &basis, &w, None).unwrap());
        assert_eq!(closest, vec![4, 6]);
    }

    #[test]
    fn test_cvp_exact_halfway() {
        let basis = vec![vec![2, 0], vec![0, 2]];
        let w = eye(2);

        // Target is directly in the middle of a 2x2 square cell [2, 0] to [4, 2]
        let target = vec![3, 1];
        let closest = exhaustive(cvp_exact(&target, &basis, &w, None).unwrap());

        // Distance from [3, 1] to any corner of its cell ([2,0], [4,0], [2,2], [4,2]) is exactly 2.
        let dist = norm_sq(&[closest[0] - target[0], closest[1] - target[1]]);
        assert_eq!(dist, 2);
    }

    #[test]
    fn cvp_top_k_ties() {
        // The four corners of the cell are equally close, and come out ordered by vector
        let basis = vec![vec![2, 0], vec![0, 2]];
        let target = vec![1, 1];
        let expected = vec![vec![0, 0], vec![0, 2], vec![2, 0], vec![2, 2]];

        let top_k = exhaustive(cvp_top_k(&target, &basis, &eye(2), 4, None).unwrap());
        assert_eq!(top_k, expected);
        let top_k_l1 = exhaustive(cvp_l1_top_k(&target, &basis, &[1, 1], 4, None).unwrap());
        assert_eq!(top_k_l1, expected);
        let closest = exhaustive(cvp_exact(&target, &basis, &eye(2), None).unwrap());
        assert_eq!(closest, vec![0, 0]);
    }

    #[test]
    fn cvp_l1_differs_from_l2() {
        // From (3, 0), the lattice point (0, 0) is off by (3, 0) and (1, -2) by (2, 2):
        // L1 distances 3 and 4, but L2 distances 9 and 8.
        let basis = vec![vec![1, -2]];
        let target = vec![3, 0];
        let res = exhaustive(cvp_l1_top_k(&target, &basis, &[1, 1], 2, None).unwrap());
        assert_eq!(res, vec![vec![0, 0], vec![1, -2]]);
        let res = exhaustive(cvp_top_k(&target, &basis, &eye(2), 2, None).unwrap());
        assert_eq!(res, vec![vec![1, -2], vec![0, 0]]);
    }

    #[test]
    fn cvp_top_k_outside_span() {
        // Lattice spanned by (1, 0, 0) and (0, 1, 0), target off the plane
        let basis = vec![vec![1, 0, 0], vec![0, 1, 0]];
        let target = vec![3, -2, 7];
        let res = exhaustive(cvp_top_k(&target, &basis, &eye(3), 5, None).unwrap());
        assert_eq!(res[0], vec![3, -2, 0]);
        assert_eq!(res.len(), 5);
        let res = exhaustive(cvp_l1_top_k(&target, &basis, &[1, 1, 1], 5, None).unwrap());
        assert_eq!(res[0], vec![3, -2, 0]);
        assert_eq!(res.len(), 5);
    }

    #[test]
    fn cvp_l1_zero_weight() {
        // Vectors (a, a + b, b): only zero has zero weight under (0, 1, 2)
        let basis = vec![vec![1, 1, 0], vec![0, 1, 1]];
        let target = vec![10, 3, 1];
        let res = exhaustive(cvp_l1_top_k(&target, &basis, &[0, 1, 2], 3, None).unwrap());
        // (2, 3, 1) is a lattice point with weighted distance 0
        assert_eq!(res[0], vec![2, 3, 1]);
        assert_eq!(res.len(), 3);
    }

    #[test]
    fn top_k_edge_cases() {
        let basis = vec![vec![1, 0], vec![0, 1]];
        let target = vec![1, 2];
        assert!(exhaustive(cvp_top_k(&target, &basis, &eye(2), 0, None).unwrap()).is_empty());
        assert!(exhaustive(cvp_l1_top_k(&target, &basis, &[1, 1], 0, None).unwrap()).is_empty());

        // Rank 0: the lattice is just the origin
        let empty: Matrix<i64> = vec![];
        assert_eq!(
            exhaustive(cvp_top_k(&target, &empty, &eye(2), 3, None).unwrap()),
            vec![vec![0, 0]]
        );
        assert_eq!(
            exhaustive(cvp_l1_top_k(&target, &empty, &[1, 1], 3, None).unwrap()),
            vec![vec![0, 0]]
        );
        let closest = exhaustive(cvp_exact(&target, &empty, &eye(2), None).unwrap());
        assert_eq!(closest, vec![0, 0]);

        // Rank 1, many more points than a small box
        let line = vec![vec![1, 1]];
        let res = exhaustive(cvp_top_k(&[0, 0], &line, &eye(2), 7, None).unwrap());
        assert_eq!(res.len(), 7);
        assert_eq!(res[0], vec![0, 0]);
        assert_eq!(res[5], vec![-3, -3]);
        assert_eq!(res[6], vec![3, 3]);

        let res = cvp_l1_top_k(&target, &basis, &[1, -1], 1, None);
        assert!(matches!(res, Err(DiophantineError::InvalidArgument(_))));
        let res = cvp_l1_top_k(&target, &basis, &[1, 1, 1], 1, None);
        assert!(matches!(res, Err(DiophantineError::InvalidDimensions(_))));
    }

    #[test]
    fn cvp_far_outside_span() {
        // The lattice spans the z = 0 plane and the target sits far off it. That distance is
        // the same for every lattice point, so it should not reach the search at all: it used
        // to scale the rounding tolerance with it, widening the search to an in-span radius of
        // thousands, and to swamp the scores the candidates were ranked by.
        let basis = vec![vec![1, 0, 0], vec![0, 1, 0]];
        let target = vec![3, -2, 100_000_000];

        let (closest, complete) = cvp_exact(&target, &basis, &eye(3), Some(10_000)).unwrap();
        assert!(
            complete,
            "an out-of-span offset should not enlarge the search"
        );
        assert_eq!(closest, vec![3, -2, 0]);

        // The point under the target first, then its four neighbours ordered by vector
        let (res, complete) = cvp_top_k(&target, &basis, &eye(3), 5, Some(10_000)).unwrap();
        assert!(complete);
        assert_eq!(
            res,
            vec![
                vec![3, -2, 0],
                vec![2, -2, 0],
                vec![3, -3, 0],
                vec![3, -1, 0],
                vec![4, -2, 0],
            ]
        );
    }

    #[test]
    fn budget_falls_back_to_nearest_plane() {
        // The first lattice point is always reached, so even a budget of nothing comes back
        // with the point Babai's nearest plane would give, and says it did not finish.
        let basis = vec![vec![2, 0], vec![0, 2]];
        let w = eye(2);
        let target = vec![5, 7];
        let babai = nearest_plane(&target, &basis, &w).unwrap();

        let (closest, complete) = cvp_exact(&target, &basis, &w, Some(0)).unwrap();
        assert!(!complete);
        assert_eq!(closest, babai);

        // Asking for more than the budget can find returns what it did find, not nothing
        let (res, complete) = cvp_top_k(&target, &basis, &w, 4, Some(0)).unwrap();
        assert!(!complete);
        assert_eq!(res, vec![babai.clone()]);

        let (res, complete) = cvp_l1_top_k(&target, &basis, &[1, 1], 4, Some(0)).unwrap();
        assert!(!complete);
        assert_eq!(res, vec![babai]);

        // The origin is the first point an SVP search reaches and is not a candidate, so the
        // shortest row stands in for it
        let skew = vec![vec![3, 4], vec![1, 0]];
        let (short, complete) = svp_exact(&skew, &w, Some(0)).unwrap();
        assert!(!complete);
        assert_eq!(short, vec![1, 0]);
    }

    #[test]
    fn cvp_l1_far_outside_span() {
        // The L1 search prunes on an L2 radius, and reading that radius off the score alone
        // makes an out-of-span offset widen it without bound: the cost used to grow with the
        // offset, 630k nodes at 1e5 and unfinished at 1e7. The lattice cannot move the last
        // coordinate at all, so the whole of its miss is part of the distance to the span.
        let basis = vec![vec![1, 0, 0], vec![0, 1, 0]];
        for offset in [10i64, 1_000, 100_000, 10_000_000, 1_000_000_000] {
            let target = vec![3, -2, offset];
            let (res, complete) =
                cvp_l1_top_k(&target, &basis, &[1, 1, 1], 3, Some(10_000)).unwrap();
            assert!(complete, "offset {offset} did not finish");
            assert_eq!(res[0], vec![3, -2, 0], "offset {offset}");
            assert_eq!(res.len(), 3);
        }

        // Only reachable in steps of 4, so the target is 1 off in that coordinate whatever
        // the search does, and the radius has to allow for it rather than assume 0
        let coarse = vec![vec![4, 0], vec![0, 7]];
        let (res, complete) = cvp_l1_top_k(&[9, 3], &coarse, &[1, 1], 1, None).unwrap();
        assert!(complete);
        assert_eq!(res[0], vec![8, 0]);
    }

    #[test]
    fn nearest_plane_ties_are_not_shared_across_norms() {
        // Rounding a coefficient can land on an exact tie, and the enumeration and
        // `nearest_plane` split it differently. Both answers are Babai's, and equally good
        // under the form the rounding used, but that says nothing about any other norm: here
        // the two are the same distance away under L2 and are not under L1. So a budget that
        // runs out is measured against where its own first descent lands, not against
        // `nearest_plane`.
        let basis = vec![
            vec![-3, -14, -20, -9, 3],
            vec![2, -12, -3, -13, 4],
            vec![-20, 16, -17, -7, -4],
            vec![-4, -13, -15, 9, 1],
            vec![-6, -6, 1, -8, 2],
        ];
        let target = vec![-1, -69, -90, 17, -79];
        let w = eye(5);
        let reduced = lll(&basis, 0.99, &w).unwrap();

        let babai = nearest_plane(&target, &reduced, &w).unwrap();
        let (first, complete) = cvp_exact(&target, &reduced, &w, Some(0)).unwrap();
        assert!(!complete);

        let diff = |x: &[i64]| -> Vec<i64> { target.iter().zip(x).map(|(&t, &c)| t - c).collect() };
        let l1 = |x: &[i64]| -> i64 { diff(x).iter().map(|e| e.abs()).sum() };

        assert_ne!(first, babai, "expected the tie to split the two apart");
        assert_eq!(
            norm_sq(&diff(&first)),
            norm_sq(&diff(&babai)),
            "both are nearest plane points, so equally far under L2"
        );
        assert_ne!(l1(&first), l1(&babai), "and not equally far under L1");
    }

    #[test]
    fn budget_large_enough_is_exact() {
        // Given room to finish, a budgeted search says so and agrees with an unbudgeted one
        let basis = vec![vec![4, 1, 0], vec![1, 5, 1], vec![0, 1, 6]];
        let w = eye(3);
        let target = vec![17, -23, 9];

        for k in [1usize, 3, 8] {
            let (bounded, complete) = cvp_top_k(&target, &basis, &w, k, Some(1_000_000)).unwrap();
            assert!(complete);
            assert_eq!(
                bounded,
                exhaustive(cvp_top_k(&target, &basis, &w, k, None).unwrap())
            );

            let (bounded, complete) =
                cvp_l1_top_k(&target, &basis, &[1, 1, 1], k, Some(1_000_000)).unwrap();
            assert!(complete);
            assert_eq!(
                bounded,
                exhaustive(cvp_l1_top_k(&target, &basis, &[1, 1, 1], k, None).unwrap())
            );
        }

        let (bounded, complete) = svp_exact(&basis, &w, Some(1_000_000)).unwrap();
        assert!(complete);
        assert_eq!(bounded, exhaustive(svp_exact(&basis, &w, None).unwrap()));
    }

    #[test]
    fn dependent_rows_rejected() {
        // A Gram-Schmidt norm of zero gives the search a level it can wander for free, so the
        // rows have to be independent rather than merely documented as such.
        let w = eye(3);
        let dependent = vec![vec![1, 2, 0], vec![0, 1, 1], vec![1, 3, 1]];
        let target = vec![4, 5, 6];

        fn is_bad<T>(e: Result<T, DiophantineError>) -> bool {
            matches!(e, Err(DiophantineError::InvalidArgument(_)))
        }
        assert!(is_bad(cvp_exact(&target, &dependent, &w, None)));
        assert!(is_bad(cvp_top_k(&target, &dependent, &w, 2, None)));
        assert!(is_bad(cvp_l1_top_k(
            &target,
            &dependent,
            &[1, 1, 1],
            2,
            None
        )));
        assert!(is_bad(svp_exact(&dependent, &w, None)));

        // A repeated row, and a row that is zero
        assert!(is_bad(svp_exact(
            &vec![vec![1, 2, 3], vec![1, 2, 3]],
            &w,
            None
        )));
        assert!(is_bad(svp_exact(
            &vec![vec![1, 2, 3], vec![0, 0, 0]],
            &w,
            None
        )));

        // The same rows scaled up, in case the test is reading an absolute size
        let big = vec![
            vec![100_000, 200_000, 0],
            vec![0, 1, 1],
            vec![100_000, 200_001, 1],
        ];
        assert!(is_bad(svp_exact(&big, &w, None)));
    }

    #[test]
    fn svp_cvp_dims() {
        let basis = vec![vec![1, 0], vec![0, 1], vec![0, 1]];
        let w = eye(3);
        let res = svp_exact(&basis, &w, None);
        assert!(matches!(res, Err(DiophantineError::InvalidDimensions(_))));

        let target = vec![1, 2, 3, 4];
        let basis = eye(3);
        let w = eye(3);
        let res = cvp_exact(&target, &basis, &w, None);
        assert!(matches!(res, Err(DiophantineError::InvalidDimensions(_))));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::{eye, integer_det};
    use proptest::prelude::*;

    fn norm_sq(v: &[i64]) -> i64 {
        v.iter().map(|x| x * x).sum()
    }

    /// Unwraps the result of a search run without a budget, asserting that it completed.
    #[track_caller]
    fn exhaustive<T>((value, complete): (T, bool)) -> T {
        assert!(complete, "a search without a budget must always complete");
        value
    }

    /// Whether `x` is an integer combination of the rows of the square, nonsingular `basis`.
    ///
    /// By Cramer's rule the coefficients are `det(B_i) / det(B)`, where `B_i` is `basis` with
    /// row `i` replaced by `x`. Exact, and unlike HNF it has no coefficient growth.
    fn in_lattice(x: &[i64], basis: &Matrix<i64>) -> bool {
        let det = integer_det(basis).unwrap();
        (0..basis.len()).all(|i| {
            let mut b_i = basis.clone();
            b_i[i] = x.to_vec();
            integer_det(&b_i).unwrap() % det == 0
        })
    }

    fn matrix(rows: usize, cols: usize, max_val: i64) -> impl Strategy<Value = Matrix<i64>> {
        proptest::collection::vec(proptest::collection::vec(-max_val..max_val, cols), rows)
    }

    prop_compose! {
        fn random_basis()(n in 2usize..=5)(mat in matrix(n, n, 30)) -> Matrix<i64> {
            mat
        }
    }

    prop_compose! {
        fn random_basis_target()(n in 2usize..=5)
        (
            basis in matrix(n, n, 20),
            target in proptest::collection::vec(-100i64..100, n),
        ) -> (Matrix<i64>, Vec<i64>) {
            (basis, target)
        }
    }

    /// B diag(q) B^T, exact
    fn weighted_gram(basis: &Matrix<i64>, q: &[i64]) -> Matrix<i64> {
        basis
            .iter()
            .map(|a| {
                basis
                    .iter()
                    .map(|b| (0..q.len()).map(|j| a[j] * q[j] * b[j]).sum())
                    .collect()
            })
            .collect()
    }

    fn combine(coeffs: &[i64], basis: &Matrix<i64>, m: usize) -> Vec<i64> {
        let mut x = vec![0; m];
        for (c, row) in coeffs.iter().zip(basis) {
            for j in 0..m {
                x[j] += c * row[j];
            }
        }
        x
    }

    /// The top `k` of the lattice points with coefficients within `r` of `center`, together
    /// with `extra`, ordered by `score` and then by the vector.
    fn bruteforce_top_k<S: Ord>(
        basis: &Matrix<i64>,
        m: usize,
        center: &[i64],
        r: i64,
        extra: &Matrix<i64>,
        k: usize,
        score: impl Fn(&[i64]) -> S,
    ) -> Matrix<i64> {
        let n = basis.len();
        let mut points = extra.clone();
        let mut coeffs: Vec<i64> = center.iter().map(|c| c - r).collect();
        loop {
            points.push(combine(&coeffs, basis, m));
            let Some(i) = (0..n).find(|&i| coeffs[i] < center[i] + r) else {
                break;
            };
            coeffs[i] += 1;
            for c in 0..i {
                coeffs[c] = center[c] - r;
            }
        }
        points.sort_by(|a, b| score(a).cmp(&score(b)).then_with(|| a.cmp(b)));
        points.dedup();
        points.truncate(k);
        points
    }

    /// A basis of `n <= m` independent rows, the coefficients of a lattice point near the
    /// target, the target, diagonal weights (possibly zero) definite on the span, and `k`.
    fn cvp_case() -> impl Strategy<Value = (Matrix<i64>, Vec<i64>, Vec<i64>, Vec<i64>, usize)> {
        (1usize..=4)
            .prop_flat_map(|m| (1..=m, Just(m)))
            .prop_flat_map(|(n, m)| {
                // A zero weight can only be definite on the span if the span is not everything
                let min_weight = if n < m { 0i64 } else { 1 };
                (
                    matrix(n, m, 6),
                    proptest::collection::vec(-3i64..=3, n),
                    proptest::collection::vec(-4i64..=4, m),
                    proptest::collection::vec(min_weight..=3, m),
                    1usize..=6,
                )
            })
            .prop_filter(
                "form must be definite on the span",
                |(basis, _, _, weights, _)| {
                    let q: Vec<i64> = weights.iter().map(|w| w * w).collect();
                    integer_det(&weighted_gram(basis, &q)).unwrap_or(0) != 0
                },
            )
            .prop_map(|(basis, center, noise, weights, k)| {
                let m = noise.len();
                let target: Vec<i64> = combine(&center, &basis, m)
                    .iter()
                    .zip(&noise)
                    .map(|(x, e)| x + e)
                    .collect();
                (basis, center, target, weights, k)
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2_000))]

        #[test]
        fn test_cvp_top_k_brute_force((basis, center, target, weights, k) in cvp_case()) {
            let m = target.len();
            let q: Vec<i64> = weights.iter().map(|w| w * w).collect();

            let mut w = vec![vec![0.0; m]; m];
            for i in 0..m {
                w[i][i] = q[i] as f64;
            }
            let l2 = |x: &[i64]| -> i64 {
                (0..m).map(|j| q[j] * (target[j] - x[j]).pow(2)).sum()
            };
            let l1 = |x: &[i64]| -> (i64, i64) {
                let a = (0..m).map(|j| weights[j] * (target[j] - x[j]).abs()).sum();
                (a, l2(x))
            };

            let reduced = lll(&basis, 0.99, &w).unwrap();

            for b in [&basis, &reduced] {
                let res = exhaustive(cvp_top_k(&target, b, &w, k, None).unwrap());
                let brute = bruteforce_top_k(&basis, m, &center, 4, &res, k, l2);
                prop_assert_eq!(&res, &brute, "L2 top-k differs from brute force");

                let res = exhaustive(cvp_l1_top_k(&target, b, &weights, k, None).unwrap());
                let brute = bruteforce_top_k(&basis, m, &center, 4, &res, k, l1);
                prop_assert_eq!(&res, &brute, "L1 top-k differs from brute force");
            }
        }

        #[test]
        fn test_cvp_top_k_basis_independent((basis, target) in random_basis_target(), k in 1usize..=8) {
            prop_assume!(integer_det(&basis).unwrap_or(0) != 0);
            let n = basis.len();
            let w = eye(n);
            let weights = vec![1; n];
            let reduced = lll(&basis, 0.99, &w).unwrap();

            prop_assert_eq!(
                exhaustive(cvp_top_k(&target, &basis, &w, k, None).unwrap()),
                exhaustive(cvp_top_k(&target, &reduced, &w, k, None).unwrap())
            );
            prop_assert_eq!(
                exhaustive(cvp_l1_top_k(&target, &basis, &weights, k, None).unwrap()),
                exhaustive(cvp_l1_top_k(&target, &reduced, &weights, k, None).unwrap())
            );
            prop_assert_eq!(
                exhaustive(cvp_top_k(&target, &reduced, &w, 1, None).unwrap())[0].clone(),
                exhaustive(cvp_exact(&target, &reduced, &w, None).unwrap())
            );
        }

        /// Whatever the budget, a search comes back with lattice points, ranked, no worse
        /// than the nearest plane, and equal to the unbudgeted answer when it says it
        /// finished.
        #[test]
        fn test_budget_properties(
            (basis, target) in random_basis_target(),
            k in 1usize..=6,
            max_nodes in 0u64..60,
        ) {
            prop_assume!(integer_det(&basis).unwrap_or(0) != 0);
            let n = basis.len();
            let w = eye(n);
            let weights = vec![1; n];
            let reduced = lll(&basis, 0.99, &w).unwrap();

            let dist = |x: &[i64]| -> i64 {
                norm_sq(&target.iter().zip(x).map(|(&t, &c)| t - c).collect::<Vec<_>>())
            };
            // What the first descent reaches on its own, which is where every budget starts.
            // Compared against instead of `nearest_plane`: both compute the same thing, but
            // an exact tie when rounding a coefficient can send them to different points,
            // equally good under `w` and not necessarily under the L1 norm below.
            let floor = &cvp_top_k(&target, &reduced, &w, 1, Some(0)).unwrap().0[0];

            let (res, complete) = cvp_top_k(&target, &reduced, &w, k, Some(max_nodes)).unwrap();

            // The first lattice point is always reached, and no more than k are kept
            prop_assert!(!res.is_empty(), "a budgeted search returned nothing");
            prop_assert!(res.len() <= k);

            for x in &res {
                prop_assert!(in_lattice(x, &reduced), "not a lattice point");
            }
            // Still ordered by distance, and still better than where the search started
            let dists: Vec<i64> = res.iter().map(|x| dist(x)).collect();
            prop_assert!(dists.windows(2).all(|d| d[0] <= d[1]), "not ordered by distance");
            prop_assert!(dists[0] <= dist(floor), "more budget did worse than none");

            let exact = exhaustive(cvp_top_k(&target, &reduced, &w, k, None).unwrap());
            if complete {
                prop_assert_eq!(&res, &exact, "a finished search should be the exact answer");
            }
            // Running out of budget can only cost quality, never improve on the exact answer
            prop_assert!(dist(&exact[0]) <= dists[0]);

            // The L1 search keeps the same guarantees under its own norm
            let l1 = |x: &[i64]| -> i64 {
                target.iter().zip(x).map(|(&t, &c)| (t - c).abs()).sum()
            };
            let (res, complete) =
                cvp_l1_top_k(&target, &reduced, &weights, k, Some(max_nodes)).unwrap();
            prop_assert!(!res.is_empty());
            prop_assert!(res.len() <= k);
            let l1s: Vec<i64> = res.iter().map(|x| l1(x)).collect();
            prop_assert!(l1s.windows(2).all(|d| d[0] <= d[1]), "not ordered by L1 distance");
            let floor_l1 = &cvp_l1_top_k(&target, &reduced, &weights, 1, Some(0)).unwrap().0[0];
            prop_assert!(l1s[0] <= l1(floor_l1), "more budget did worse than none");
            if complete {
                let exact = exhaustive(cvp_l1_top_k(&target, &reduced, &weights, k, None).unwrap());
                prop_assert_eq!(&res, &exact);
            }

            // An SVP search always has a nonzero vector to fall back on
            let (short, complete) = svp_exact(&reduced, &w, Some(max_nodes)).unwrap();
            prop_assert!(short.iter().any(|&x| x != 0), "SVP returned the zero vector");
            prop_assert!(in_lattice(&short, &reduced));
            prop_assert!(norm_sq(&short) <= norm_sq(&reduced[0]), "worse than the shortest row");
            if complete {
                prop_assert_eq!(&short, &exhaustive(svp_exact(&reduced, &w, None).unwrap()));
            }
        }

        /// How far a target is from the span of the lattice is the same for every lattice
        /// point, so however large it grows it should change neither the answer nor the work
        /// needed to find it.
        #[test]
        fn test_out_of_span_offset_is_free(
            (basis, target) in random_basis_target(),
            offset in 1i64..1_000_000_000,
            k in 1usize..=4,
        ) {
            prop_assume!(integer_det(&basis).unwrap_or(0) != 0);
            let n = basis.len();
            let w = eye(n + 1);

            // Lift into one more dimension, which the lattice does not reach into, so that
            // the last coordinate of the target is purely outside the span
            let lifted: Matrix<i64> = basis
                .iter()
                .map(|row| row.iter().copied().chain([0]).collect())
                .collect();
            let reduced = lll(&lifted, 0.99, &w).unwrap();

            let near: Vec<i64> = target.iter().copied().chain([0]).collect();
            let far: Vec<i64> = target.iter().copied().chain([offset]).collect();

            // Enough to finish either search many times over if the offset stays out of it
            let budget = Some(100_000);
            let (near_res, near_done) = cvp_top_k(&near, &reduced, &w, k, budget).unwrap();
            let (far_res, far_done) = cvp_top_k(&far, &reduced, &w, k, budget).unwrap();

            prop_assert!(near_done && far_done, "an offset out of the span enlarged the search");

            // Compared by distance rather than by vector: past the point where squared
            // distances to the target stay exact, the search ranks by distances to its
            // projection instead, and equidistant points no longer tie exactly enough for
            // the tie to break the same way. Which of them comes back may differ; how good
            // it is may not.
            let dists = |res: &Matrix<i64>, v: &[i64]| -> Vec<i64> {
                res.iter()
                    .map(|x| norm_sq(&v.iter().zip(x).map(|(&t, &c)| t - c).collect::<Vec<_>>()))
                    .collect()
            };
            // Both measured against the target without the offset. Every lattice point is
            // the same amount further from the target with it, so the two are the same
            // comparison, but this one is not dominated by that shared amount.
            prop_assert_eq!(
                dists(&near_res, &near),
                dists(&far_res, &near),
                "an offset out of the span changed how close the search got"
            );

            // Same for the L1 search, which reaches its radius by a longer route
            let weights = vec![1; n + 1];
            let (near_res, near_done) =
                cvp_l1_top_k(&near, &reduced, &weights, k, budget).unwrap();
            let (far_res, far_done) = cvp_l1_top_k(&far, &reduced, &weights, k, budget).unwrap();
            prop_assert!(
                near_done && far_done,
                "an offset out of the span enlarged the L1 search"
            );
            let l1s = |res: &Matrix<i64>, v: &[i64]| -> Vec<i64> {
                res.iter()
                    .map(|x| v.iter().zip(x).map(|(&t, &c)| (t - c).abs()).sum())
                    .collect::<Vec<i64>>()
            };
            prop_assert_eq!(
                l1s(&near_res, &near),
                l1s(&far_res, &near),
                "an offset out of the span changed how close the L1 search got"
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10_000))]

        #[test]
        fn test_lll_properties(
            basis in random_basis()
        ) {
            // Skip singular matrices
            let det_orig = integer_det(&basis).unwrap_or(0);
            prop_assume!(det_orig != 0);

            let n = basis.len();

            let w = eye(n);
            let delta = 0.75;
            let reduced = lll(&basis, delta, &w).unwrap();

            // Determinant is preserved
            let det_red = integer_det(&reduced).unwrap();
            prop_assert_eq!(det_orig.abs(), det_red.abs(), "Determinant magnitude changed!");

            let ortho = gramschmidt(&reduced, &w);
            let n = reduced.len();

            // Size Reduction Condition: |mu_{i,j}| <= 0.5
            for i in 0..n {
                for j in 0..i {
                    let mu_ij = mu(&reduced, &ortho, &w, i, j);
                    // Add tiny epsilon for f64 math
                    prop_assert!(
                        mu_ij.abs() <= 0.500001,
                        "Size reduction failed: mu_{},{} = {}", i, j, mu_ij
                    );
                }
            }

            // Lovasz Condition: delta * ||b*_{k-1}||^2 <= ||b*_k||^2 + mu_{k, k-1}^2 ||b*_{k-1}||^2
            for k in 1..n {
                let mu_k_k1 = mu(&reduced, &ortho, &w, k, k - 1);
                let norm_ortho_k1 = inner_prod(&ortho[k - 1], &ortho[k - 1], &w);
                let norm_ortho_k = inner_prod(&ortho[k], &ortho[k], &w);

                let lhs = delta * norm_ortho_k1;
                let rhs = norm_ortho_k + mu_k_k1.powi(2) * norm_ortho_k1;

                prop_assert!(
                    lhs <= rhs + 1e-6,
                    "Lovasz condition failed at step {}: {} > {}", k, lhs, rhs
                );
            }
        }

        #[test]
        fn test_nearest_plane_properties((basis, target) in random_basis_target()) {
            prop_assume!(integer_det(&basis).unwrap_or(0) != 0);

            let n = basis.len();
            let w = eye(n);

            // Nearest plane requires an LLL-reduced basis to work effectively
            let reduced = lll(&basis, 0.75, &w).unwrap();
            let np = nearest_plane(&target, &reduced, &w).unwrap();

            // The result must be a point on the lattice.
            prop_assert!(in_lattice(&np, &reduced), "nearest_plane result is not an integer combination of the basis!");

            // The error vector must fall within the fundamental parallelepiped
            // of the Gram-Schmidt basis.
            let ortho = gramschmidt(&reduced, &w);
            let error: Vec<f64> = target.iter().zip(np.iter()).map(|(&t, &p)| (t - p) as f64).collect();

            for i in 0..n {
                let a = &ortho[i];
                let num = inner_prod(a, &error, &w);
                let den = inner_prod(a, a, &w);
                let mu_err = if den.abs() < 1e-9 { 0.0 } else { num / den };

                prop_assert!(
                    mu_err.abs() <= 0.500001,
                    "Error vector projection onto GS vector {} exceeds 0.5: {}", i, mu_err
                );
            }
        }
        #[test]
        fn test_svp_exact_properties(basis in random_basis()) {
            let det_orig = integer_det(&basis).unwrap_or(0);
            prop_assume!(det_orig != 0);

            let n = basis.len();
            let w = eye(n);

            // LLL-reduce first
            let reduced = lll(&basis, 0.99, &w).unwrap();
            let svp_res = exhaustive(svp_exact(&reduced, &w, None).unwrap());

            // Result must be non-zero
            prop_assert!(svp_res.iter().any(|&x| x != 0), "SVP exact returned the zero vector!");

            let svp_norm = norm_sq(&svp_res);
            let lll_first_norm = norm_sq(&reduced[0]);

            // SVP must find a vector at least as short as LLL's best approximation
            prop_assert!(
                svp_norm <= lll_first_norm,
                "SVP exact found a longer vector ({}) than LLL ({})", svp_norm, lll_first_norm
            );

            // Must be a valid lattice point
            prop_assert!(in_lattice(&svp_res, &reduced), "SVP exact result is not in the lattice!");

            // No nonzero point with small coefficients is shorter
            let box_min = bruteforce_top_k(&reduced, n, &vec![0; n], 2, &vec![], 2, norm_sq)
                .iter()
                .map(|x| norm_sq(x))
                .find(|&s| s != 0)
                .unwrap();
            prop_assert!(
                svp_norm <= box_min,
                "SVP exact ({}) is longer than a small combination ({})", svp_norm, box_min
            );
        }

        #[test]
        fn test_cvp_exact_properties((basis, target) in random_basis_target()) {
            let det_orig = integer_det(&basis).unwrap_or(0);
            prop_assume!(det_orig != 0);

            let n = basis.len();
            let w = eye(n);

            // Perform CVP on a reduced basis
            let reduced = lll(&basis, 0.99, &w).unwrap();

            let cvp_res = exhaustive(cvp_exact(&target, &reduced, &w, None).unwrap());
            let babai_res = nearest_plane(&target, &reduced, &w).unwrap();

            let error_cvp: Vec<i64> = target.iter().zip(cvp_res.iter()).map(|(&t, &c)| t - c).collect();
            let error_babai: Vec<i64> = target.iter().zip(babai_res.iter()).map(|(&t, &c)| t - c).collect();

            let dist_cvp = norm_sq(&error_cvp);
            let dist_babai = norm_sq(&error_babai);

            // Exact CVP must always yield a distance <= the approximate Babai's nearest plane distance
            prop_assert!(
                dist_cvp <= dist_babai,
                "Exact CVP yielded worse distance {} than Babai's approximate {}", dist_cvp, dist_babai
            );

            // Must be a valid lattice point (Linear combination of the reduced basis)
            prop_assert!(in_lattice(&cvp_res, &reduced), "CVP exact result is not in the lattice!");
        }
    }
}
