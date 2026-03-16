//! Utility functions.

use crate::Matrix;
use std::fmt::Display;

// Pretty print a matrix.
// Currently unused, but helpful for debugging.
#[allow(unused)]
fn pretty_print<T: Display>(matrix: &Matrix<T>) {
    if matrix.is_empty() {
        println!("[[]]");
        return;
    }

    // Convert all values to strings and find the max width per column
    let stringified: Vec<Vec<String>> = matrix
        .iter()
        .map(|row| row.iter().map(|v| v.to_string()).collect())
        .collect();

    let col_count = stringified.iter().map(|r| r.len()).max().unwrap_or(0);

    let col_widths: Vec<usize> = (0..col_count)
        .map(|c| {
            stringified
                .iter()
                .filter_map(|row| row.get(c))
                .map(|s| s.len())
                .max()
                .unwrap_or(0)
        })
        .collect();

    let n = stringified.len();
    for (i, row) in stringified.iter().enumerate() {
        let cells: String = (0..col_count)
            .map(|c| {
                let s = row.get(c).map(String::as_str).unwrap_or("");
                format!(" {:>width$} ", s, width = col_widths[c])
            })
            .collect::<Vec<_>>()
            .join("");

        if i == 0 {
            println!("[[{cells}]");
        } else if i == n - 1 {
            println!(" [{cells}]]");
        } else {
            println!(" [{cells}]");
        }
    }
}
