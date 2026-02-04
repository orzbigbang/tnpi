use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

fn zscore_inplace(x: &mut [f64]) {
    let mut sum = 0.0;
    for v in x.iter() {
        sum += *v;
    }
    let mean = sum / (x.len() as f64);
    let mut var = 0.0;
    for v in x.iter() {
        let d = *v - mean;
        var += d * d;
    }
    let sd = (var / (x.len() as f64)).sqrt();
    if !sd.is_finite() || sd < 1e-12 {
        for v in x.iter_mut() {
            *v = *v - mean;
        }
        return;
    }
    for v in x.iter_mut() {
        *v = (*v - mean) / sd;
    }
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x - *y;
        sum += d * d;
    }
    (sum / (a.len() as f64)).sqrt()
}

#[pyfunction]
fn best_shift_ms(
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
    grid_ms: i32,
    search_ms: i32,
    min_points: usize,
) -> PyResult<f64> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    if a_slice.len() < min_points || b_slice.len() < min_points {
        return Ok(f64::NAN);
    }
    if grid_ms <= 0 || search_ms <= 0 {
        return Ok(f64::NAN);
    }

    let mut a_vec = a_slice.to_vec();
    let mut b_vec = b_slice.to_vec();
    zscore_inplace(&mut a_vec);
    zscore_inplace(&mut b_vec);

    let max_shift = (search_ms / grid_ms) as isize;
    let mut best_shift: isize = 0;
    let mut best_rmse = f64::INFINITY;

    for sh in -max_shift..=max_shift {
        let (aa, bb) = if sh < 0 {
            let s = (-sh) as usize;
            if s >= a_vec.len() {
                continue;
            }
            let aa = &a_vec[s..];
            let bb = &b_vec[..aa.len().min(b_vec.len())];
            (aa, bb)
        } else if sh > 0 {
            let s = sh as usize;
            if s >= b_vec.len() {
                continue;
            }
            let bb = &b_vec[s..];
            let aa = &a_vec[..bb.len().min(a_vec.len())];
            (aa, bb)
        } else {
            let len = a_vec.len().min(b_vec.len());
            (&a_vec[..len], &b_vec[..len])
        };

        if aa.len() < min_points || bb.len() < min_points {
            continue;
        }
        let cur = rmse(aa, bb);
        if cur < best_rmse {
            best_rmse = cur;
            best_shift = sh;
        }
    }

    Ok((best_shift as f64) * (grid_ms as f64))
}

#[pymodule]
fn odg_accel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(best_shift_ms, m)?)?;
    Ok(())
}
