Build (Windows):
1) Install Rust + Python build deps (maturin).
2) In this folder: `maturin develop` or `maturin build`.

Runtime:
- Python will use Rust automatically if `odg_accel` is importable.
- Fallback is the pure-Python/Numpy implementation.
