# jaxinterp2d

Basic bilinear interpolation on grids with jax. This package provides one function:
`interp2d(x, y, xp, yp, zp)`.

## Getting started

Install with

```bash
pip install jaxinterp2d
```

This will install jax and jaxlib if you don't already have them. The test
requires numpy and can be run with `pytest`. Check them out for a usage example.

Note that `interp2d` does not do any bounds checking.
