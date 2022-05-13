# jaxinterp2d

Basic bilinear interpolation on grids with jax. This package provides a class `CartesianGrid`
for data defined on a regular grid and `interp2d(x, y, xp, yp, zp)` for data on
irregular grids. `CartesianGrid` is a jax clone of the class from [regulargrid](https://github.com/JohannesBuchner/regulargrid).

## Getting started

Install by cloning the repository and running

```bash
pip install .
```

You can install this directly via

```bash
python -m pip install git+https://github.com/adam-coogan/jaxinterp2d.git@master
```

This will install jax and jaxlib if you don't already have them. The test
requires numpy and can be run with `pytest`. Check them out for a usage example
for `interp2d`.

Note that `interp2d` does not do any bounds checking.
