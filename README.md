# PDMatsSingular

[![Build Status](https://github.com/olivierverdier/PDMatsSingular.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/olivierverdier/PDMatsSingular.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/olivierverdier/PDMatsSingular.jl/graph/badge.svg?token=oHxH3LVZ9a)](https://codecov.io/gh/olivierverdier/PDMatsSingular.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://olivierverdier.github.io/PDMatsSingular.jl/)

This package allows to use `PDMats` with singular covariance matrices.

For instance:
```julia
Σ = covariance_from([1; 0;;])
```
Then:
```julia
Matrix(Σ)
```
gives
```
2×2 Matrix{Int64}:
 1  0
 0  0
```

The random variable with this covariance can be sampled as follows.
With
```julia
rng = Random.default_rng()
```

```julia
sample(rng, Σ)
```
gives samples with the singular covariance above, that is, the second component is guaranteed to be zero.


