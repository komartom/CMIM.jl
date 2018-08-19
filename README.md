# Conditional Mutual Information Maximization (CMIM.jl)
[![Build Status](https://travis-ci.org/komartom/CMIM.jl.svg?branch=master)](https://travis-ci.org/komartom/CMIM.jl)
[![codecov.io](http://codecov.io/github/komartom/CMIM.jl/coverage.svg?branch=master)](http://codecov.io/github/komartom/CMIM.jl?branch=master)

Julia implementation of CMIM feature selection algorithm proposed in [Fleuret, Francois. (2004). Fast Binary Feature Selection with Conditional Mutual Information. Journal of Machine Learning Research. 5. 1531-1555](http://www.jmlr.org/papers/volume5/fleuret04a/fleuret04a.pdf)


## Installation
CMIM.jl can be installed using Julia's package manager
```julia
Pkg.clone("https://github.com/komartom/CMIM.jl.git")
```

## Simple example
Selecting the best 100 features on a text-based dataset
```julia
using CMIM, MAT

# Download BASEHOCK dataset
download("http://featureselection.asu.edu/download_file.php?filename=BASEHOCK.mat&dir=files/datasets/", "BASEHOCK.mat")
dataset = matread("./BASEHOCK.mat")

# Binarize features (word counts)
X = convert(Matrix{Bool}, dataset["X"] .> 0)
Y = convert(Vector{Bool}, dataset["Y"][:] .> 1)

# Select the best 100 features
features = cmim(100, X, Y)
```
