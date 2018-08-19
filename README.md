# Conditional Mutual Information Maximization (CMIM.jl)

Fast Binary Feature Selection Method in Julia

The implementation is based on the original [Fleuret's paper](www.jmlr.org/papers/volume5/fleuret04a/fleuret04a.pdf):
Fleuret, Francois. (2004). Fast Binary Feature Selection with Conditional Mutual Information. Journal of Machine Learning Research. 5. 1531-1555.


## Installation
CMIM.jl can be installed using Julia's package manager
```julia
Pkg.clone("https://github.com/komartom/CMIM.jl.git")
```

## Simple example
Feature selection on a text-based dataset
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