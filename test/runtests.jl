using CMIM
using Base.Test


# Testing data
X = Bool[true true true false false false true; true false true false false false true; false false false true false false true; false true true false true false true; true false false true false false true; false true true true true false true; true false false false false false true; true true true true true false false; false true false true false false false; true true true true false false false; true false false false true false true; false false false false false true true; true true true false false true true; true true false true false true false; false false true true false true true; false true false false true false true; true false false false false true true]
y = Bool[true, true, true, false, true, false, true, true, false, false, false, true, true, true, false, true, true]


# Assign a score to all features
features = cmim(size(X, 2), X, y)


# Compare the results against R implementation of CMIM algorithm
# https://www.rdocumentation.org/packages/praznik/versions/5.0.0/topics/CMIM
@test first.(features) == [1, 3, 5, 4, 2, 6, 7]
@test ≈(last.(features), [0.0736965, 0.0426288, 0.041282, 0.0395759, 0.00908506, 0.00341897, 0.0], atol=0.00001)
