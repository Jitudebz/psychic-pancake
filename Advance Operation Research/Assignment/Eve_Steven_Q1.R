# Import lpSolve package
library(lpSolve)

# Set coefficients of the objective function
f.obj <- c(4.5, 4.9, 7.8, 7.2, 3.6, 4.3, 2.9, 3.1)

# Set matrix corresponding to coefficients of constraints by rows
# Do not consider the non-negative constraint; it is automatically assumed
f.con <- matrix(c(1, 0, 1, 0, 1, 0, 1, 0,
                  0, 1, 0, 1, 0, 1, 0, 1,
                  1, 1, 0, 0, 0, 0, 0, 0, 
                  0, 0, 1, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 1, 1, 0, 0,
                  0, 0, 0, 0, 0, 0, 1, 1), nrow = 6, byrow = TRUE)

# Set unequality signs
f.dir <- c("=",
           "=",
           "=",
           "=",
           "=",
           "=")

# Set right hand side coefficients
f.rhs <- c(2,
           2,
           1,
           1,
           1,
           1)

# Final value (z)
Z = lp("min", f.obj, f.con, f.dir, f.rhs, all.bin = TRUE)
Z

# Variables final values
Z$solution