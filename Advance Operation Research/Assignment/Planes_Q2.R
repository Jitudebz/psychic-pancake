
library(lpSolve)

#set coefficents of the objective function

f.obj <- c(4.2,3,2.3)

# Set matrix corresponding to coefficients of constraints by rows
# Do not consider the non-negative constraint; it is automatically assumed
f.con <- matrix(c(67,50,35,1,1,1,5/3,4/3,1), nrow = 3, byrow = TRUE)


# Set unequality signs
f.dir <- c("<=",
           "<=",
           "<=")


# Set right hand side coefficients
f.rhs <- c(1500,
           30,
           40)


# Final value (z)
Z <- lp("max", f.obj, f.con, f.dir, f.rhs, compute.sens=TRUE, all.int = T)
Z

# Variables final values
Z$solution