load_all()

library(AER)

# load fatalities data from AER package
data(Fatalities)

# run regular model with state & year fixed effects
regular_model <- fepoisson(fatal ~ beertax | state + year | state, data = Fatalities)
summary(regular_model)

# Formula: fatal ~ beertax | state + year | state

# Family: Poisson

# Estimates:

# |         | Estimate | Std. Error | z value | Pr(>|z|) |
# |---------|----------|------------|---------|----------|
# | beertax |  -0.3473 |     0.1728 | -2.0096 | 0.0445 * |

# Significance codes: ** p < 0.01; * p < 0.05; + p < 0.10

# Pseudo R-squared: 0.9925 

# Fixed effects:
#   state: 48
#   year: 7

# Number of observations: Full 336; Missing 0; Perfect classification 0 

# Number of Fisher Scoring iterations: 6

# === COMPARE WITH PPMLHDFE ====

# . * import the data
# . import delimited using  "fatalities.csv"
# (34 vars, 336 obs)

# * run the model
# ppmlhdfe fatal beertax, absorb(state year) vce(cl state) d
# Iteration 1:   deviance = 7.5532e+03  eps = .         iters = 3    tol = 1.0e-04  min(eta) =  -1.43  P   
# Iteration 2:   deviance = 1.6438e+03  eps = 3.59e+00  iters = 2    tol = 1.0e-04  min(eta) =  -1.97      
# Iteration 3:   deviance = 1.4156e+03  eps = 1.61e-01  iters = 2    tol = 1.0e-04  min(eta) =  -2.18      
# Iteration 4:   deviance = 1.4138e+03  eps = 1.25e-03  iters = 2    tol = 1.0e-04  min(eta) =  -2.21      
# Iteration 5:   deviance = 1.4138e+03  eps = 1.64e-07  iters = 2    tol = 1.0e-04  min(eta) =  -2.21      
# Iteration 6:   deviance = 1.4138e+03  eps = 3.08e-15  iters = 1    tol = 1.0e-05  min(eta) =  -2.21   S  
# Iteration 7:   deviance = 1.4138e+03  eps = 1.47e-16  iters = 1    tol = 1.0e-08  min(eta) =  -2.21   S O
# ------------------------------------------------------------------------------------------------------------
# (legend: p: exact partial-out   s: exact solver   h: step-halving   o: epsilon below tolerance)
# Converged in 7 iterations and 13 HDFE sub-iterations (tol = 1.0e-08)

# HDFE PPML regression                              No. of obs      =        336
# Absorbing 2 HDFE groups                           Residual df     =         47
# Statistics robust to heteroskedasticity           Wald chi2(1)    =       4.04
# Deviance             =  1413.784422               Prob > chi2     =     0.0445
# Log pseudolikelihood = -2095.542592               Pseudo R2       =     0.9825

# Number of clusters (state)  =         48
#                                  (Std. Err. adjusted for 48 clusters in state)
# ------------------------------------------------------------------------------
#              |               Robust
#        fatal |      Coef.   Std. Err.      z    P>|z|     [95% Conf. Interval]
# -------------+----------------------------------------------------------------
#      beertax |  -.3472736   .1728049    -2.01   0.044    -.6859649   -.0085822
#        _cons |   7.396313   .0928461    79.66   0.000     7.214338    7.578288
# ------------------------------------------------------------------------------

# Absorbed degrees of freedom:
# -----------------------------------------------------+
#  Absorbed FE | Categories  - Redundant  = Num. Coefs |
# -------------+---------------------------------------|
#        state |        48          48           0    *|
#         year |         7           1           6     |
# -----------------------------------------------------+
# * = FE nested within cluster; treated as redundant for DoF computation
