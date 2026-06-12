# srr_stats (tests)
# {G1.0} Implements tests to detect deterministic relations among predictors.
# {RE2.2} Ensures that models correctly fail when predictors are linearly dependent.
# {RE5.1} Confirms that the function provides meaningful error messages for invalid input.
# {RE5.2} Verifies that the model throws an error when dependent columns are included in the formula.
# {RE5.4} Checks robustness against deterministic linear relationships in the design matrix.
# {RE7.0} Exact relationships return a collinearity error.
# {RE7.0a} Perfectly noiseless input data is rejected, we have the `solve()` function for that.

# deterministic relations
local({
  set.seed(123)
  d <- data.frame(
    y = rnorm(100),
    f = 1
  )

  d$x <- 2 * d$y

  d$x2 <- d$x

  d$x2[1] <- NA

  # the solution is beta = 0.5 but we have the solve() function to
  # solve a linear system of equations!
  s <- summary(feglm(y ~ x2 | f, d))
  expect_equal(s[["nobs"]][["nobs_full"]] - 1L, s[["nobs"]][["nobs"]])
})
