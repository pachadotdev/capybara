#' srr_stats (tests)
#' @srrstats {G5.2} Validates that covariance matrices align with theoretical expectations under different estimation methods.
#' @srrstats {RE3.3} Ensures consistency of `vcov` results for sandwich and outer-product estimators.
#' @srrstats {RE4.3} Confirms robustness of covariance matrix calculations under varied model specifications.
#' @srrstats {RE6.0} Ensures that covariance estimations respond correctly to model clustering and input variations.
#' @noRd
NULL

test_that("vcov returns correct structure for feglm", {
  skip_on_cran()

  # Model without clustering - returns inverse Hessian
  m1 <- fepoisson(mpg ~ wt + disp | cyl, mtcars)

  v1 <- vcov(m1)
  expect_true(is.matrix(v1))
  expect_equal(nrow(v1), 2L)
  expect_equal(ncol(v1), 2L)
  expect_true(all(is.finite(v1)))

  # Model with clustering - returns sandwich vcov
  m2 <- fepoisson(mpg ~ wt + disp | cyl | carb, mtcars)

  v2 <- vcov(m2)
  expect_true(is.matrix(v2))
  expect_equal(nrow(v2), 2L)
  expect_equal(ncol(v2), 2L)
  expect_true(all(is.finite(v2)))
})

test_that("vcov is precomputed during fitting", {
  skip_on_cran()

  # Model without clustering
  m1 <- fepoisson(mpg ~ wt + disp | cyl, mtcars)
  expect_true(!is.null(m1$vcov))

  # Model with clustering
  m2 <- fepoisson(mpg ~ wt + disp | cyl | carb, mtcars)
  expect_true(!is.null(m2$vcov))
})

test_that("sandwich vcov is symmetric and positive semi-definite", {
  skip_on_cran()

  m <- fepoisson(mpg ~ wt + disp | cyl | carb, mtcars)
  v <- vcov(m)

  # Should be symmetric
  expect_equal(v, t(v))

  # Should be positive semi-definite (all eigenvalues >= 0)
  eigen_vals <- eigen(v, symmetric = TRUE, only.values = TRUE)$values
  expect_true(all(eigen_vals >= -1e-10)) # Allow small numerical error
})

test_that("clustered SEs are larger with positive within-cluster correlation", {
  skip_on_cran()

  # Simulate panel data with strong positive within-cluster correlation
  # In this case, sandwich SEs should be larger than model-based SEs
  set.seed(42)

  n_clusters <- 30
  n_per_cluster <- 10
  n <- n_clusters * n_per_cluster

  # Create cluster IDs and fixed effects
  cluster_id <- rep(1:n_clusters, each = n_per_cluster)
  fe <- rep(1:3, length.out = n)

  # Strong cluster-level effect (creates positive within-cluster correlation)
  cluster_effect <- rnorm(n_clusters, sd = 1.0)[cluster_id]

  # Covariates - also add cluster-level variation to induce correlation
  x1 <- rnorm(n) + 0.5 * cluster_effect
  x2 <- rnorm(n) + 0.3 * cluster_effect

  # Linear predictor
  eta <- 0.5 + 0.3 * x1 - 0.2 * x2 + cluster_effect

  # Poisson response
  y <- rpois(n, lambda = exp(eta))

  sim_data <- data.frame(
    y = y,
    x1 = x1,
    x2 = x2,
    fe = factor(fe),
    cluster_id = factor(cluster_id)
  )

  # Fit models with and without clustering
  fit_no_cluster <- fepoisson(y ~ x1 + x2 | fe, data = sim_data)
  fit_clustered <- fepoisson(y ~ x1 + x2 | fe | cluster_id, data = sim_data)

  # Get vcov matrices
  v_hessian <- vcov(fit_no_cluster)
  v_sandwich <- vcov(fit_clustered)

  # With positive within-cluster correlation, sandwich SEs should be larger
  se_hessian <- sqrt(diag(v_hessian))
  se_sandwich <- sqrt(diag(v_sandwich))

  expect_true(all(se_sandwich >= se_hessian * 0.99)) # Allow tiny numerical tolerance
})

test_that("clustered vs non-clustered vcov give different results", {
  skip_on_cran()

  # Fit same model with and without clustering
  m_no_cluster <- fepoisson(mpg ~ wt + disp | cyl, mtcars)
  m_clustered <- fepoisson(mpg ~ wt + disp | cyl | carb, mtcars)

  v_hessian <- vcov(m_no_cluster)
  v_sandwich <- vcov(m_clustered)

  # Different methods should give different results
  expect_false(isTRUE(all.equal(
    v_hessian,
    v_sandwich,
    check.attributes = FALSE
  )))
})

test_that("vcov works for felm models", {
  skip_on_cran()

  m <- felm(mpg ~ wt + disp | cyl, mtcars)
  v <- vcov(m)

  expect_true(is.matrix(v))
  expect_equal(nrow(v), 2L)
  expect_equal(ncol(v), 2L)
  expect_true(all(is.finite(v)))
})

test_that("vcov works for felm with clustering", {
  skip_on_cran()

  m <- felm(mpg ~ wt + disp | cyl | carb, mtcars)
  v <- vcov(m)

  expect_true(is.matrix(v))
  expect_equal(nrow(v), 2L)
  expect_equal(ncol(v), 2L)
  expect_true(all(is.finite(v)))
})

test_that("vcov has correct row and column names", {
  skip_on_cran()

  m <- fepoisson(mpg ~ wt + disp | cyl, mtcars)
  v <- vcov(m)

  expect_equal(rownames(v), c("wt", "disp"))
  expect_equal(colnames(v), c("wt", "disp"))
})

test_that("vcov works with single predictor", {
  skip_on_cran()

  m <- fepoisson(mpg ~ wt | cyl, mtcars)
  v <- vcov(m)

  expect_true(is.matrix(v))
  expect_equal(nrow(v), 1L)
  expect_equal(ncol(v), 1L)
})

test_that("vcov works for binomial feglm", {
  skip_on_cran()

  m <- feglm(am ~ wt + disp | cyl, mtcars, family = binomial())
  v <- vcov(m)

  expect_true(is.matrix(v))
  expect_equal(nrow(v), 2L)
  expect_equal(ncol(v), 2L)
})

test_that("vcov works for fenegbin", {
  skip_on_cran()

  m <- fenegbin(mpg ~ wt | cyl, mtcars)
  v <- vcov(m)

  expect_true(is.matrix(v))
  expect_true(all(is.finite(v)))
})
