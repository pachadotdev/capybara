#' srr_stats (tests)
#' @srrstats {G5.2} Validates that covariance matrices align with theoretical expectations under different estimation methods.
#' @srrstats {RE3.3} Ensures consistency of `vcov` results for sandwich and outer-product estimators.
#' @srrstats {RE4.3} Confirms robustness of covariance matrix calculations under varied model specifications.
#' @srrstats {RE6.0} Ensures that covariance estimations respond correctly to model clustering and input variations.
#' @noRd
NULL

source(system.file("tinytest", "helper.R", package = "capybara"))

# vcov returns correct structure for feglm"
local({
  skip_on_cran()

  # IID  (no cluster part in formula)
  fml <- mpg ~ wt + disp | cyl
  fit_iid <- felm(fml, data = mtcars, vcov = "iid")
  vcov_iid <- vcov(fit_iid)

  # Heteroskedastic-robust (HC0)
  fit_hetero <- felm(fml, data = mtcars, vcov = "hetero")
  vcov_hetero <- vcov(fit_hetero)

  # One-way
  fml2 <- update(Formula::as.Formula(fml), . ~ . | . | gear)

  fit_gear <- felm(fml2, data = mtcars, vcov = "cluster")
  vcov_gear <- vcov(fit_gear)

  fit_carb <- felm(update(fml2, . ~ . | . | carb), data = mtcars, vcov = "cluster")
  vcov_carb <- vcov(fit_carb)

  # Dyadic-robust: Cameron-Miller (2014) sandwich with cross-dyad correlations
  fit_dyadic <- felm(update(fml2, . ~ . | . | gear + carb), data = mtcars, vcov = "dyadic")
  vcov_dyadic <- vcov(fit_dyadic)

  # the determinants must be different
  expect_true(det(vcov_iid) != det(vcov_hetero))
  expect_true(det(vcov_iid) != det(vcov_gear))
  expect_true(det(vcov_iid) != det(vcov_carb))
  expect_true(det(vcov_iid) != det(vcov_dyadic))

  # R re-computation

  fit <- felm(fml, data = mtcars, control = fit_control(keep_tx = TRUE, return_hessian = TRUE))
  vcov_hetero2 <- sandwich_vcov(fit, type = "hetero")
  vcov_gear2 <- sandwich_vcov(fit, cluster1 = mtcars$gear, type = "clustered")
  vcov_carb2 <- sandwich_vcov(fit, cluster1 = mtcars$carb, type = "clustered")
  vcov_dyadic2 <- sandwich_vcov(fit, cluster1 = mtcars$gear, cluster2 = mtcars$carb, type = "dyadic")

  expect_true(all.equal(vcov_hetero, vcov_hetero2))
  expect_true(all.equal(vcov_gear, vcov_gear2))
  expect_true(all.equal(vcov_carb, vcov_carb2))
  expect_true(all.equal(vcov_dyadic, vcov_dyadic2))
})
