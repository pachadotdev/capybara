# srr_stats (tests)
# {G5.2} Validates that covariance matrices align with theoretical expectations under different estimation methods.
# {RE3.3} Ensures consistency of `vcov` results for sandwich and outer-product estimators.
# {RE4.3} Confirms robustness of covariance matrix calculations under varied model specifications.
# {RE6.0} Ensures that covariance estimations respond correctly to model clustering and input variations.

source(system.file("tinytest", "helper.R", package = "capybara"))

local({
  if (Sys.getenv("CAPYBARA_FULL_TESTING") != "yes") {
    return(NULL)
  }

  # vcov returns correct structure for feglm ----

  # IID  (no cluster part in formula)
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  fml <- ltrade ~ ldist + border | ctry1
  fit_iid <- felm(fml, data = ross2004_subset, vcov = "iid")
  vcov_iid <- vcov(fit_iid)

  # Heteroskedastic-robust (HC0)
  fit_hetero <- felm(fml, data = ross2004_subset, vcov = "hetero")
  vcov_hetero <- vcov(fit_hetero)

  # One-way
  fml2 <- update(Formula::as.Formula(fml), . ~ . | . | ctry2)

  fit_exp <- felm(fml2, data = ross2004_subset, vcov = "cluster")
  vcov_exp <- vcov(fit_exp)

  fit_imp <- felm(update(fml2, . ~ . | . | year), data = ross2004_subset, vcov = "cluster")
  vcov_imp <- vcov(fit_imp)

  # Dyadic-robust: Cameron-Miller (2014) sandwich with cross-dyad correlations
  fit_dyadic <- felm(update(fml2, . ~ . | . | ctry2 + year), data = ross2004_subset, vcov = "dyadic")
  vcov_dyadic <- vcov(fit_dyadic)

  # the determinants must be different
  expect_true(det(vcov_iid) != det(vcov_hetero))
  expect_true(det(vcov_iid) != det(vcov_exp))
  expect_true(det(vcov_iid) != det(vcov_imp))
  expect_true(det(vcov_iid) != det(vcov_dyadic))

  # R re-computation

  fit <- felm(fml, data = ross2004_subset, control = fit_control(keep_tx = TRUE, return_hessian = TRUE))
  vcov_hetero2 <- sandwich_vcov(fit, type = "hetero")
  vcov_exp2 <- sandwich_vcov(fit, cluster1 = ross2004_subset$ctry2, type = "clustered")
  vcov_imp2 <- sandwich_vcov(fit, cluster1 = ross2004_subset$year, type = "clustered")
  vcov_dyadic2 <- sandwich_vcov(fit, cluster1 = ross2004_subset$ctry2, cluster2 = ross2004_subset$year, type = "dyadic")

  expect_true(all.equal(vcov_hetero, vcov_hetero2))
  expect_true(all.equal(vcov_exp, vcov_exp2))
  expect_true(all.equal(vcov_imp, vcov_imp2))
  expect_true(all.equal(vcov_dyadic, vcov_dyadic2))
})
