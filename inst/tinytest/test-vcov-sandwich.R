# srr_stats (tests)
# {G5.2} Validates that covariance matrices align with theoretical expectations under different estimation methods.
# {RE3.3} Ensures consistency of `vcov` results for sandwich and outer-product estimators.
# {RE4.3} Confirms robustness of covariance matrix calculations under varied model specifications.
# {RE6.0} Ensures that covariance estimations respond correctly to model clustering and input variations.

source(system.file("tinytest", "helper.R", package = "capybara"))

# vcov returns correct structure for feglm
local({
  skip_on_cran()

  # IID  (no cluster part in formula)
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  fml <- trade ~ log_dist + cntg | exp_year
  fit_iid <- felm(fml, data = yotov2017_subset, vcov = "iid")
  vcov_iid <- vcov(fit_iid)

  # Heteroskedastic-robust (HC0)
  fit_hetero <- felm(fml, data = yotov2017_subset, vcov = "hetero")
  vcov_hetero <- vcov(fit_hetero)

  # One-way
  fml2 <- update(Formula::as.Formula(fml), . ~ . | . | imp_year)

  fit_exp <- felm(fml2, data = yotov2017_subset, vcov = "cluster")
  vcov_exp <- vcov(fit_exp)

  fit_imp <- felm(update(fml2, . ~ . | . | year), data = yotov2017_subset, vcov = "cluster")
  vcov_imp <- vcov(fit_imp)

  # Dyadic-robust: Cameron-Miller (2014) sandwich with cross-dyad correlations
  fit_dyadic <- felm(update(fml2, . ~ . | . | imp_year + year), data = yotov2017_subset, vcov = "dyadic")
  vcov_dyadic <- vcov(fit_dyadic)

  # the determinants must be different
  expect_true(det(vcov_iid) != det(vcov_hetero))
  expect_true(det(vcov_iid) != det(vcov_exp))
  expect_true(det(vcov_iid) != det(vcov_imp))
  expect_true(det(vcov_iid) != det(vcov_dyadic))

  # R re-computation

  fit <- felm(fml, data = yotov2017_subset, control = fit_control(keep_tx = TRUE, return_hessian = TRUE))
  vcov_hetero2 <- sandwich_vcov(fit, type = "hetero")
  vcov_exp2 <- sandwich_vcov(fit, cluster1 = yotov2017_subset$imp_year, type = "clustered")
  vcov_imp2 <- sandwich_vcov(fit, cluster1 = yotov2017_subset$year, type = "clustered")
  vcov_dyadic2 <- sandwich_vcov(fit, cluster1 = yotov2017_subset$imp_year, cluster2 = yotov2017_subset$year, type = "dyadic")

  expect_true(all.equal(vcov_hetero, vcov_hetero2))
  expect_true(all.equal(vcov_exp, vcov_exp2))
  expect_true(all.equal(vcov_imp, vcov_imp2))
  expect_true(all.equal(vcov_dyadic, vcov_dyadic2))
})
