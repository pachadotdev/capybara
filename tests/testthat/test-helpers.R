#' srr_stats (tests)
#' @srrstats {G5.4} Tests for helper functions
#' @srrstats {G5.4a} Tests edge cases and typical scenarios
#' @noRd
NULL

# ---- feglm_helpers tests ----

test_that("model fitting works with different families", {
  skip_on_cran()

  # Poisson
  mod_pois <- feglm(mpg ~ wt | cyl, mtcars, family = poisson())
  expect_s3_class(mod_pois, "feglm")

  # Binomial
  mod_binom <- feglm(am ~ wt | cyl, mtcars, family = binomial())
  expect_s3_class(mod_binom, "feglm")

  # Gaussian
  mod_gauss <- feglm(mpg ~ wt | cyl, mtcars, family = gaussian())
  expect_s3_class(mod_gauss, "feglm")
})

test_that("model handles different link functions", {
  skip_on_cran()

  # Poisson with different links
  mod1 <- feglm(mpg ~ wt | cyl, mtcars, family = poisson(link = "log"))
  expect_s3_class(mod1, "feglm")

  # Binomial with logit link (only supported link for binomial)
  mod2 <- feglm(am ~ wt | cyl, mtcars, family = binomial(link = "logit"))
  expect_s3_class(mod2, "feglm")
})

test_that("model works without keep_tx option", {
  skip_on_cran()

  ctrl <- fit_control(keep_tx = FALSE)
  mod <- felm(mpg ~ wt | cyl, mtcars, control = ctrl)

  expect_s3_class(mod, "felm")
})

test_that("model handles collinearity detection", {
  skip_on_cran()

  # Create data with collinear variables
  mtcars2 <- mtcars
  mtcars2$wt2 <- mtcars2$wt * 2 # Perfect collinearity

  mod <- felm(mpg ~ wt + wt2 | cyl, mtcars2)

  # Should still fit, dropping collinear variables
  expect_s3_class(mod, "felm")
})

test_that("weighted regression works", {
  skip_on_cran()

  mtcars2 <- mtcars
  mtcars2$w <- runif(nrow(mtcars2), 0.5, 1.5)

  mod <- felm(mpg ~ wt | cyl, mtcars2, weights = ~w)

  expect_s3_class(mod, "felm")
  expect_true(!is.null(mod$weights))
})

# ---- Offset tests ----

test_that("offset works with formula specification", {
  skip_on_cran()

  mod <- fepoisson(mpg ~ wt | cyl, mtcars, offset = ~ log(hp))

  expect_s3_class(mod, "feglm")
  expect_true("offset" %in% names(mod))
})

test_that("offset affects fitted values", {
  skip_on_cran()

  mod_no_offset <- fepoisson(mpg ~ wt | cyl, mtcars)
  mod_offset <- fepoisson(mpg ~ wt | cyl, mtcars, offset = ~ log(hp))

  # Fitted values should be different
  expect_false(isTRUE(all.equal(fitted(mod_no_offset), fitted(mod_offset))))
})

test_that("model works with different numbers of fixed effects", {
  skip_on_cran()

  # Single FE
  mod1 <- fepoisson(mpg ~ wt | cyl, mtcars)
  expect_equal(length(mod1$fixed_effects), 1)

  # Multiple FEs
  mod2 <- fepoisson(mpg ~ wt | cyl + am, mtcars)
  expect_equal(length(mod2$fixed_effects), 2)

  # Three FEs
  mod3 <- fepoisson(mpg ~ wt | cyl + am + gear, mtcars)
  expect_equal(length(mod3$fixed_effects), 3)
})

test_that("model handles different tolerance settings", {
  skip_on_cran()

  ctrl1 <- fit_control(dev_tol = 1e-6, center_tol = 1e-6)
  mod1 <- felm(mpg ~ wt | cyl, mtcars, control = ctrl1)

  ctrl2 <- fit_control(dev_tol = 1e-10, center_tol = 1e-10)
  mod2 <- felm(mpg ~ wt | cyl, mtcars, control = ctrl2)

  # Both should converge but potentially to slightly different values
  expect_s3_class(mod1, "felm")
  expect_s3_class(mod2, "felm")
})

test_that("model handles different iteration limits", {
  skip_on_cran()

  ctrl <- fit_control(iter_max = 100L, iter_center_max = 5000L)
  mod <- felm(mpg ~ wt | cyl, mtcars, control = ctrl)

  expect_s3_class(mod, "felm")
})

# ---- Data transformation tests ----

test_that("model handles factor variables correctly", {
  skip_on_cran()

  mtcars2 <- mtcars
  mtcars2$cyl <- factor(mtcars2$cyl)
  mtcars2$am <- factor(mtcars2$am)

  mod <- fepoisson(mpg ~ wt | cyl + am, mtcars2)

  expect_s3_class(mod, "feglm")
  expect_equal(length(mod$fixed_effects), 2)
})

test_that("model handles character fixed effects", {
  skip_on_cran()

  mtcars2 <- mtcars
  mtcars2$cyl_char <- as.character(mtcars2$cyl)

  mod <- fepoisson(mpg ~ wt | cyl_char, mtcars2)

  expect_s3_class(mod, "feglm")
})

test_that("model works with interactions in predictors", {
  skip_on_cran()

  mod <- felm(mpg ~ wt * hp | cyl, mtcars)

  expect_s3_class(mod, "felm")
  expect_true(length(coef(mod)) >= 2)
})

test_that("model handles polynomial terms", {
  skip_on_cran()

  mod <- felm(mpg ~ poly(wt, 2) | cyl, mtcars)

  expect_s3_class(mod, "felm")
})

test_that("model handles I() transformations", {
  skip_on_cran()

  mod <- felm(mpg ~ I(wt^2) + I(log(hp)) | cyl, mtcars)

  expect_s3_class(mod, "felm")
})

# ---- Edge cases ----

test_that("model handles small sample sizes", {
  skip_on_cran()

  small_data <- mtcars[1:10, ]
  mod <- fepoisson(mpg ~ wt | cyl, small_data)

  expect_s3_class(mod, "feglm")
})

test_that("model handles many fixed effect levels", {
  skip_on_cran()

  data("trade_panel", package = "capybara")

  # This has many levels in exp_year and imp_iso
  mod <- fepoisson(trade ~ log_dist | exp_year, trade_panel)

  expect_s3_class(mod, "feglm")
})

test_that("model returns correct number of observations", {
  skip_on_cran()

  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  expect_equal(as.numeric(mod$nobs["nobs"]), nrow(mtcars))
})

test_that("model matrix operations work correctly", {
  skip_on_cran()

  mod <- felm(mpg ~ wt + hp + disp | cyl, mtcars)

  # Check dimensions
  expect_equal(length(coef(mod)), 3)
  expect_equal(nrow(vcov(mod)), 3)
  expect_equal(ncol(vcov(mod)), 3)
})
