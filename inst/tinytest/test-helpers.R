#' srr_stats (tests)
#' @srrstats {G5.4} Tests for helper functions
#' @srrstats {G5.4a} Tests edge cases and typical scenarios
#' @noRd
NULL

source(system.file("tinytest", "helper.R", package = "capybara"))

# ---- feglm_helpers tests ----

# model fitting works with different families"
local({
  skip_on_cran()

  # Poisson
  mod_pois <- feglm(mpg ~ wt | cyl, mtcars, family = poisson())
  expect_true(inherits(mod_pois, "feglm"))

  # Binomial
  mod_binom <- feglm(am ~ wt | cyl, mtcars, family = binomial())
  expect_true(inherits(mod_binom, "feglm"))

  # Gaussian
  mod_gauss <- feglm(mpg ~ wt | cyl, mtcars, family = gaussian())
  expect_true(inherits(mod_gauss, "feglm"))
})

# model handles different link functions"
local({
  skip_on_cran()

  # Poisson with different links
  mod1 <- feglm(mpg ~ wt | cyl, mtcars, family = poisson(link = "log"))
  expect_true(inherits(mod1, "feglm"))

  # Binomial with logit link (only supported link for binomial)
  mod2 <- feglm(am ~ wt | cyl, mtcars, family = binomial(link = "logit"))
  expect_true(inherits(mod2, "feglm"))
})

# model works without keep_tx option"
local({
  skip_on_cran()

  ctrl <- fit_control(keep_tx = FALSE)
  mod <- felm(mpg ~ wt | cyl, mtcars, control = ctrl)

  expect_true(inherits(mod, "felm"))
})

# model handles collinearity detection"
local({
  skip_on_cran()

  # Create data with collinear variables
  mtcars2 <- mtcars
  mtcars2$wt2 <- mtcars2$wt * 2 # Perfect collinearity

  mod <- felm(mpg ~ wt + wt2 | cyl, mtcars2)

  # Should still fit, dropping collinear variables
  expect_true(inherits(mod, "felm"))
})

# weighted regression works"
local({
  skip_on_cran()

  mtcars2 <- mtcars
  mtcars2$w <- runif(nrow(mtcars2), 0.5, 1.5)

  mod <- felm(mpg ~ wt | cyl, mtcars2, weights = ~w)

  expect_true(inherits(mod, "felm"))
  expect_true(!is.null(mod$weights))
})

# ---- Offset tests ----

# offset works with formula specification"
local({
  skip_on_cran()

  mod <- fepoisson(mpg ~ wt | cyl, mtcars, offset = ~ log(hp))

  expect_true(inherits(mod, "feglm"))
  expect_true("offset" %in% names(mod))
})

# offset affects fitted values"
local({
  skip_on_cran()

  mod_no_offset <- fepoisson(mpg ~ wt | cyl, mtcars)
  mod_offset <- fepoisson(mpg ~ wt | cyl, mtcars, offset = ~ log(hp))

  # Fitted values should be different
  expect_false(isTRUE(all.equal(fitted(mod_no_offset), fitted(mod_offset))))
})

# model works with different numbers of fixed effects"
local({
  skip_on_cran()

  # Single FE
  mod1 <- fepoisson(mpg ~ wt | cyl, mtcars, control = fit_control(return_fe = TRUE))
  expect_equal(length(mod1$fixed_effects), 1)

  # Multiple FEs
  mod2 <- fepoisson(mpg ~ wt | cyl + am, mtcars, control = fit_control(return_fe = TRUE))
  expect_equal(length(mod2$fixed_effects), 2)

  # Three FEs
  mod3 <- fepoisson(mpg ~ wt | cyl + am + gear, mtcars, control = fit_control(return_fe = TRUE))
  expect_equal(length(mod3$fixed_effects), 3)
})

# model handles different tolerance settings"
local({
  skip_on_cran()

  ctrl1 <- fit_control(dev_tol = 1e-6, center_tol = 1e-6)
  mod1 <- felm(mpg ~ wt | cyl, mtcars, control = ctrl1)

  ctrl2 <- fit_control(dev_tol = 1e-10, center_tol = 1e-10)
  mod2 <- felm(mpg ~ wt | cyl, mtcars, control = ctrl2)

  # Both should converge but potentially to slightly different values
  expect_true(inherits(mod1, "felm"))
  expect_true(inherits(mod2, "felm"))
})

# model handles different iteration limits"
local({
  skip_on_cran()

  ctrl <- fit_control(iter_max = 100L, iter_center_max = 5000L)
  mod <- felm(mpg ~ wt | cyl, mtcars, control = ctrl)

  expect_true(inherits(mod, "felm"))
})

# ---- Data transformation tests ----

# model handles factor variables correctly"
local({
  skip_on_cran()

  mtcars2 <- mtcars
  mtcars2$cyl <- factor(mtcars2$cyl)
  mtcars2$am <- factor(mtcars2$am)

  mod <- fepoisson(mpg ~ wt | cyl + am, mtcars2, control = fit_control(return_fe = TRUE))

  expect_true(inherits(mod, "feglm"))
  expect_equal(length(mod$fixed_effects), 2)
})

# model handles character fixed effects"
local({
  skip_on_cran()

  mtcars2 <- mtcars
  mtcars2$cyl_char <- as.character(mtcars2$cyl)

  mod <- fepoisson(mpg ~ wt | cyl_char, mtcars2)

  expect_true(inherits(mod, "feglm"))
})

# model works with interactions in predictors"
local({
  skip_on_cran()

  mod <- felm(mpg ~ wt * hp | cyl, mtcars)

  expect_true(inherits(mod, "felm"))
  expect_true(length(coef(mod)) >= 2)
})

# ---- Edge cases ----

# model handles small sample sizes"
local({
  skip_on_cran()

  small_data <- mtcars[1:10, ]
  mod <- fepoisson(mpg ~ wt | cyl, small_data)

  expect_true(inherits(mod, "feglm"))
})

# model handles many fixed effect levels"
local({
  skip_on_cran()

  data("yotov2017", package = "capybara")

  # This has many levels in exp_year and imp_iso
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017)

  expect_true(inherits(mod, "feglm"))
})

# model returns correct number of observations"
local({
  skip_on_cran()

  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  expect_equal(as.numeric(mod$nobs["nobs"]), nrow(mtcars))
})

# model matrix operations work correctly"
local({
  skip_on_cran()

  mod <- felm(mpg ~ wt + hp + disp | cyl, mtcars)

  # Check dimensions
  expect_equal(length(coef(mod)), 3)
  expect_equal(nrow(vcov(mod)), 3)
  expect_equal(ncol(vcov(mod)), 3)
})
