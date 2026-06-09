#' srr_stats (tests)
#' @srrstats {G5.4} Tests for helper functions
#' @srrstats {G5.4a} Tests edge cases and typical scenarios
#' @noRd
NULL

source(system.file("tinytest", "helper.R", package = "capybara"))

# ---- feglm_helpers tests ----

# model fitting works with different families
local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  # Poisson
  mod_pois <- feglm(trade ~ log_dist | exp_year, yotov2017_subset, family = poisson())
  expect_true(inherits(mod_pois, "feglm"))

  # Gaussian
  mod_gauss <- feglm(trade ~ log_dist | exp_year, yotov2017_subset, family = gaussian())
  expect_true(inherits(mod_gauss, "feglm"))
})

# model works without keep_tx option
local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  ctrl <- fit_control(keep_tx = FALSE)
  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset, control = ctrl)

  expect_true(inherits(mod, "felm"))
})

# model handles collinearity detection
local({
  skip_on_cran()

  # Create data with collinear variables
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  yotov2017_subset$log_dist2 <- yotov2017_subset$log_dist * 2 # Perfect collinearity

  mod <- felm(trade ~ log_dist + log_dist2 | exp_year, yotov2017_subset)

  # Should still fit, dropping collinear variables
  expect_true(inherits(mod, "felm"))
})

# weighted regression works
local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  yotov2017_subset$w <- runif(nrow(yotov2017_subset), 0.5, 1.5)

  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset, weights = ~w)

  expect_true(inherits(mod, "felm"))
  expect_true(!is.null(mod$weights))
})

# ---- Offset tests ----

# offset works with formula specification
local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset, offset = ~ log(dist))

  expect_true(inherits(mod, "feglm"))
  expect_true("offset" %in% names(mod))
})

# offset affects fitted values
local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod_no_offset <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)
  mod_offset <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset, offset = ~ log(dist))

  # Fitted values should be different
  expect_false(isTRUE(all.equal(fitted(mod_no_offset), fitted(mod_offset))))
})

# model works with different numbers of fixed effects
local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  # Single FE
  mod1 <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset, control = fit_control(return_fe = TRUE))
  expect_equal(length(mod1$fixed_effects), 1)

  # Multiple FEs
  mod2 <- fepoisson(trade ~ log_dist | exp_year + imp_year, yotov2017_subset, control = fit_control(return_fe = TRUE))
  expect_equal(length(mod2$fixed_effects), 2)

  # Three FEs
  mod3 <- fepoisson(trade ~ log_dist | exp_year + imp_year + year, yotov2017_subset, control = fit_control(return_fe = TRUE))
  expect_equal(length(mod3$fixed_effects), 3)
})

# model handles different tolerance settings
local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  ctrl1 <- fit_control(dev_tol = 1e-6, center_tol = 1e-6)
  mod1 <- felm(trade ~ log_dist | exp_year, yotov2017_subset, control = ctrl1)

  ctrl2 <- fit_control(dev_tol = 1e-10, center_tol = 1e-10)
  mod2 <- felm(trade ~ log_dist | exp_year, yotov2017_subset, control = ctrl2)

  # Both should converge but potentially to slightly different values
  expect_true(inherits(mod1, "felm"))
  expect_true(inherits(mod2, "felm"))
})

# model handles different iteration limits
local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  ctrl <- fit_control(iter_max = 100L, iter_center_max = 5000L)
  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset, control = ctrl)

  expect_true(inherits(mod, "felm"))
})

# ---- Data transformation tests ----

# model handles factor variables correctly
local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  yotov2017_subset$exp_year <- factor(yotov2017_subset$exp_year)
  
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset, control = fit_control(return_fe = TRUE))

  expect_true(inherits(mod, "feglm"))
  expect_equal(length(mod$fixed_effects), 2)
})

# model handles character fixed effects
local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  yotov2017_subset$exp_year <- as.character(yotov2017_subset$exp_year)

  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  expect_true(inherits(mod, "feglm"))
})

# ---- Edge cases ----

# model handles small sample sizes
local({
  skip_on_cran()

  small_data <- yotov2017[yotov2017$year %in% c(2002, 2006), ]
  small_data <- do.call(rbind, lapply(split(small_data, small_data$year), head, 100))

  mod <- fepoisson(trade ~ log_dist | year, small_data)

  expect_true(inherits(mod, "feglm"))
})

# model handles many fixed effect levels
local({
  skip_on_cran()

  data("yotov2017", package = "capybara")

  # This has many levels in exp_year and imp_iso
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017)

  expect_true(inherits(mod, "feglm"))
})

# model returns correct number of observations
local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  expect_equal(as.numeric(mod$nobs["nobs"]), nrow(yotov2017_subset))
})

# model matrix operations work correctly
local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  mod <- felm(trade ~ log_dist + trade + cntg | exp_year, yotov2017_subset)

  # Check dimensions
  expect_equal(length(coef(mod)), 3)
  expect_equal(nrow(vcov(mod)), 3)
  expect_equal(ncol(vcov(mod)), 3)
})
