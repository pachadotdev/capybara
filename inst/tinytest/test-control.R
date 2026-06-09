#' srr_stats (tests)
#' @srrstats {G5.4} Tests for fit_control parameters
#' @srrstats {G2.0} Tests parameter validation
#' @noRd
NULL

source(system.file("tinytest", "helper.R", package = "capybara"))

# fit_control validates positive tolerance parameters
local({
  expect_error(
    fit_control(dev_tol = -0.01),
    "All tolerance parameters should be greater than zero"
  )

  expect_error(
    fit_control(center_tol = 0),
    "All tolerance parameters should be greater than zero"
  )

  expect_error(
    fit_control(alpha_tol = -1e-8),
    "All tolerance parameters should be greater than zero"
  )
})

# fit_control validates iteration parameters
local({
  expect_error(
    fit_control(iter_max = 0L),
    "All iteration parameters should be greater than or equal to one"
  )

  expect_error(
    fit_control(iter_center_max = -5L),
    "All iteration parameters should be greater than or equal to one"
  )
})

# fit_control validates logical parameters
local({
  expect_error(
    fit_control(return_fe = NA),
    "All logical parameters should be TRUE or FALSE"
  )

  expect_error(
    fit_control(keep_tx = NA),
    "All logical parameters should be TRUE or FALSE"
  )
})

# fit_control validates step_halving_memory
local({
  expect_error(
    fit_control(step_halving_memory = 0),
    "step_halving_memory should be between 0 and 1"
  )

  expect_error(
    fit_control(step_halving_memory = 1),
    "step_halving_memory should be between 0 and 1"
  )

  expect_error(
    fit_control(step_halving_memory = 1.5),
    "step_halving_memory should be between 0 and 1"
  )
})

# fit_control validates max_step_halving
local({
  expect_error(
    fit_control(max_step_halving = -1L),
    "max_step_halving should be greater than or equal to zero"
  )
})

# fit_control validates start_inner_tol
local({
  expect_error(
    fit_control(start_inner_tol = 0),
    "start_inner_tol should be greater than zero"
  )

  expect_error(
    fit_control(start_inner_tol = -1e-6),
    "start_inner_tol should be greater than zero"
  )
})

# fit_control validates centering
local({
  expect_error(
    fit_control(centering = "invalid_option"),
    "should be one of"
  )
})

# fit_control returns correct structure
local({
  ctrl <- fit_control()

  expect_true(is.list(ctrl))
  expect_true("dev_tol" %in% names(ctrl))
  expect_true("center_tol" %in% names(ctrl))
  expect_true("iter_max" %in% names(ctrl))
  expect_true("keep_tx" %in% names(ctrl))
})

# fit_control accepts valid custom parameters
local({
  ctrl <- fit_control(
    dev_tol = 1e-10,
    center_tol = 1e-9,
    iter_max = 50L,
    keep_tx = TRUE,
    return_fe = FALSE
  )

  expect_equal(ctrl$dev_tol, 1e-10)
  expect_equal(ctrl$center_tol, 1e-9)
  expect_equal(ctrl$iter_max, 50L)
  expect_true(ctrl$keep_tx)
  expect_false(ctrl$return_fe)
})

# fit_control coerces integers correctly
local({
  ctrl <- fit_control(
    iter_max = 100, # Not explicitly integer
    iter_center_max = 5000
  )

  expect_true(is.integer(ctrl$iter_max))
  expect_true(is.integer(ctrl$iter_center_max))
})

# fit_control has sensible defaults
local({
  ctrl <- fit_control()

  expect_true(ctrl$dev_tol > 0)
  expect_true(ctrl$center_tol > 0)
  expect_true(ctrl$iter_max >= 1)
  expect_true(is.logical(ctrl$return_fe))
  expect_true(is.logical(ctrl$keep_tx))
})

# fit_control works with models
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  ctrl <- list(dev_tol = 1e-10)

  mod <- felm(trade ~ log_dist | exp_year, yotov2017_subset, control = ctrl)

  expect_true(inherits(mod, "felm"))
})

# different control settings affect convergence
local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  # Tight tolerance
  ctrl_tight <- list(dev_tol = 1e-12, center_tol = 1e-12)
  mod_tight <- felm(trade ~ log_dist | exp_year, yotov2017_subset, control = ctrl_tight)

  # Loose tolerance
  ctrl_loose <- list(dev_tol = 1e-4, center_tol = 1e-4)
  mod_loose <- felm(trade ~ log_dist | exp_year, yotov2017_subset, control = ctrl_loose)

  # Both should converge
  expect_true(inherits(mod_tight, "felm"))
  expect_true(inherits(mod_loose, "felm"))

  # Coefficients should be similar but might differ slightly
  expect_equal(coef(mod_tight), coef(mod_loose), tolerance = 1e-3)
})

# init_theta parameter works for fenegbin
local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  ctrl <- list(init_theta = 0.5)

  mod <- fenegbin(trade ~ log_dist | exp_year, yotov2017_subset, control = ctrl)

  expect_true(inherits(mod, "feglm"))
  expect_true(mod$theta > 0)
})
