#' srr_stats (tests)
#' @srrstats {RE2.1} Ensures that models throw meaningful error messages when input parameters or data are invalid.
#' @srrstats {RE5.1} Validates appropriate error handling for omitted arguments, such as missing formula or data.
#' @srrstats {RE5.2} Confirms that incorrect control settings result in appropriate error messages.
#' @srrstats {RE5.3} Verifies that the function stops execution when given unsupported model families or inappropriate responses.
#' @srrstats {RE5.4} Ensures that the model gracefully handles invalid starting values for beta, eta, or theta.
#' @srrstats {RE6.0} Implements robust testing for invalid combinations of fixed effects or missing parameters in APEs and GLMs.
#' @noRd
NULL

source(system.file("tinytest", "helper.R", package = "capybara"))

# error conditions in GLMs"
local({
  trade_short <- yotov2017[yotov2017$year == 2002, ]
  trade_short$trade_200 <- ifelse(trade_short$trade >= 100, 1, 0)
  trade_short$trade_200_100 <- as.factor(ifelse(
    trade_short$trade >= 200,
    1,
    ifelse(trade_short$trade >= 100, 0.5, 0)
  ))
  trade_short$trade_1_minus1 <- ifelse(trade_short$trade >= 100, 1, -1)

  # 0 rows in the data

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = trade_short[trade_short$year == 3000, ]
    ),
    "zero observations"
  )

  # incorrect deviance tolerance

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = trade_short,
      control = list(dev_tol = -1.0)
    ),
    "greater than zero"
  )

  # bad number of iterations

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = trade_short,
      control = list(iter_max = 0)
    ),
    "greater than zero"
  )

  # bad number of iterations

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = trade_short,
      control = list(iter_max = 0)
    ),
    "greater than zero"
  )
})

# error conditions in helpers"
local({
  trade_short <- yotov2017[yotov2017$year == 2002, ]
  trade_short$trade_200 <- ifelse(trade_short$trade >= 100, 1, 0)
  trade_short$trade_200_100 <- as.factor(ifelse(
    trade_short$trade >= 200,
    1,
    ifelse(trade_short$trade >= 100, 0.5, 0)
  ))
  trade_short$trade_1_minus1 <- ifelse(trade_short$trade >= 100, 1, -1)

  # no formula

  expect_error(feglm(data = trade_short), "'formula' has to be specified")

  # incorrect formula

  expect_error(
    feglm(
      formula = "a ~ b",
      data = trade_short
    ),
    "'formula' has to be of class 'formula'"
  )

  # null data

  expect_error(
    fepoisson(trade ~ log_dist | rta, data = NULL),
    "'data' must be specified"
  )

  # empty data

  expect_error(
    fepoisson(trade ~ log_dist | rta, data = list()),
    "'data' must be a data.frame"
  )

  # incorrect control

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = trade_short,
      control = c(1, 2)
    ),
    "'control' has to be a list"
  )

  # we have the cluster estimator to do the same as quasi-Poisson

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_short,
      family = quasipoisson()
    ),
    "should be one of"
  )

  # fitting a negative binomial model with the GLM function

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_short,
      family = MASS::neg.bin(theta = 1)
    ),
    "use 'fenegbin' instead"
  )

  # incorrect data + link = bad response

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_short,
      family = binomial()
    ),
    "Model response must be within"
  )

  # incorrect data + link = bad response

  expect_error(
    feglm(
      trade_200_100 ~ log_dist | rta,
      data = trade_short,
      family = binomial()
    ),
    "response has to be binary"
  )

  # incorrect data + link = bad response

  expect_error(
    feglm(
      trade_1_minus1 ~ log_dist | rta,
      data = trade_short,
      family = Gamma()
    ),
    "response has to be positive"
  )

  # incorrect data + link = bad response

  expect_error(
    feglm(
      trade_1_minus1 ~ log_dist | rta,
      data = trade_short,
      family = inverse.gaussian()
    ),
    "response has to be positive"
  )

  # incorrect beta

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_short,
      beta_start = NA # not allowed
    ),
    "Invalid input type"
  )

  # incorrect eta

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_short,
      eta_start = rep(NA, nrow(trade_short))
    ),
    "Invalid input type"
  )

  # incorrect theta

  expect_error(
    fenegbin(
      trade ~ log_dist | rta,
      data = trade_short,
      init_theta = -1 # not allowed
    ),
    "positive scalar"
  )

  # intentionally break the data with unusable weights

  trade_short$bad_weights <- NA

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_short,
      weights = "bad_weights"
    ),
    "Weights must be numeric"
  )
})

# ---- Additional error tests ----

# model errors on missing data"
local({
  expect_error(
    fepoisson(mpg ~ wt | cyl),
    "data"
  )
})

# model errors on invalid formula"
local({
  expect_error(
    fepoisson(~ wt | cyl, mtcars),
    "formula"
  )
})

# model errors on non-existent variables"
local({
  expect_error(
    fepoisson(mpg ~ nonexistent | cyl, mtcars),
    "undefined columns"
  )
})

# model errors on empty fixed effects"
local({
  skip_on_cran()

  # This should work - no FE is valid
  mod <- fepoisson(mpg ~ wt, mtcars)
  expect_true(inherits(mod, "feglm"))
})

# predict errors on missing newdata variables"
local({
  mod <- fepoisson(mpg ~ wt + hp | cyl, mtcars, control = fit_control(return_fe = TRUE))

  newdata <- data.frame(wt = c(2.5, 3.0)) # Missing hp and cyl

  expect_error(
    predict(mod, newdata = newdata),
    "undefined columns selected"
  )
})

# vcov works correctly"
local({
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)
  v <- vcov(mod)

  expect_true(is.matrix(v))
  expect_equal(dim(v), c(1, 1))
})

# summary works for all model types"
local({
  mod_felm <- felm(mpg ~ wt | cyl, mtcars)
  mod_feglm <- fepoisson(mpg ~ wt | cyl, mtcars)
  mod_fenegbin <- fenegbin(mpg ~ wt | cyl, mtcars)

  expect_true(inherits(summary(mod_felm), "summary.felm"))
  expect_true(inherits(summary(mod_feglm), "summary.feglm"))
  expect_true(inherits(summary(mod_fenegbin), "summary.feglm"))
})

# coef extraction works"
local({
  mod <- fepoisson(mpg ~ wt + hp | cyl, mtcars)
  cf <- coef(mod)

  expect_equal(length(cf), 2)
  expect_true(all(names(cf) %in% c("wt", "hp")))
})

# model handles zero counts in Poisson"
local({
  skip_on_cran()

  mtcars2 <- mtcars
  mtcars2$mpg[1:3] <- 0

  mod <- fepoisson(mpg ~ wt | cyl, mtcars2)

  expect_true(inherits(mod, "feglm"))
})

# model handles extreme values"
local({
  skip_on_cran()

  mtcars2 <- mtcars
  mtcars2$wt_large <- mtcars2$wt * 1000

  mod <- felm(mpg ~ wt_large | cyl, mtcars2)

  expect_true(inherits(mod, "felm"))
})
