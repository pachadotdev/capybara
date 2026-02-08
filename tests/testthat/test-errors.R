#' srr_stats (tests)
#' @srrstats {RE2.1} Ensures that models throw meaningful error messages when input parameters or data are invalid.
#' @srrstats {RE5.1} Validates appropriate error handling for omitted arguments, such as missing formula or data.
#' @srrstats {RE5.2} Confirms that incorrect control settings result in appropriate error messages.
#' @srrstats {RE5.3} Verifies that the function stops execution when given unsupported model families or inappropriate responses.
#' @srrstats {RE5.4} Ensures that the model gracefully handles invalid starting values for beta, eta, or theta.
#' @srrstats {RE6.0} Implements robust testing for invalid combinations of fixed effects or missing parameters in APEs and GLMs.
#' @noRd
NULL

test_that("error conditions in APEs", {
  skip_on_cran()

  trade_short <- trade_panel[trade_panel$exp_year == "CAN1994", ]
  trade_short <- trade_short[trade_short$trade > 100, ]
  trade_short$trade_200 <- ifelse(trade_short$trade >= 200, 1, 0)
  trade_short$trade_200_100 <- as.factor(ifelse(
    trade_short$trade >= 200,
    1,
    ifelse(trade_short$trade >= 200, 0.5, 0)
  ))
  trade_short$trade_1_minus1 <- ifelse(trade_short$trade >= 200, 1, -1)

  # no model

  expect_error(apes(), "specified")

  expect_error(
    apes(lm(trade ~ log_dist, data = trade_short)),
    "non-'feglm'"
  )

  # not using two-way fixed effects

  expect_error(
    apes(
      feglm(
        trade_200 ~ log_dist | rta + cntg + clny + lang,
        data = trade_short,
        family = binomial()
      ),
      panel_structure = "classic"
    ),
    "two-way"
  )

  # not using three-way fixed effects

  expect_error(
    apes(
      feglm(
        trade_200 ~ log_dist | rta + cntg + clny + lang,
        data = trade_short,
        family = binomial()
      ),
      panel_structure = "network"
    ),
    "three-way"
  )

  # wrong population size

  expect_error(
    apes(
      feglm(
        trade_200 ~ lang | year,
        data = trade_short,
        family = binomial()
      ),
      # n_pop = 4692
      n_pop = NA
    ),
    "missing value"
  )
})

test_that("error conditions in GLMs", {
  trade_short <- trade_panel[trade_panel$year == 2002, ]
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

test_that("error conditions in helpers", {
  trade_short <- trade_panel[trade_panel$year == 2002, ]
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
    "'data' has zero observations"
  )

  # empty data

  expect_error(
    fepoisson(trade ~ log_dist | rta, data = list()),
    "'data' has zero observations"
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

  # incorrect family

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_short,
      family = "poisson"
    ),
    "subscript out of bounds"
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

test_that("model errors on missing data", {
  expect_error(
    fepoisson(mpg ~ wt | cyl),
    "data"
  )
})

test_that("model errors on invalid formula", {
  expect_error(
    fepoisson(~ wt | cyl, mtcars),
    "formula"
  )
})

test_that("model errors on non-existent variables", {
  expect_error(
    fepoisson(mpg ~ nonexistent | cyl, mtcars),
    "column not found"
  )
})

test_that("model errors on empty fixed effects", {
  skip_on_cran()

  # This should work - no FE is valid
  mod <- fepoisson(mpg ~ wt, mtcars)
  expect_s3_class(mod, "feglm")
})

test_that("predict errors on missing newdata variables", {
  mod <- fepoisson(mpg ~ wt + hp | cyl, mtcars)

  newdata <- data.frame(wt = c(2.5, 3.0)) # Missing hp and cyl

  expect_error(
    predict(mod, newdata = newdata),
    "columns not found"
  )
})

test_that("vcov works correctly", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)
  v <- vcov(mod)

  expect_true(is.matrix(v))
  expect_equal(dim(v), c(1, 1))
})

test_that("summary works for all model types", {
  mod_felm <- felm(mpg ~ wt | cyl, mtcars)
  mod_feglm <- fepoisson(mpg ~ wt | cyl, mtcars)
  mod_fenegbin <- fenegbin(mpg ~ wt | cyl, mtcars)

  expect_s3_class(summary(mod_felm), "summary.felm")
  expect_s3_class(summary(mod_feglm), "summary.feglm")
  expect_s3_class(summary(mod_fenegbin), "summary.feglm")
})

test_that("coef extraction works", {
  mod <- fepoisson(mpg ~ wt + hp | cyl, mtcars)
  cf <- coef(mod)

  expect_equal(length(cf), 2)
  expect_named(cf, c("wt", "hp"))
})

test_that("model handles zero counts in Poisson", {
  skip_on_cran()

  mtcars2 <- mtcars
  mtcars2$mpg[1:3] <- 0

  mod <- fepoisson(mpg ~ wt | cyl, mtcars2)

  expect_s3_class(mod, "feglm")
})

test_that("model handles extreme values", {
  skip_on_cran()

  mtcars2 <- mtcars
  mtcars2$wt_large <- mtcars2$wt * 1000

  mod <- felm(mpg ~ wt_large | cyl, mtcars2)

  expect_s3_class(mod, "felm")
})

test_that("print methods work", {
  mod_felm <- felm(mpg ~ wt | cyl, mtcars)
  mod_feglm <- fepoisson(mpg ~ wt | cyl, mtcars)

  expect_output(print(mod_felm))
  expect_output(print(mod_feglm))
  expect_output(print(summary(mod_felm)))
  expect_output(print(summary(mod_feglm)))
})
