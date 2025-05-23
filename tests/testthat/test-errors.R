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
  trade_short$trade_200_100 <- as.factor(ifelse(trade_short$trade >=
    200, 1, ifelse(trade_short$trade >= 200, 0.5, 0)))
  trade_short$trade_1_minus1 <- ifelse(trade_short$trade >= 200, 1,
    -1
  )

  # no model

  expect_error(apes(), "specified")

  expect_error(
    apes(lm(trade ~ log_dist, data = trade_short)),
    "non-'feglm'"
  )

  # using APEs with Poisson

  expect_error(
    apes(fepoisson(trade ~ log_dist | rta, data = trade_short)),
    "binary choice"
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
    ), "two-way"
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
    ), "three-way"
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
    ), "missing value"
  )
})

test_that("error conditions in GLMs", {
  trade_short <- trade_panel[trade_panel$year == 2002, ]
  trade_short$trade_200 <- ifelse(trade_short$trade >= 100, 1, 0)
  trade_short$trade_200_100 <- as.factor(ifelse(trade_short$trade >=
    200, 1, ifelse(trade_short$trade >= 100, 0.5, 0)))
  trade_short$trade_1_minus1 <- ifelse(trade_short$trade >= 100, 1,
    -1
  )

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
      control = list(limit = 0)
    ),
    "greater than zero"
  )
})

test_that("error conditions in helpers", {
  trade_short <- trade_panel[trade_panel$year == 2002, ]
  trade_short$trade_200 <- ifelse(trade_short$trade >= 100, 1, 0)
  trade_short$trade_200_100 <- as.factor(ifelse(trade_short$trade >=
    200, 1, ifelse(trade_short$trade >= 100, 0.5, 0)))
  trade_short$trade_1_minus1 <- ifelse(trade_short$trade >= 100, 1,
    -1
  )

  # no formula

  expect_error(
    feglm(data = trade_short), "'formula' has to be specified"
  )

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
    fepoisson(
      trade ~ log_dist | rta,
      data = NULL
    ),
    "'data' must be specified"
  )

  # empty data

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = list()
    ),
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
    "has to be strictly positive"
  )

  # intentionally break the data with unusable weights

  trade_short$bad_weights <- NA

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_short,
      weights = "bad_weights"
    ),
    "Linear dependent terms detected"
  )
})
