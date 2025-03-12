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
  trade_panel_2002 <- trade_panel[trade_panel$year == 2002, ]
  trade_panel_2002$trade_100 <- ifelse(trade_panel_2002$trade >= 100, 1, 0)
  trade_panel_2002$trade_200_100 <- as.factor(ifelse(trade_panel_2002$trade >=
    200, 1, ifelse(trade_panel_2002$trade >= 100, 0.5, 0)))
  trade_panel_2002$trade_1_minus1 <- ifelse(trade_panel_2002$trade >= 100, 1,
    -1
  )

  # no model

  expect_error(apes(), "specified")

  expect_error(
    apes(lm(trade ~ log_dist, data = trade_panel_2002)),
    "non-'feglm'"
  )

  # using APEs with Poisson

  expect_error(
    apes(fepoisson(trade ~ log_dist | rta, data = trade_panel_2002)),
    "binary choice"
  )

  # not using two-way fixed effects

  expect_error(
    apes(
      feglm(
        trade_100 ~ log_dist | rta + cntg + clny + lang,
        data = trade_panel_2002,
        family = binomial()
      ),
      panel_structure = "classic"
    ), "two-way"
  )

  # not using three-way fixed effects

  expect_error(
    apes(
      feglm(
        trade_100 ~ log_dist | rta + cntg + clny + lang,
        data = trade_panel_2002,
        family = binomial()
      ),
      panel_structure = "network"
    ), "three-way"
  )

  # wrong population size

  trade_panel_2002$tradebin <- ifelse(trade_panel_2002$trade > 100, 1L, 0L)

  expect_error(
    apes(
      feglm(
        tradebin ~ lang | year,
        data = trade_panel_2002,
        family = binomial()
      ),
      # n_pop = 4692
      n_pop = NA
    ), "missing value"
  )
})

test_that("error conditions in GLMs", {
  trade_panel_2002 <- trade_panel[trade_panel$year == 2002, ]
  trade_panel_2002$trade_100 <- ifelse(trade_panel_2002$trade >= 100, 1, 0)
  trade_panel_2002$trade_200_100 <- as.factor(ifelse(trade_panel_2002$trade >=
    200, 1, ifelse(trade_panel_2002$trade >= 100, 0.5, 0)))
  trade_panel_2002$trade_1_minus1 <- ifelse(trade_panel_2002$trade >= 100, 1,
    -1
  )

  # 0 rows in the data

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = trade_panel_2002[trade_panel_2002$year == 3000, ]
    ),
    "zero observations"
  )

  # incorrect deviance tolerance

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      control = list(dev_tol = -1.0)
    ),
    "greater than zero"
  )

  # bad number of iterations

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      control = list(iter_max = 0)
    ),
    "at least one"
  )

  # bad number of iterations

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      control = list(limit = 0)
    ),
    "at least one"
  )
})

test_that("error conditions in helpers", {
  trade_panel_2002 <- trade_panel[trade_panel$year == 2002, ]
  trade_panel_2002$trade_100 <- ifelse(trade_panel_2002$trade >= 100, 1, 0)
  trade_panel_2002$trade_200_100 <- as.factor(ifelse(trade_panel_2002$trade >=
    200, 1, ifelse(trade_panel_2002$trade >= 100, 0.5, 0)))
  trade_panel_2002$trade_1_minus1 <- ifelse(trade_panel_2002$trade >= 100, 1,
    -1
  )

  # no formula

  expect_error(
    feglm(data = trade_panel_2002), "'formula' has to be specified"
  )

  # incorrect formula

  expect_error(
    feglm(
      formula = "a ~ b",
      data = trade_panel_2002
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
      data = trade_panel_2002,
      control = c(1, 2)
    ),
    "'control' has to be a list"
  )

  # incorrect family

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      family = "poisson"
    ),
    "subscript out of bounds"
  )

  # we have the cluster estimator to do the same as quasi-Poisson

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      family = quasipoisson()
    ),
    "should be one of"
  )

  # fitting a negative binomial model with the GLM function

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      family = MASS::neg.bin(theta = 1)
    ),
    "use 'fenegbin' instead"
  )

  # not adding fixed effects

  expect_error(
    fepoisson(
      trade ~ log_dist,
      data = trade_panel_2002
    ),
    "'formula' incorrectly specified"
  )

  # incorrect data + link = bad response

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      family = binomial()
    ),
    "Model response must be within"
  )

  # incorrect data + link = bad response

  expect_error(
    feglm(
      trade_200_100 ~ log_dist | rta,
      data = trade_panel_2002,
      family = binomial()
    ),
    "response has to be binary"
  )

  # incorrect data + link = bad response

  expect_error(
    feglm(
      trade_1_minus1 ~ log_dist | rta,
      data = trade_panel_2002,
      family = Gamma()
    ),
    "response has to be positive"
  )

  # incorrect data + link = bad response

  expect_error(
    feglm(
      trade_1_minus1 ~ log_dist | rta,
      data = trade_panel_2002,
      family = inverse.gaussian()
    ),
    "response has to be positive"
  )

  # incorrect beta

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      beta_start = NA # not allowed
    ),
    "Invalid input type"
  )

  # incorrect eta

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      eta_start = rep(NA, nrow(trade_panel_2002))
    ),
    "Invalid input type"
  )

  # incorrect theta

  expect_error(
    fenegbin(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      init_theta = -1 # not allowed
    ),
    "has to be strictly positive"
  )

  # intentionally break the data with unusable weights

  trade_panel_2002$bad_weights <- NA

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      weights = "bad_weights"
    ),
    "Linear dependent terms detected"
  )
})
