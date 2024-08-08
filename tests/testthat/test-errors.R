test_that("error conditions", {
  trade_panel_2002 <- trade_panel[trade_panel$year == 2002, ]
  trade_panel_2002$trade_100 <- ifelse(trade_panel_2002$trade >= 100, 1, 0)
  trade_panel_2002$trade_200_100 <- as.factor(ifelse(trade_panel_2002$trade >= 200, 1,
    ifelse(trade_panel_2002$trade >= 100, 0.5, 0)
  ))
  trade_panel_2002$trade_1_minus1 <- ifelse(trade_panel_2002$trade >= 100, 1, -1)


  # APEs ----

  # TODO: test n.pop argument and the rest of apes()

  expect_error(apes(), "specified")

  expect_error(
    apes(lm(trade ~ log_dist, data = trade_panel_2002)),
    "non-'feglm'"
  )

  expect_error(
    apes(fepoisson(trade ~ log_dist | rta, data = trade_panel_2002)),
    "binary choice"
  )

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

  # GLMs ----

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = trade_panel_2002[trade_panel_2002$year == 3000, ]
    ),
    "zero observations"
  )

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      control = list(dev_tol = -1.0)
    ),
    "greater than zero"
  )

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      control = list(iter_max = 0)
    ),
    "at least one"
  )

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      control = list(limit = 0)
    ),
    "at least one"
  )

  # Helpers ----

  # TODO:
  # weights
  # linear dependence
  # init.theta
  # beta.start
  # eta.start

  expect_error(
    feglm(data = trade_panel_2002), "'formula' has to be specified"
  )

  expect_error(
    feglm(formula = "a ~ b", data = trade_panel_2002), "'formula' has to be of class 'formula'"
  )

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = NULL
    ),
    "'data' has to be specified"
  )

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = list()
    ),
    "'data' has to be of class data.frame"
  )

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = list()
    ),
    "'data' has to be of class data.frame"
  )

  expect_error(
    fepoisson(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      control = c(1, 2)
    ),
    "'control' has to be a list"
  )

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      family = "poisson"
    ),
    "'family' has to be of class family"
  )

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      family = quasipoisson()
    ),
    "Quasi-variants of 'family' are not supported"
  )

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      family = MASS::neg.bin(theta = 1)
    ),
    "use 'fenegbin' instead"
  )

  expect_error(
    fepoisson(
      trade ~ log_dist,
      data = trade_panel_2002
    ),
    "'formula' incorrectly specified"
  )

  expect_error(
    feglm(
      trade ~ log_dist | rta,
      data = trade_panel_2002,
      family = binomial()
    ),
    "response has to be within the unit interval"
  )

  expect_error(
    feglm(
      trade_200_100 ~ log_dist | rta,
      data = trade_panel_2002,
      family = binomial()
    ),
    "response has to be binary"
  )

  expect_error(
    feglm(
      trade_1_minus1 ~ log_dist | rta,
      data = trade_panel_2002,
      family = Gamma()
    ),
    "response has to be strictly positive"
  )

  expect_error(
    feglm(
      trade_1_minus1 ~ log_dist | rta,
      data = trade_panel_2002,
      family = inverse.gaussian()
    ),
    "response has to be strictly positive"
  )
})
