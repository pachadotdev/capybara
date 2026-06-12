# srr_stats (tests)
# {RE2.1} Ensures that models throw meaningful error messages when input parameters or data are invalid.
# {RE5.1} Validates appropriate error handling for omitted arguments, such as missing formula or data.
# {RE5.2} Confirms that incorrect control settings result in appropriate error messages.
# {RE5.3} Verifies that the function stops execution when given unsupported model families or inappropriate responses.
# {RE5.4} Ensures that the model gracefully handles invalid starting values for beta, eta, or theta.
# {RE6.0} Implements robust testing for invalid combinations of fixed effects or missing parameters in APEs and GLMs.

source(system.file("tinytest", "helper.R", package = "capybara"))

local({
  if (Sys.getenv("CAPYBARA_FULL_TESTING") != "yes") {
    return(NULL)
  }

  # 0 rows in the data ----

  expect_error(
    fepoisson(
      ltrade ~ ldist | ctry1,
      data = ross2004[ross2004$year == 3000, ]
    ),
    "zero observations"
  )

  # incorrect deviance tolerance ---

  expect_error(
    fepoisson(
      ltrade ~ ldist | ctry1,
      data = ross2004,
      control = list(dev_tol = -1.0)
    ),
    "greater than zero"
  )

  # bad number of iterations ----

  expect_error(
    fepoisson(
      ltrade ~ ldist | ctry1,
      data = ross2004,
      control = list(iter_max = 0)
    ),
    "greater than zero"
  )

  # bad number of iterations ----

  expect_error(
    fepoisson(
      ltrade ~ ldist | ctry1,
      data = ross2004,
      control = list(iter_max = 0)
    ),
    "greater than zero"
  )

  # no formula ----

  expect_error(feglm(data = ross2004), "'formula' has to be specified")

  # incorrect formula ----

  expect_error(
    feglm(
      formula = "a ~ b",
      data = ross2004
    ),
    "'formula' has to be of class 'formula'"
  )

  # null data ----

  expect_error(
    fepoisson(ltrade ~ ldist | ctry1, data = NULL),
    "'data' must be specified"
  )

  # empty data ----

  expect_error(
    fepoisson(ltrade ~ ldist | ctry1, data = list()),
    "'data' must be a data.frame"
  )

  # incorrect control ----

  expect_error(
    fepoisson(
      ltrade ~ ldist | ctry1,
      data = ross2004,
      control = c(1, 2)
    ),
    "'control' has to be a list"
  )

  # we have the cluster estimator to do the same as quasi-Poisson ----

  expect_error(
    feglm(
      ltrade ~ ldist | ctry1,
      data = ross2004,
      family = quasipoisson()
    ),
    "should be one of"
  )

  # fitting a negative binomial model with the GLM function ----

  expect_error(
    feglm(
      ltrade ~ ldist | ctry1,
      data = ross2004,
      family = MASS::neg.bin(theta = 1)
    ),
    "use 'fenegbin' instead"
  )

  # incorrect beta ----

  expect_error(
    feglm(
      ltrade ~ ldist | ctry1,
      data = ross2004,
      beta_start = NA # not allowed
    ),
    "Invalid input type"
  )

  # incorrect eta ----

  expect_error(
    feglm(
      ltrade ~ ldist | ctry1,
      data = ross2004,
      eta_start = rep(NA, nrow(ross2004))
    ),
    "Invalid input type"
  )

  # incorrect theta ----

  expect_error(
    fenegbin(
      ltrade ~ ldist | ctry1,
      data = ross2004,
      init_theta = -1 # not allowed
    ),
    "positive scalar"
  )

  # intentionally break the data with unusable weights ----

  ross2004$bad_weights <- NA

  expect_error(
    feglm(
      ltrade ~ ldist | ctry1,
      data = ross2004,
      weights = "bad_weights"
    ),
    "Weights must be numeric"
  )

  # model errors on missing data ----

  expect_error(
    fepoisson(ltrade ~ ldist | ctry1),
    "data"
  )

  # model errors on invalid formula

  expect_error(
    fepoisson(~ ldist | ctry1, ross2004),
    "formula"
  )

  # model errors on non-existent variables ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  expect_error(
    fepoisson(ltrade ~ nonexistent | ctry1, ross2004_subset),
    "undefined columns"
  )

  # predict errors on missing newdata variables ----

  mod <- fepoisson(ltrade ~ ldist + border | ctry1, ross2004_subset, control = fit_control(return_fe = TRUE))

  newdata <- data.frame(ldist = c(7, 8)) # Missing border and ctry1

  expect_error(
    predict(mod, newdata = newdata),
    "undefined columns selected"
  )

  # vcov works correctly ----

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)
  v <- vcov(mod)

  expect_true(is.matrix(v))
  expect_equal(dim(v), c(1, 1))

  # summary works for all model types ----

  mod_felm <- felm(ltrade ~ ldist | ctry1, ross2004_subset)
  mod_feglm <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)
  mod_fenegbin <- fenegbin(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_true(inherits(summary(mod_felm), "summary.felm"))
  expect_true(inherits(summary(mod_feglm), "summary.feglm"))
  expect_true(inherits(summary(mod_fenegbin), "summary.feglm"))

  # coef extraction works ----

  mod <- fepoisson(ltrade ~ ldist + border | ctry1, ross2004_subset)
  cf <- coef(mod)

  expect_equal(length(cf), 2)
  expect_true(all(names(cf) %in% c("ldist", "border")))

  # model handles zero counts in Poisson ----

  ross2004_subset$ltrade[1:3] <- 0

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_true(inherits(mod, "feglm"))

  ross2004_subset$ldist[1:3] <- Inf

  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_true(inherits(mod, "felm"))
})
