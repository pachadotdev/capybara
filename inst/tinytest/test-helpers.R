# srr_stats (tests)
# {G5.4} Tests for helper functions
# {G5.4a} Tests edge cases and typical scenarios

source(system.file("tinytest", "helper.R", package = "capybara"))

local({
  fit <- structure(list(fixed_effects = list(
    exporter = c(A = 0, B = log(2)),
    importer = c(X = log(3), Y = 0)
  )), class = "feglm")
  data <- data.frame(
    exporter = factor(c("B", "A", "B", "new", NA), levels = c("new", "A", "B")),
    importer = c("Y", "X", "X", "Y", NA),
    row.names = c("r5", "r2", "r3", "r1", "r4")
  )
  before <- data
  result <- add_fe_to_df(data, fit)
  expect_identical(data, before)
  expect_identical(result[names(data)], data)
  expect_equal(result$exporter_fe, c(log(2), 0, log(2), NA, NA))
  expect_equal(result$importer_fe, c(0, log(3), log(3), 0, NA))

  result <- add_fe_to_df(
    data, fit,
    col_names = c(importer = "imp_fe", exporter = "exp_fe"),
    transform = exp
  )
  expect_equal(result$exp_fe, c(2, 1, 2, NA, NA))
  expect_equal(result$imp_fe, c(1, 3, 3, 1, NA))
  expect_identical(names(result), c(names(data), "exp_fe", "imp_fe"))
  result <- add_fe_to_df(result, fit, which = "importer", col_names = "imp_fe")
  expect_equal(result$imp_fe, c(0, log(3), log(3), 0, NA))
  expect_equal(result$exp_fe, c(2, 1, 2, NA, NA))
  expect_identical(nrow(add_fe_to_df(data[FALSE, ], fit)), 0L)
  expect_identical(add_fe_to_df(data, fit, which = character()), data)

  expect_error(add_fe_to_df(list(), fit), "must be a data.frame")
  expect_error(add_fe_to_df(data, list()), "not a felm or feglm")
  expect_error(
    add_fe_to_df(data, structure(list(), class = "felm")),
    "return_fe = TRUE"
  )
  expect_error(add_fe_to_df(data, fit, which = "missing"), "dimensions")
  expect_error(
    add_fe_to_df(data["exporter"], fit), "missing fixed-effect columns"
  )
  expect_error(add_fe_to_df(data, fit, col_names = c("fe", "fe")), "distinct")
  expect_error(
    add_fe_to_df(data, fit, col_names = c(exporter = "fe", other = "fe2")),
    "Names of"
  )
  expect_error(
    add_fe_to_df(data, fit, col_names = c("exporter", "imp_fe")),
    "cannot overwrite"
  )
  expect_error(add_fe_to_df(data, fit, transform = "exp"), "must be a function")
  expect_error(add_fe_to_df(data, fit, transform = sum), "same length")

  training <- data.frame(
    group = rep(c("a", "b"), each = 4L),
    x = rep(seq_len(4L), 2L),
    y = rep(seq_len(4L), 2L) + rep(c(0, 3), each = 4L)
  )
  fit <- felm(y ~ x | group, training)
  result <- add_fe_to_df(training[c(8L, 1L, 5L), ], fit)
  expect_equal(
    result$group_fe, unname(fixed_effects(fit, "group")[c("b", "a", "b")])
  )
})

local({
  if (Sys.getenv("CAPYBARA_FULL_TESTING") != "yes") {
    return(NULL)
  }

  # model handles collinearity detection ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]
  ross2004_subset$ldist2 <- ross2004_subset$ldist * 2 # Perfect collinearity

  mod <- felm(ltrade ~ ldist + ldist2 | ctry1, ross2004_subset)

  # Should still fit, dropping collinear variables
  expect_true(inherits(mod, "felm"))

  # offset works with formula specification ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  # add offset column based on total ltrade by exporter
  total_trade_by_exp <- aggregate(ltrade ~ ctry1, ross2004_subset, sum)
  names(total_trade_by_exp)[2] <- "total_trade"
  ross2004_subset <- merge(ross2004_subset, total_trade_by_exp, by = "ctry1")
  ross2004_subset$offset_var <- log(ross2004_subset$total_trade)

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset, offset = ~offset_var)

  expect_true(inherits(mod, "feglm"))
  expect_true("offset" %in% names(mod))

  # model handles different tolerance settings ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  ctrl1 <- fit_control(dev_tol = 1e-6, center_tol = 1e-6)
  mod1 <- felm(ltrade ~ ldist | ctry1, ross2004_subset, control = ctrl1)

  ctrl2 <- fit_control(dev_tol = 1e-10, center_tol = 1e-10)
  mod2 <- felm(ltrade ~ ldist | ctry1, ross2004_subset, control = ctrl2)

  # Both should converge but potentially to slightly different values
  expect_true(inherits(mod1, "felm"))
  expect_true(inherits(mod2, "felm"))

  # model handles different iteration limits ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  ctrl <- fit_control(iter_max = 100L, iter_center_max = 5000L)
  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset, control = ctrl)

  expect_true(inherits(mod, "felm"))

  # model handles character fixed effects ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]
  ross2004_subset$ctry1 <- as.character(ross2004_subset$ctry1)

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_true(inherits(mod, "feglm"))

  # model handles small sample sizes ----

  small_data <- ross2004[ross2004$year %in% c(1994, 1999), ]
  small_data <- do.call(rbind, lapply(split(small_data, small_data$year), head, 100))

  mod <- fepoisson(ltrade ~ ldist | year, small_data)

  expect_true(inherits(mod, "feglm"))

  # model returns correct number of observations ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_equal(as.numeric(mod$nobs["nobs"]), nrow(ross2004_subset))

  # model matrix operations work correctly ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- felm(ltrade ~ ldist + ltrade + border | ctry1, ross2004_subset)

  # Check dimensions
  expect_equal(length(coef(mod)), 3)
  expect_equal(nrow(vcov(mod)), 3)
  expect_equal(ncol(vcov(mod)), 3)
})
