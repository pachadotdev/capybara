#' srr_stats (tests)
#' @srrstats {G1.0} Implements unit testing for summary_table functionality.
#' @srrstats {G2.3} Tests various output formats and model combinations.
#' @srrstats {RE3.1} Verifies the correctness of formatted regression tables.
#' @noRd
NULL

# summary_table works with single model
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  m1 <- felm(trade ~ log_dist | exp_year, yotov2017_subset)

  result <- summary_table(m1)

  expect_true(inherits(result, "summary_table"))
  expect_true(is.list(result))
  expect_true("content" %in% names(result))
})

# summary_table works with multiple models
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  m1 <- felm(trade ~ log_dist | exp_year, yotov2017_subset)
  m2 <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  result <- summary_table(m1, m2)

  expect_true(inherits(result, "summary_table"))
  expect_true(is.list(result))
})

# summary_table works with custom model names
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  m1 <- felm(trade ~ log_dist | exp_year, yotov2017_subset)
  m2 <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  result <- summary_table(m1, m2, model_names = c("OLS", "Poisson"))

  expect_true(inherits(result, "summary_table"))
  expect_true(grepl("OLS", result$content))
  expect_true(grepl("Poisson", result$content))
})

# summary_table works with latex output
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  m1 <- felm(trade ~ log_dist | exp_year, yotov2017_subset)

  result <- summary_table(m1, latex = TRUE)

  expect_true(inherits(result, "summary_table"))
  expect_equal(result$type, "latex")
  expect_true(grepl("tabular", result$content))
})

# summary_table works with latex caption and label
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  m1 <- felm(trade ~ log_dist | exp_year, yotov2017_subset)

  result <- summary_table(
    m1,
    latex = TRUE,
    caption = "My Table",
    label = "tab:mytable"
  )

  expect_true(inherits(result, "summary_table"))
  expect_true(grepl("caption", result$content))
  expect_true(grepl("label", result$content))
})

# summary_table works without stars
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  m1 <- felm(trade ~ log_dist | exp_year, yotov2017_subset)

  result <- summary_table(m1, stars = FALSE)

  expect_true(inherits(result, "summary_table"))
  expect_false(grepl("\\*", result$content))
})

# summary_table respects digit settings
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  m1 <- felm(trade ~ log_dist | exp_year, yotov2017_subset)

  result <- summary_table(m1, coef_digits = 5, se_digits = 5)

  expect_true(inherits(result, "summary_table"))
  expect_true(is.list(result))
})

# summary_table errors on invalid input
local({
  expect_error(summary_table(1L), "not a felm or feglm")
  expect_error(summary_table(lm(y ~ x, data.frame(x = 1:10, y = 1:10))), "not a felm or feglm")
})

# summary_table errors on mismatched model_names length
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  m1 <- felm(trade ~ log_dist | exp_year, yotov2017_subset)
  m2 <- fepoisson(trade ~ log_dist | exp_year, yotov2017_subset)

  expect_error(
    summary_table(m1, m2, model_names = c("Only One")),
    "Length of model_names"
  )
})

# summary_table works with models without fixed effects
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  m1 <- felm(trade ~ log_dist, yotov2017_subset)
  m2 <- fepoisson(trade ~ log_dist, yotov2017_subset)

  result <- summary_table(m1, m2)

  expect_true(inherits(result, "summary_table"))
  expect_true(is.list(result))
})

# summary_table handles models with different variables
local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  m1 <- felm(trade ~ log_dist | exp_year, yotov2017_subset)
  m2 <- felm(trade ~ log_dist + cntg | exp_year, yotov2017_subset)

  result <- summary_table(m1, m2)

  expect_true(inherits(result, "summary_table"))
  expect_true(grepl("cntg", result$content))
})
