# srr_stats (tests)
# {G1.0} Implements unit testing for summary_table functionality.
# {G2.3} Tests various output formats and model combinations.
# {RE3.1} Verifies the correctness of formatted regression tables.

local({
  if (Sys.getenv("CAPYBARA_FULL_TESTING") != "yes") {
    return(NULL)
  }

  # summary_table works with single model ----

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  m1 <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  result <- summary_table(m1)

  expect_true(inherits(result, "summary_table"))
  expect_true(is.list(result))
  expect_true("content" %in% names(result))

  # summary_table works with multiple models ----

  m1 <- felm(ltrade ~ ldist | ctry1, ross2004_subset)
  m2 <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  result <- summary_table(m1, m2)

  expect_true(inherits(result, "summary_table"))
  expect_true(is.list(result))

  # summary_table works with custom model names ----

  m1 <- felm(ltrade ~ ldist | ctry1, ross2004_subset)
  m2 <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  result <- summary_table(m1, m2, model_names = c("OLS", "Poisson"))

  expect_true(inherits(result, "summary_table"))
  expect_true(grepl("OLS", result$content))
  expect_true(grepl("Poisson", result$content))

  # summary_table works with latex output ----

  m1 <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  result <- summary_table(m1, latex = TRUE)

  expect_true(inherits(result, "summary_table"))
  expect_equal(result$type, "latex")
  expect_true(grepl("tabular", result$content))

  # summary_table works with latex caption and label ----

  m1 <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  result <- summary_table(
    m1,
    latex = TRUE,
    caption = "My Table",
    label = "tab:mytable"
  )

  expect_true(inherits(result, "summary_table"))
  expect_true(grepl("caption", result$content))
  expect_true(grepl("label", result$content))

  # summary_table works without stars ----

  m1 <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  result <- summary_table(m1, stars = FALSE)

  expect_true(inherits(result, "summary_table"))
  expect_false(grepl("\\*", result$content))

  # summary_table respects digit settings ----

  m1 <- felm(ltrade ~ ldist | ctry1, ross2004_subset)

  result <- summary_table(m1, coef_digits = 5, se_digits = 5)

  expect_true(inherits(result, "summary_table"))
  expect_true(is.list(result))

  # summary_table errors on invalid input ----

  expect_error(summary_table(1L), "not a felm or feglm")
  expect_error(summary_table(lm(y ~ x, data.frame(x = 1:10, y = 1:10))), "not a felm or feglm")

  # summary_table errors on mismatched model_names length ----

  m1 <- felm(ltrade ~ ldist | ctry1, ross2004_subset)
  m2 <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_error(
    summary_table(m1, m2, model_names = c("Only One")),
    "Length of model_names"
  )

  # summary_table works with models without fixed effects ----

  m1 <- felm(ltrade ~ ldist, ross2004_subset)
  m2 <- fepoisson(ltrade ~ ldist, ross2004_subset)

  result <- summary_table(m1, m2)

  expect_true(inherits(result, "summary_table"))
  expect_true(is.list(result))

  # summary_table handles models with different variables ----

  m1 <- felm(ltrade ~ ldist | ctry1, ross2004_subset)
  m2 <- felm(ltrade ~ ldist + border | ctry1, ross2004_subset)

  result <- summary_table(m1, m2)

  expect_true(inherits(result, "summary_table"))
  expect_true(grepl("border", result$content))
})
