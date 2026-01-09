#' srr_stats (tests)
#' @srrstats {G1.0} Implements unit testing for summary_table functionality.
#' @srrstats {G2.3} Tests various output formats and model combinations.
#' @srrstats {RE3.1} Verifies the correctness of formatted regression tables.
#' @noRd
NULL

test_that("summary_table works with single model", {
    m1 <- felm(mpg ~ wt | cyl, mtcars)

    result <- summary_table(m1)

    expect_s3_class(result, "summary_table")
    expect_type(result, "list")
    expect_true("content" %in% names(result))
})

test_that("summary_table works with multiple models", {
    m1 <- felm(mpg ~ wt | cyl, mtcars)
    m2 <- fepoisson(mpg ~ wt | cyl, mtcars)

    result <- summary_table(m1, m2)

    expect_s3_class(result, "summary_table")
    expect_type(result, "list")
})

test_that("summary_table works with custom model names", {
    m1 <- felm(mpg ~ wt | cyl, mtcars)
    m2 <- fepoisson(mpg ~ wt | cyl, mtcars)

    result <- summary_table(m1, m2, model_names = c("OLS", "Poisson"))

    expect_s3_class(result, "summary_table")
    expect_true(grepl("OLS", result$content))
    expect_true(grepl("Poisson", result$content))
})

test_that("summary_table works with latex output", {
    m1 <- felm(mpg ~ wt | cyl, mtcars)

    result <- summary_table(m1, latex = TRUE)

    expect_s3_class(result, "summary_table")
    expect_equal(result$type, "latex")
    expect_true(grepl("tabular", result$content))
})

test_that("summary_table works with latex caption and label", {
    m1 <- felm(mpg ~ wt | cyl, mtcars)

    result <- summary_table(
        m1,
        latex = TRUE,
        caption = "My Table",
        label = "tab:mytable"
    )

    expect_s3_class(result, "summary_table")
    expect_true(grepl("caption", result$content))
    expect_true(grepl("label", result$content))
})

test_that("summary_table works without stars", {
    m1 <- felm(mpg ~ wt | cyl, mtcars)

    result <- summary_table(m1, stars = FALSE)

    expect_s3_class(result, "summary_table")
    expect_false(grepl("\\*", result$content))
})

test_that("summary_table respects digit settings", {
    m1 <- felm(mpg ~ wt | cyl, mtcars)

    result <- summary_table(m1, coef_digits = 5, se_digits = 5)

    expect_s3_class(result, "summary_table")
    expect_type(result, "list")
})

test_that("summary_table errors on invalid input", {
    expect_error(summary_table(1L), "not a felm or feglm")
    expect_error(summary_table(lm(mpg ~ wt, mtcars)), "not a felm or feglm")
})

test_that("summary_table errors on mismatched model_names length", {
    m1 <- felm(mpg ~ wt | cyl, mtcars)
    m2 <- fepoisson(mpg ~ wt | cyl, mtcars)

    expect_error(
        summary_table(m1, m2, model_names = c("Only One")),
        "Length of model_names"
    )
})

test_that("summary_table works with models without fixed effects", {
    m1 <- felm(mpg ~ wt, mtcars)
    m2 <- fepoisson(mpg ~ wt, mtcars)

    result <- summary_table(m1, m2)

    expect_s3_class(result, "summary_table")
    expect_type(result, "list")
})

test_that("summary_table handles models with different variables", {
    m1 <- felm(mpg ~ wt | cyl, mtcars)
    m2 <- felm(mpg ~ wt + hp | cyl, mtcars)

    result <- summary_table(m1, m2)

    expect_s3_class(result, "summary_table")
    expect_true(grepl("hp", result$content))
})

test_that("summary_table works with feglm binomial", {
    m1 <- feglm(am ~ wt | cyl, mtcars, family = binomial())

    result <- summary_table(m1)

    expect_s3_class(result, "summary_table")
    expect_type(result, "list")
})
