#' srr_stats (tests)
#' @srrstats {G1.0} Implements unit testing for broom generics (tidy, glance, augment).
#' @srrstats {G2.3} Tests compatibility with broom package conventions.
#' @srrstats {RE3.1} Verifies the correctness of extracted model statistics.
#' @noRd
NULL

# ---- glance tests ----

test_that("glance.feglm returns correct structure", {
    mod <- fepoisson(mpg ~ wt | cyl, mtcars)

    result <- glance(mod)

    expect_s3_class(result, "data.frame")
    expect_true("deviance" %in% names(result))
    expect_true("null_deviance" %in% names(result))
    expect_true("nobs" %in% names(result))
})

test_that("glance.feglm works with binomial", {
    mod <- feglm(am ~ wt | cyl, mtcars, family = binomial())

    result <- glance(mod)

    expect_s3_class(result, "data.frame")
    expect_true(is.numeric(result$deviance))
})

test_that("glance.felm returns correct structure", {
    mod <- felm(mpg ~ wt | cyl, mtcars)

    result <- glance(mod)

    expect_s3_class(result, "data.frame")
    expect_true("r_squared" %in% names(result))
    expect_true("adj_r_squared" %in% names(result))
    expect_true("nobs" %in% names(result))
})

test_that("glance.felm works with multiple fixed effects", {
    mod <- felm(mpg ~ wt | cyl + am, mtcars)

    result <- glance(mod)

    expect_s3_class(result, "data.frame")
    expect_true(result$r_squared > 0 && result$r_squared < 1)
})

# ---- tidy tests ----

test_that("tidy.feglm returns correct structure", {
    mod <- fepoisson(mpg ~ wt | cyl, mtcars)

    result <- tidy(mod)

    expect_s3_class(result, "data.frame")
    expect_equal(
        names(result),
        c("estimate", "std.error", "statistic", "p.value")
    )
})

test_that("tidy.feglm works with conf_int", {
    mod <- fepoisson(mpg ~ wt | cyl, mtcars)

    result <- tidy(mod, conf_int = TRUE)

    expect_s3_class(result, "data.frame")
    expect_true("conf.low" %in% names(result))
    expect_true("conf.high" %in% names(result))
    expect_true(all(result$conf.low < result$estimate))
    expect_true(all(result$conf.high > result$estimate))
})

test_that("tidy.feglm respects conf_level", {
    mod <- fepoisson(mpg ~ wt | cyl, mtcars)

    result_95 <- tidy(mod, conf_int = TRUE, conf_level = 0.95)
    result_99 <- tidy(mod, conf_int = TRUE, conf_level = 0.99)

    # 99% CI should be wider than 95% CI
    width_95 <- result_95$conf.high - result_95$conf.low
    width_99 <- result_99$conf.high - result_99$conf.low

    expect_true(all(width_99 > width_95))
})

test_that("tidy.felm returns correct structure", {
    mod <- felm(mpg ~ wt | cyl, mtcars)

    result <- tidy(mod)

    expect_s3_class(result, "data.frame")
    expect_equal(
        names(result),
        c("estimate", "std.error", "statistic", "p.value")
    )
})

test_that("tidy.felm works with conf_int", {
    mod <- felm(mpg ~ wt | cyl, mtcars)

    result <- tidy(mod, conf_int = TRUE)

    expect_true("conf.low" %in% names(result))
    expect_true("conf.high" %in% names(result))
})

test_that("tidy works with multiple predictors", {
    mod <- felm(mpg ~ wt + hp + qsec | cyl, mtcars)

    result <- tidy(mod)

    expect_equal(nrow(result), 3)
})

# ---- augment tests ----

test_that("augment.feglm returns correct structure", {
    mod <- fepoisson(mpg ~ wt | cyl, mtcars)

    result <- augment(mod)

    expect_s3_class(result, "data.frame")
    expect_true(".fitted" %in% names(result))
    expect_true(".residuals" %in% names(result))
    expect_equal(nrow(result), nrow(mtcars))
})

test_that("augment.feglm preserves original columns", {
    mod <- fepoisson(mpg ~ wt | cyl, mtcars)

    result <- augment(mod)

    expect_true("mpg" %in% names(result))
    expect_true("wt" %in% names(result))
    expect_true("cyl" %in% names(result))
})

test_that("augment.felm returns correct structure", {
    mod <- felm(mpg ~ wt | cyl, mtcars)

    result <- augment(mod)

    expect_s3_class(result, "data.frame")
    expect_true(".fitted" %in% names(result))
    expect_true(".residuals" %in% names(result))
})

test_that("augment.felm fitted values are reasonable", {
    mod <- felm(mpg ~ wt | cyl, mtcars)

    result <- augment(mod)

    # Fitted values should be in a reasonable range
    expect_true(all(result$.fitted > 0))
    expect_true(all(result$.fitted < 50))
})

test_that("augment works with binomial model", {
    mod <- feglm(am ~ wt | cyl, mtcars, family = binomial())

    result <- augment(mod)

    expect_s3_class(result, "data.frame")
    expect_true(".fitted" %in% names(result))
    # Fitted values for binomial should be probabilities
    expect_true(all(result$.fitted >= 0 & result$.fitted <= 1))
})

# ---- fitted tests ----

test_that("fitted.feglm returns correct values", {
    mod <- fepoisson(mpg ~ wt | cyl, mtcars)

    result <- fitted(mod)

    expect_equal(length(result), nrow(mtcars))
    expect_true(all(result > 0))
})

test_that("fitted.felm returns correct values", {
    mod <- felm(mpg ~ wt | cyl, mtcars)

    result <- fitted(mod)

    expect_equal(length(result), nrow(mtcars))
})
