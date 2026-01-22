# #' srr_stats (tests)
# #' @srrstats {G5.4} Tests for separation detection in GLMs
# #' @srrstats {G5.4a} Tests edge cases and typical separation scenarios
# #' @srrstats {RE4.6} Validates separation detection algorithm
# #' @noRd
# NULL

test_that("check_separation works as expected", {
    skip_on_cran()

    fit1 <- coef(fepoisson(
        y ~ x1 + x2 | i + j,
        data = ppmlhdfe$fe1
    ))

    fit2 <- coef(fepoisson(
        y ~ x1 + x2 | i + j,
        data = ppmlhdfe$fe1,
        control = list(check_separation = FALSE)
    ))

    expect_true(is.na(fit1[2]))
    expect_true(fit2[2] < 0)
})
