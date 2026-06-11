# srr_stats (tests)
# {G5.4} Tests for separation detection in GLMs
# {G5.4a} Tests edge cases and typical separation scenarios
# {RE4.6} Validates separation detection algorithm

source(system.file("tinytest", "helper.R", package = "capybara"))

local({
  if (Sys.getenv("CAPYBARA_FULL_TESTING") != "yes") {
    return(NULL)
  }

  # check_separation works as expected ----

  fit1 <- coef(fepoisson(
    y ~ x1 + x2 | i + j,
    data = correia2019$fe1
  ))

  fit2 <- coef(fepoisson(
    y ~ x1 + x2 | i + j,
    data = correia2019$fe1,
    control = list(check_separation = FALSE)
  ))

  expect_true(is.na(fit1[2]))
  expect_true(fit2[2] < 0)

  # fepoisson_asymmetric slopes change with/without separation check ----

  mod1 <- fepoisson_asymmetric(
    y ~ x1 + x2 + x3 + x4,
    data = correia2019$example1,
    control = fit_control(expectile = 0.75, return_fe = TRUE)
  )

  mod2 <- fepoisson_asymmetric(
    y ~ x1 + x2 + x3 + x4,
    data = correia2019$example1,
    control = fit_control(expectile = 0.75, return_fe = TRUE, check_separation = FALSE)
  )

  expect_equal(unname(coef(mod1)[3]), NA_real_)
  expect_true(unname(coef(mod2)[3]) > 0)
})
