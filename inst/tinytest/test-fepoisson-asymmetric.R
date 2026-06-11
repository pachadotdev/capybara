# srr_stats (tests)
# {RE3.1} Validates consistency between `fepoisson` and other established R models like `glm` with comparable families.
# {RE3.2} Compares coefficients produced by `fepoisson` with those from base R models to validate similarity.
# {RE4.3} Ensures stable estimates when adding negligible noise to the data.
# {RE5.1} Validates proper output generation for the model summary and printing methods.
# {RE7.3} Confirms that estimated coefficients are within a reasonable variation threshold compared to baseline models.
# {G5.11} The extended tests do not require additional downloads.
# {G5.11a} As for G5.11., the extended tests do not require additional downloads.
# {G5.12} The extended tests verify that the algorithm fitting time is robust to noise. This has to be tested with a larger dataset to see that time(clean) <= time(noisy).

source(system.file("tinytest", "helper.R", package = "capybara"))

# fepoisson_asymmetric slopes are smaller than fepoisson at 25% expectile
local({
  if (Sys.getenv("CAPYBARA_FULL_TESTING") != "yes") {
    return(NULL)
  }

  ross2004_subset <- ross2004[ross2004$year == 1999, ]

  mod1 <- fepoisson_asymmetric(
    ltrade ~ ldist | ctry1, ross2004_subset,
    control = fit_control(expectile = 0.25, return_fe = TRUE)
  )

  mod2 <- fepoisson(
    ltrade ~ ldist | ctry1, ross2004_subset,
    control = fit_control(return_fe = TRUE)
  )

  expect_true(coef(mod1) < coef(mod2))
})
