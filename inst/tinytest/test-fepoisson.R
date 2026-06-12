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

local({
  if (Sys.getenv("CAPYBARA_FULL_TESTING") != "yes") {
    return(NULL)
  }

  # fepoisson is similar to base for K=1,2 ----

  # K = 1

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset, control = fit_control(return_fe = TRUE))

  mod_base <- glm(
    ltrade ~ ldist + as.factor(ctry1),
    ross2004_subset,
    family = quasipoisson(link = "log")
  )

  coef_dist_base <- coef(mod_base)[2]

  dist_variation <- unname(abs((coef(mod)[1] - coef_dist_base) / coef(mod)[1]))

  expect_equal(dist_variation, 0.0, tolerance = 1e-2)

  n <- unname(mod[["nobs"]]["nobs_full"])

  expect_equal(length(fitted(mod)), n)
  expect_equal(length(predict(mod)), n)
  expect_equal(length(coef(mod)), 1)

  smod <- summary(mod)

  expect_equal(length(coef(smod)[, 1]), 1)

  # K = 2

  mod <- fepoisson(ltrade ~ ldist | ctry1 + ctry2, ross2004_subset, control = fit_control(return_fe = TRUE))

  mod_base <- glm(
    ltrade ~ ldist + as.factor(ctry1) + as.factor(ctry2),
    ross2004_subset,
    family = quasipoisson(link = "log")
  )

  coef_dist_base <- coef(mod_base)[2]

  dist_variation <- abs((coef(mod)[1] - coef_dist_base) / coef(mod)[1])

  expect_true(dist_variation < 0.05)

  # fepoisson is similar to base for K=3 ----

  ross2004_subset <- ross2004[ross2004$year %in% c(1994, 1999), ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  mod <- fepoisson(ltrade ~ ldist | ctry1 + ctry2 + year, ross2004_subset, control = fit_control(return_fe = TRUE))

  mod_base <- glm(
    ltrade ~ ldist + as.factor(ctry1) + as.factor(ctry2) + as.factor(year),
    ross2004_subset,
    family = quasipoisson(link = "log")
  )

  coef_dist_base <- coef(mod_base)[2]

  dist_variation <- abs((coef(mod)[1] - coef_dist_base) / coef(mod)[1])

  expect_true(dist_variation < 0.05)

  expect_equal(mod[["fitted_values"]], mod_base[["fitted.values"]], tolerance = 1e-2)

  pred_mod <- predict(mod, type = "response")
  pred_mod_base <- predict(mod_base, type = "response")

  pred_mod_link <- predict(mod, type = "link")
  pred_mod_base_link <- predict(mod_base, type = "link")

  expect_equal(pred_mod, pred_mod_base, tolerance = 1e-2)
  expect_equal(pred_mod_link, pred_mod_base_link, tolerance = 1e-2)

  pred_mod <- predict(mod, type = "response", newdata = ross2004_subset[1:10, ])
  pred_mod_base <- predict(mod_base, type = "response", newdata = ross2004_subset[1:10, ])

  pred_mod_link <- predict(mod, type = "link", newdata = ross2004_subset[1:10, ])
  pred_mod_base_link <- predict(mod_base, type = "link", newdata = ross2004_subset[1:10, ])

  expect_equal(unname(pred_mod), unname(pred_mod_base), tolerance = 1e-2)
  expect_equal(unname(pred_mod_link), unname(pred_mod_base_link), tolerance = 1e-2)

  # fepoisson estimation is the same adding noise to the data ----

  set.seed(123)
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > quantile(ross2004_subset$ltrade, 0.75), ]

  d <- ross2004_subset[, c("ltrade", "ldist", "ctry1")]
  d$ldist2 <- d$ldist + pmax(rnorm(nrow(d)), 0) * .Machine$double.eps

  m1 <- fepoisson(ltrade ~ ldist | ctry1, d)
  m2 <- fepoisson(ltrade ~ ldist2 | ctry1, d)

  expect_equal(unname(coef(m1)), unname(coef(m2)))
  expect_equal(m1$fixed.effects, m2$fixed.effects)
})
