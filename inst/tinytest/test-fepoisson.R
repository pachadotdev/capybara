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

# fepoisson is similar to base for K=1,2 ----

local({  
  # K = 1

  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  mod <- fepoisson(trade ~ log_dist | exp_year | imp_year, yotov2017_subset, control = fit_control(return_fe = TRUE))

  mod_base <- glm(
    trade ~ log_dist + as.factor(exp_year),
    yotov2017_subset,
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

  mod <- fepoisson(trade ~ log_dist | exp_year + imp_year, yotov2017_subset, control = fit_control(return_fe = TRUE))

  mod_base <- glm(
    trade ~ log_dist + as.factor(exp_year) + as.factor(imp_year),
    yotov2017_subset,
    family = quasipoisson(link = "log")
  )

  coef_dist_base <- coef(mod_base)[2]

  dist_variation <- abs((coef(mod)[1] - coef_dist_base) / coef(mod)[1])

  expect_true(dist_variation < 0.05)
})

# fepoisson is similar to base for K=3 ----

local({
  skip_on_cran()

  yotov2017_subset <- yotov2017[yotov2017$year %in% c(2002, 2006), ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]

  mod <- fepoisson(trade ~ log_dist | exp_year + imp_year + year, yotov2017_subset, control = fit_control(return_fe = TRUE))

  mod_base <- glm(
    trade ~ log_dist + as.factor(exp_year) + as.factor(imp_year) + as.factor(year),
    yotov2017_subset,
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

  pred_mod <- predict(mod, type = "response", newdata = yotov2017_subset[1:10, ])
  pred_mod_base <- predict(mod_base, type = "response", newdata = yotov2017_subset[1:10, ])

  pred_mod_link <- predict(mod, type = "link", newdata = yotov2017_subset[1:10, ])
  pred_mod_base_link <- predict(mod_base, type = "link", newdata = yotov2017_subset[1:10, ])

  expect_equal(unname(pred_mod), unname(pred_mod_base), tolerance = 1e-2)
  expect_equal(unname(pred_mod_link), unname(pred_mod_base_link), tolerance = 1e-2)
})

# fepoisson estimation is the same adding noise to the data ----

local({
  set.seed(123)
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  
  d <- yotov2017_subset[, c("trade", "log_dist", "exp_year")]
  d$log_dist2 <- d$log_dist + pmax(rnorm(nrow(d)), 0) * .Machine$double.eps

  m1 <- fepoisson(trade ~ log_dist | exp_year, d)
  m2 <- fepoisson(trade ~ log_dist2 | exp_year, d)

  expect_equal(unname(coef(m1)), unname(coef(m2)))
  expect_equal(m1$fixed.effects, m2$fixed.effects)
})
