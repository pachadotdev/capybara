# srr_stats (tests)
# {G5.4} Tests for helper functions
# {G5.4a} Tests edge cases and typical scenarios

source(system.file("tinytest", "helper.R", package = "capybara"))

# model handles collinearity detection ----

local({
  skip_on_cran()

  # Create data with collinear variables
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  ross2004_subset$ldist2 <- ross2004_subset$ldist * 2 # Perfect collinearity

  mod <- felm(ltrade ~ ldist + ldist2 | ctry1, ross2004_subset)

  # Should still fit, dropping collinear variables
  expect_true(inherits(mod, "felm"))
})

# offset works with formula specification ----

local({
  skip_on_cran()

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]

  # add offset column based on total ltrade by exporter
  total_trade_by_exp <- aggregate(ltrade ~ ctry1, ross2004_subset, sum)
  names(total_trade_by_exp)[2] <- "total_trade"
  ross2004_subset <- merge(ross2004_subset, total_trade_by_exp, by = "ctry1")
  ross2004_subset$offset_var <- log(ross2004_subset$total_trade)

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset, offset = ~offset_var)

  expect_true(inherits(mod, "feglm"))
  expect_true("offset" %in% names(mod))
})

# model handles different tolerance settings ----

local({
  skip_on_cran()

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  
  ctrl1 <- fit_control(dev_tol = 1e-6, center_tol = 1e-6)
  mod1 <- felm(ltrade ~ ldist | ctry1, ross2004_subset, control = ctrl1)

  ctrl2 <- fit_control(dev_tol = 1e-10, center_tol = 1e-10)
  mod2 <- felm(ltrade ~ ldist | ctry1, ross2004_subset, control = ctrl2)

  # Both should converge but potentially to slightly different values
  expect_true(inherits(mod1, "felm"))
  expect_true(inherits(mod2, "felm"))
})

# model handles different iteration limits ----

local({
  skip_on_cran()

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  
  ctrl <- fit_control(iter_max = 100L, iter_center_max = 5000L)
  mod <- felm(ltrade ~ ldist | ctry1, ross2004_subset, control = ctrl)

  expect_true(inherits(mod, "felm"))
})

# model handles character fixed effects ----

local({
  skip_on_cran()

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  ross2004_subset$ctry1 <- as.character(ross2004_subset$ctry1)

  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_true(inherits(mod, "feglm"))
})

# model handles small sample sizes ----

local({
  skip_on_cran()

  small_data <- ross2004[ross2004$year %in% c(1994, 1999), ]
  small_data <- do.call(rbind, lapply(split(small_data, small_data$year), head, 100))

  mod <- fepoisson(ltrade ~ ldist | year, small_data)

  expect_true(inherits(mod, "feglm"))
})

# model returns correct number of observations ----

local({
  skip_on_cran()

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  
  mod <- fepoisson(ltrade ~ ldist | ctry1, ross2004_subset)

  expect_equal(as.numeric(mod$nobs["nobs"]), nrow(ross2004_subset))
})

# model matrix operations work correctly ----
local({
  skip_on_cran()

  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  
  mod <- felm(ltrade ~ ldist + ltrade + border | ctry1, ross2004_subset)

  # Check dimensions
  expect_equal(length(coef(mod)), 3)
  expect_equal(nrow(vcov(mod)), 3)
  expect_equal(ncol(vcov(mod)), 3)
})
