# These tests are a formality but an important one to check NA/Inf/NaN handling. This is justified provided that I am not using base R's model.matrix

# srr_stats (tests)
# {RE5.5} Ensures accuracy of prediction methods with unseen data subsets, maintaining expected patterns of error.

# NAs on the lhs ----

local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  ross2004_subset$ltrade[c(1, 3, 5)] <- NA

  m1 <- felm(ltrade ~ ldist | ctry1, data = ross2004_subset)
  m2 <- lm(ltrade ~ 0 + ldist + as.factor(ctry1), data = ross2004_subset)

  expect_equal(coef(m1), coef(m2)[1])
})

# NAs on the rhs ----

local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  ross2004_subset$ctry1[c(1, 3, 5)] <- NA

  m1 <- felm(ltrade ~ ldist | ctry1, data = ross2004_subset)
  m2 <- lm(ltrade ~ 0 + ldist + as.factor(ctry1), data = ross2004_subset)

  expect_equal(coef(m1), coef(m2)[1])
})

# 0+log on the rhs ----

local({
  ross2004_subset <- ross2004[ross2004$year == 1999, ]
  ross2004_subset <- ross2004_subset[ross2004_subset$ltrade > 0, ]
  ross2004_subset$ltrade[c(1, 3, 5)] <- 0

  # we need a subset to avoid breaking lm!
  # otherwise
  # Error in lm.fit(x, y, offset = offset, singular.ok = singular.ok, ...) :
  # NA/NaN/Inf in 'y'
  ross2004_subset_subset <- ross2004_subset[-c(1, 3, 5), ]

  m1 <- felm(ltrade ~ ldist | ctry1, data = ross2004_subset)
  m2 <- lm(ltrade ~ 0 + ldist + as.factor(ctry1), data = ross2004_subset_subset)

  expect_equal(coef(m1), coef(m2)[1], tolerance = 1e-2)
})
