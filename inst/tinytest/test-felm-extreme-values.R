# srr_stats (tests)
# {RE5.5} Ensures accuracy of prediction methods with unseen data subsets, maintaining expected patterns of error.
# these tests are a formality but an important one to check NA/Inf/NaN handling
# this is justified provided that I am not using base R's model.matrix

# NAs on the lhs ----

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  yotov2017_subset$trade[c(1, 3, 5)] <- NA

  m1 <- felm(trade ~ log_dist | exp_year, data = yotov2017_subset)
  m2 <- lm(trade ~ 0 + log_dist + as.factor(exp_year), data = yotov2017_subset)

  expect_equal(coef(m1), coef(m2)[1])
})

# NAs on the rhs ----

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  yotov2017_subset$exp_year[c(1, 3, 5)] <- NA

  m1 <- felm(trade ~ log_dist | exp_year, data = yotov2017_subset)
  m2 <- lm(trade ~ 0 + log_dist + as.factor(exp_year), data = yotov2017_subset)

  expect_equal(coef(m1), coef(m2)[1])
})

# 0+log on the rhs ----

local({
  yotov2017_subset <- yotov2017[yotov2017$year == 2006, ]
  yotov2017_subset <- yotov2017_subset[yotov2017_subset$trade > 0, ]
  yotov2017_subset$trade[c(1, 3, 5)] <- 0

  # we need a subset to avoid breaking lm!
  # otherwise
  # Error in lm.fit(x, y, offset = offset, singular.ok = singular.ok, ...) :
  # NA/NaN/Inf in 'y'
  yotov2017_subset_subset <- yotov2017_subset[-c(1, 3, 5), ]

  m1 <- felm(trade ~ log_dist | exp_year, data = yotov2017_subset)
  m2 <- lm(trade ~ 0 + log_dist + as.factor(exp_year), data = yotov2017_subset_subset)

  expect_equal(coef(m1), coef(m2)[1], tolerance = 1e-2)
})
