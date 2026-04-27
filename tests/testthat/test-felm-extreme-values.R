#' srr_stats (tests)
#' @srrstats {RE5.5} Ensures accuracy of prediction methods with unseen data subsets, maintaining expected patterns of error.
#' @noRd
NULL

# these tests are a formality but an important one to check NA/Inf/NaN handling
# this is justified provided that I am not using base R's model.matrix

test_that("NAs on the lhs", {
  mtcars2 <- mtcars
  mtcars2$mpg[c(1, 3, 5)] <- NA

  m1 <- felm(mpg ~ hp | cyl, data = mtcars2)
  m2 <- lm(mpg ~ 0 + hp + as.factor(cyl), data = mtcars2)

  expect_equal(coef(m1), coef(m2)[1])
})

test_that("NAs on the rhs", {
  mtcars2 <- mtcars
  mtcars2$cyl[c(1, 3, 5)] <- NA

  m1 <- felm(mpg ~ hp | cyl, data = mtcars2)
  m2 <- lm(mpg ~ 0 + hp + as.factor(cyl), data = mtcars2)

  expect_equal(coef(m1), coef(m2)[1])
})

test_that("0+log on the lhs", {
  mtcars2 <- mtcars
  mtcars2$mpg[c(1, 3, 5)] <- 0
  mtcars2$log_mpg <- log(mtcars2$mpg)
  mtcars2$log_hp <- log(mtcars2$hp)

  # we need a subset to avoid breaking lm!
  # otherwise
  # Error in lm.fit(x, y, offset = offset, singular.ok = singular.ok, ...) :
  # NA/NaN/Inf in 'y'
  mtcars2_subset <- mtcars2[-c(1, 3, 5), ]

  m1 <- felm(log_mpg ~ log_hp | cyl, data = mtcars2)
  m2 <- lm(log_mpg ~ 0 + log_hp + as.factor(cyl), data = mtcars2_subset)

  expect_equal(coef(m1), coef(m2)[1])
})

test_that("0+log on the rhs", {
  mtcars2 <- mtcars
  mtcars2$hp[c(1, 3, 5)] <- 0
  mtcars2$log_mpg <- log(mtcars2$mpg)
  mtcars2$log_hp <- log(mtcars2$hp)

  # we need a subset to avoid breaking lm!
  # otherwise
  # Error in lm.fit(x, y, offset = offset, singular.ok = singular.ok, ...) :
  # NA/NaN/Inf in 'y'
  mtcars2_subset <- mtcars2[-c(1, 3, 5), ]

  m1 <- felm(log_mpg ~ log_hp | cyl, data = mtcars2)
  m2 <- lm(log_mpg ~ 0 + log_hp + as.factor(cyl), data = mtcars2_subset)

  expect_equal(coef(m1), coef(m2)[1])
})
