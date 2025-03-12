#' srr_stats (tests)
#' @srrstats {RE2.1} Ensures that models throw meaningful error messages when input parameters or data are invalid.
#' @srrstats {RE3.1} Validates consistency between `felm` and base R `lm` in terms of coefficients, R-squared, and fitted values.
#' @srrstats {RE3.2} Compares model outputs against established benchmarks such as base R's `lm`.
#' @srrstats {RE5.1} Validates appropriate error handling for omitted arguments or missing data.
#' @srrstats {RE6.0} Implements robust testing for invalid or collinear regressors.
#' @srrstats {RE7.1} Validates that proportional regressors or collinear terms are detected and produce errors.
#' @srrstats {RE7.1a} Adding noise to the depending variable minimally affects the speed. I tested that explicitly.
#' @srrstats {RE7.2} Confirms that model computations remain consistent when small noise is added to data.
#' @srrstats {RE8.1} Ensures computational times remain consistent under similar model specifications.
#' @noRd
NULL

test_that("felm works", {
  m1 <- felm(mpg ~ wt | cyl, mtcars)
  m2 <- lm(mpg ~ wt + as.factor(cyl), mtcars)

  expect_equal(round(coef(m1), 2), round(coef(m2)[2], 2))

  n <- nrow(mtcars)
  expect_equal(length(fitted(m1)), n)
  expect_equal(length(predict(m1)), n)
  expect_equal(length(coef(m1)), 1)
  expect_equal(length(coef(summary(m1))), 4)

  m1 <- felm(mpg ~ wt + qsec | cyl, mtcars)
  m2 <- lm(mpg ~ wt + qsec + as.factor(cyl), mtcars)

  expect_equal(round(coef(m1), 2), round(coef(m2)[c(2, 3)], 2))

  m1 <- felm(mpg ~ wt + qsec | cyl + am, mtcars)
  m2 <- lm(mpg ~ wt + qsec + as.factor(cyl) + as.factor(am), mtcars)

  expect_equal(round(coef(m1), 2), round(coef(m2)[c(2, 3)], 2))

  s1 <- summary(m1)
  s2 <- summary(m2)

  expect_equal(s1$r.squared, s2$r.squared)
  expect_equal(s1$adj.r.squared, s2$adj.r.squared)

  mtcars2 <- mtcars
  mtcars2$wt[2] <- NA

  m1 <- felm(mpg ~ wt + qsec | cyl + am, mtcars2)
  m2 <- lm(mpg ~ wt + qsec + as.factor(cyl) + as.factor(am), mtcars2)

  expect_equal(round(coef(m1), 2), round(coef(m2)[c(2, 3)], 2))

  s1 <- summary(m1)
  s2 <- summary(m2)

  expect_equal(s1$r.squared, s2$r.squared)
  expect_equal(s1$adj.r.squared, s2$adj.r.squared, tolerance = 1e-2)

  m1 <- felm(mpg ~ wt + qsec | cyl + am | carb, mtcars)

  expect_equal(round(coef(m1), 2), round(coef(m2)[c(2, 3)], 2))
})

test_that("felm time is the minimally affected when adding noise to the data", {
  mtcars2 <- mtcars[, c("mpg", "wt", "cyl")]
  set.seed(200100)
  mtcars2$mpg <- mtcars2$mpg + rbinom(nrow(mtcars2), 1, 0.5) *
    .Machine$double.eps
  m1 <- felm(mpg ~ wt | cyl, mtcars)
  m2 <- felm(mpg ~ wt | cyl, mtcars2)
  expect_equal(coef(m1), coef(m2))
  expect_equal(fixed_effects(m1), fixed_effects(m2))

  t1 <- rep(NA, 10)
  t2 <- rep(NA, 10)
  for (i in 1:10) {
    a <- Sys.time()
    m1 <- felm(mpg ~ wt | cyl, mtcars)
    b <- Sys.time()
    t1[i] <- b - a

    a <- Sys.time()
    m2 <- felm(mpg ~ wt | cyl, mtcars2)
    b <- Sys.time()
    t2[i] <- b - a
  }
  expect_lte(abs(median(t1) - median(t2)), 0.05)
})

test_that("proportional regressors produce an error", {
  set.seed(200100)
  d <- data.frame(
    y = rnorm(100),
    x1 = rnorm(100),
    f = factor(sample(1:2, 1000, replace = TRUE))
  )
  d$x2 <- 2 * d$x1

  expect_error(felm(y ~ x1 + x2 | f, data = d), "dependent terms")
})
