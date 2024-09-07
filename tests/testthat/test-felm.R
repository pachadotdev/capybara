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

  # m1r2 <- s1$r.squared
  # m1r2a <- 1 - (1 - m1r2) * (s1$nobs["nobs"] - 1) / (s1$nobs["nobs"] - length(coef(m1)) -
  #   sum(vapply(m1[["nms_fe"]], length, integer(1))) + 1)

  # m2r2 <- s2$r.squared
  # m2r2a <- 1 - (1 - m2r2) * ((length(m2$residuals) - 1) / m2$df.residual)

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
  expect_equal(s1$adj.r.squared, s2$adj.r.squared)

  m1 <- felm(mpg ~ wt + qsec | cyl + am | carb, mtcars)

  expect_equal(round(coef(m1), 2), round(coef(m2)[c(2, 3)], 2))
})

test_that("felm works with perfect relationships", {
  set.seed(200100)
  d <- data.frame(
    y = rnorm(100),
    f = factor(sample(1:2, 1000, replace = TRUE))
  )
  d$x <- 2 * d$y

  fit <- felm(y ~ x | f, data = d)
  s1 <- summary(fit)
  expect_equal(s1$r.squared, 1)
  expect_equal(s1$adj.r.squared, 1)
  expect_equal(s1$cm[4], 0)
})

test_that("felm time is the same adding noise to the data", {
  mtcars2 <- mtcars[, c("mpg", "wt", "cyl")]
  set.seed(200100)
  mtcars2$mpg <- mtcars2$mpg + rbinom(nrow(mtcars2), 1, 0.5) * .Machine$double.eps
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
  expect_lte(abs(median(t1) - median(t2)), 5e-04)
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
