#' srr_stats (tests)
#' @srrstatsVerbose TRUE
#' @srrstats {G5.4b} See test-feglm.R
#' @noRd
NULL

test_that("fepoisson is similar to fixest", {
  mod <- fepoisson(mpg ~ wt | cyl | am, mtcars)

  mod_base <- glm(
    mpg ~ wt + as.factor(cyl),
    mtcars,
    family = quasipoisson(link = "log")
  )

  coef_dist_base <- coef(mod_base)[2]

  dist_variation <- abs((coef(mod)[1] - coef_dist_base) / coef(mod)[1])

  expect_lt(dist_variation, 0.05)

  expect_output(print(mod))

  expect_visible(summary(mod, type = "cluster"))

  fes <- fixed_effects(mod)
  n <- unname(mod[["nobs"]]["nobs"])
  expect_equal(length(fes), 1)
  expect_equal(length(fitted(mod)), n)
  expect_equal(length(predict(mod)), n)
  expect_equal(length(coef(mod)), 1)
  expect_equal(length(fes), 1)
  
  expect_equal(
    round(fes[["cyl"]][1], 2),
    unname(round(coef(glm(mpg ~ wt + as.factor(cyl), mtcars, family = quasipoisson(link = "log")))[1], 2))
  )

  smod <- summary(mod)

  expect_equal(length(coef(smod)[, 1]), 1)
  expect_output(summary_formula_(smod))
  expect_output(summary_family_(smod))
  expect_output(summary_estimates_(smod, 3))
  expect_output(summary_r2_(smod, 3))
  expect_output(summary_nobs_(smod))
  expect_output(summary_fisher_(smod))
})

#' srr_stats (tests)
#'
#' @srrstatsVerbose TRUE
#'
#' @srrstats {G5.9a} Here we add a censored white noise (i.e., y cannot be < 0
#'  in a Poisson model). The noise is rnorm * .Machine$double.eps to check that
#'  the slopes do not change. See test-feglm.R.
#' @srrstats {RE7.1a} Model fitting is at least as fast or (preferably) faster
#'  than testing with equivalent noisy data (see RE2.4b).*
#'
#' @noRd
NULL

test_that("fepoisson time is the same adding noise to the data", {
  set.seed(123)
  d <- data.frame(
    x = rnorm(1000),
    y = rpois(1000, 1),
    f = factor(rep(1:10, 100))
  )

  set.seed(123)
  d$y2 <- d$y + pmax(rnorm(nrow(d)), 0) * .Machine$double.eps

  m1 <- fepoisson(y ~ x | f, d)
  m2 <- fepoisson(y2 ~ x | f, d)
  expect_equal(coef(m1), coef(m2))
  expect_equal(fixed_effects(m1), fixed_effects(m2))

  t1 <- rep(NA, 10)
  t2 <- rep(NA, 10)
  for (i in 1:10) {
    a <- Sys.time()
    m1 <- fepoisson(y ~ x | f, d)
    b <- Sys.time()
    t1[i] <- b - a

    a <- Sys.time()
    m2 <- fepoisson(y2 ~ x | f, d)
    b <- Sys.time()
    t2[i] <- b - a
  }
  expect_gt(abs(median(t1) / median(t2)), 0.9)
  expect_lt(abs(median(t1) / median(t2)), 1)
  expect_lt(median(t1), median(t2))
})
