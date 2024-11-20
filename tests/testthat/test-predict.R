#' srr_stats (tests)
#' 
#' @srrstats {G5.0} The tests use the widely known mtcars data set. It has few
#'  observations, and it is easy to compare the results with the base R
#'  functions.
#' @srrstats {G5.4b} We determine correctess for GLMs by comparison, checking
#'  the estimates versus base R and hardcoded values obtained with Alpaca
#'  (Stammann, 2018).
#' @srrstats {RE7.4} Extending directly from **RE4.15**, where appropriate, tests should demonstrate and confirm that forecast errors, confidence intervals, or equivalent values increase with forecast horizons.
#' 
#' @noRd
NULL

test_that("predicted values increase the error outside the inter-quartile range for GLMs", {
  m1 <- fepoisson(mpg ~ wt + disp | cyl, mtcars)

  d1 <- mtcars[mtcars$mpg >= quantile(mtcars$mpg, 0.25) & mtcars$mpg <= quantile(mtcars$mpg, 0.75), ]
  d2 <- mtcars[mtcars$mpg < quantile(mtcars$mpg, 0.25) | mtcars$mpg > quantile(mtcars$mpg, 0.75), ]

  pred1 <- predict(m1, newdata = d1, type = "response")
  pred2 <- predict(m1, newdata = d2, type = "response")

  mape <- function(y, yhat) {
    mean(abs(y - yhat) / y)
  }

  mape1 <- mape(d1$mpg, pred1)
  mape2 <- mape(d2$mpg, pred2)

  expect_lt(mape1, mape2)

  # verify prediction compared to base R
  m2 <- glm(mpg ~ wt + disp + as.factor(cyl), mtcars, family = quasipoisson())

  pred1_base <- predict(m2, newdata = d1, type = "response")
  pred2_base <- predict(m2, newdata = d2, type = "response")

  expect_equal(round(pred1, 3), round(unname(pred1_base), 3))
  expect_equal(round(pred2, 3), round(unname(pred2_base), 3))
})

test_that("predicted values increase the error outside the inter-quartile range for LMs", {
  m1 <- felm(mpg ~ wt + disp | cyl, mtcars)

  d1 <- mtcars[mtcars$mpg >= quantile(mtcars$mpg, 0.25) & mtcars$mpg <= quantile(mtcars$mpg, 0.75), ]
  d2 <- mtcars[mtcars$mpg < quantile(mtcars$mpg, 0.25) | mtcars$mpg > quantile(mtcars$mpg, 0.75), ]

  pred1 <- predict(m1, newdata = d1)
  pred2 <- predict(m1, newdata = d2)

  mape <- function(y, yhat) {
    mean(abs(y - yhat) / y)
  }

  mape1 <- mape(d1$mpg, pred1)
  mape2 <- mape(d2$mpg, pred2)

  expect_lt(mape1, mape2)

  # verify prediction compared to base R
  m2 <- lm(mpg ~ wt + disp + as.factor(cyl), mtcars)

  pred1_base <- predict(m2, newdata = d1)
  pred2_base <- predict(m2, newdata = d2)

  expect_equal(round(pred1, 3), round(unname(pred1_base), 3))
  expect_equal(round(pred2, 3), round(unname(pred2_base), 3))
})
