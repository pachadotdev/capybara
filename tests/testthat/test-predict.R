#' srr_stats (tests)
#' @srrstats {G5.2} Confirms that prediction errors increase outside the inter-quartile range, ensuring model generalization testing.
#' @srrstats {RE3.2} Compares predictions from `fepoisson` and `felm` to those from base R models like `glm` and `lm`.
#' @srrstats {RE3.3} Validates alignment of predictions with established models under similar conditions.
#' @srrstats {RE4.3} Tests robustness of predicted values using inter-quartile and outlier data subsets.
#' @srrstats {RE5.5} Ensures accuracy of prediction methods with unseen data subsets, maintaining expected patterns of error.
#' @srrstats {RE4.15} This is not a time-series package, so I show that the error increases when we predict outside the inter-quartile range.
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
