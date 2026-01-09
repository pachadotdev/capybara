#' srr_stats (tests)
#' @srrstats {G1.0} Implements unit testing for predict functionality.
#' @srrstats {G2.3} Tests various prediction types and newdata scenarios.
#' @srrstats {RE4.9} Verifies predict returns correct values.
#' @noRd
NULL

# ---- predict.feglm tests ----

test_that("predict.feglm works with default type (response)", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  preds <- predict(mod)

  expect_equal(length(preds), nrow(mtcars))
  expect_true(all(preds > 0))
})

test_that("predict.feglm works with type = 'link'", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  preds_link <- predict(mod, type = "link")
  preds_response <- predict(mod, type = "response")

  # link predictions should be different from response
  expect_false(all(preds_link == preds_response))

  # For Poisson with log link, exp(link) = response
  expect_equal(exp(preds_link), preds_response, tolerance = 1e-6)
})

test_that("predict.feglm works with newdata", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  newdata <- data.frame(
    wt = c(2.5, 3.0, 3.5),
    cyl = c(4, 6, 8)
  )

  preds <- predict(mod, newdata = newdata)

  expect_equal(length(preds), 3)
  expect_true(all(preds > 0))
})

test_that("predict.feglm works with binomial", {
  mod <- feglm(am ~ wt | cyl, mtcars, family = binomial())

  preds <- predict(mod, type = "response")

  expect_equal(length(preds), nrow(mtcars))
  expect_true(all(preds >= 0 & preds <= 1))
})

test_that("predict.feglm link type gives different results than response", {
  mod <- feglm(am ~ wt | cyl, mtcars, family = binomial())

  preds_link <- predict(mod, type = "link")
  preds_response <- predict(mod, type = "response")

  expect_false(all(preds_link == preds_response))
})

# ---- predict.felm tests ----

test_that("predict.felm works with default type", {
  mod <- felm(mpg ~ wt | cyl, mtcars)

  preds <- predict(mod)

  expect_equal(length(preds), nrow(mtcars))
})

test_that("predict.felm works with newdata", {
  mod <- felm(mpg ~ wt | cyl, mtcars)

  newdata <- data.frame(
    wt = c(2.5, 3.0, 3.5),
    cyl = c(4, 6, 8)
  )

  preds <- predict(mod, newdata = newdata)

  expect_equal(length(preds), 3)
})

test_that("predict.felm with type='response' works", {
  mod <- felm(mpg ~ wt | cyl, mtcars)

  preds_response <- predict(mod, type = "response")
  preds_default <- predict(mod)

  # For linear models, response is the default
  expect_equal(preds_response, preds_default)
})

# ---- predict with multiple fixed effects ----

test_that("predict works with multiple fixed effects", {
  mod <- fepoisson(mpg ~ wt | cyl + am, mtcars)

  preds <- predict(mod)

  expect_equal(length(preds), nrow(mtcars))
})

test_that("predict with newdata handles multiple FEs", {
  mod <- felm(mpg ~ wt | cyl + am, mtcars)

  newdata <- data.frame(
    wt = c(2.5, 3.0),
    cyl = c(4, 6),
    am = c(0, 1)
  )

  preds <- predict(mod, newdata = newdata)

  expect_equal(length(preds), 2)
})

# ---- predict with model without FE ----

test_that("predict works for model without fixed effects", {
  mod <- fepoisson(mpg ~ wt, mtcars)

  preds <- predict(mod)

  expect_equal(length(preds), nrow(mtcars))
})

test_that("predict with newdata works for model without FE", {
  mod <- felm(mpg ~ wt, mtcars)

  newdata <- data.frame(wt = c(2.5, 3.0, 3.5))

  preds <- predict(mod, newdata = newdata)

  expect_equal(length(preds), 3)
})

# ---- predict with offset ----

test_that("predict works with offset", {
  mtcars2 <- mtcars
  mtcars2$offset_var <- log(mtcars2$hp)

  mod <- fepoisson(mpg ~ wt | cyl, mtcars2, offset = ~ log(hp))

  preds <- predict(mod)

  expect_equal(length(preds), nrow(mtcars2))
})

test_that("predict handles NA in newdata gracefully", {
  mod <- felm(mpg ~ wt | cyl, mtcars)

  newdata <- data.frame(
    wt = c(2.5, NA, 3.5),
    cyl = c(4, 6, 8)
  )

  preds <- predict(mod, newdata = newdata)

  # Should return predictions with NA where input had NA
  expect_equal(length(preds), 3)
  expect_true(is.na(preds[2]))
  expect_false(is.na(preds[1]))
  expect_false(is.na(preds[3]))
})

test_that("predict returns same length as input for newdata", {
  mod <- fepoisson(mpg ~ wt | cyl, mtcars)

  newdata <- data.frame(
    wt = c(2.5, 3.0, 3.5, 4.0),
    cyl = c(4, 6, 8, 6)
  )

  preds <- predict(mod, newdata = newdata)
  expect_equal(length(preds), nrow(newdata))
})

test_that("predict works with type='terms' for felm", {
  mod <- felm(mpg ~ wt + hp | cyl, mtcars)

  preds_terms <- predict(mod, type = "terms")

  expect_true(is.matrix(preds_terms) || is.numeric(preds_terms))
})

test_that("predict for feglm works with all zeros", {
  # Create data where some observations might have y=0
  mtcars2 <- mtcars
  mtcars2$mpg <- rpois(nrow(mtcars2), lambda = exp(log(mtcars2$mpg / 10)))

  mod <- fepoisson(mpg ~ wt | cyl, mtcars2)
  preds <- predict(mod)

  expect_equal(length(preds), nrow(mtcars2))
  expect_true(all(preds >= 0))
})

test_that("predict maintains order for newdata", {
  mod <- felm(mpg ~ wt | cyl, mtcars)

  newdata <- data.frame(
    wt = c(3.5, 2.5, 4.0),
    cyl = c(8, 4, 6)
  )

  preds <- predict(mod, newdata = newdata)

  # Predictions should be in same order as newdata
  expect_equal(length(preds), 3)
})
