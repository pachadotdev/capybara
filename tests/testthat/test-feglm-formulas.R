#' srr_stats (tests)
#' @srrstats {RE3.2} Compares model outputs (coefficients) against established benchmarks like base R's `glm`.
#' @noRd
NULL

# log(y) ~ x and log(y) ~ log(x) are already tested in test-feglm.R
# this is to test all other formula operators such as
# I(x1 * x2): multiplication
# x1 * x2: interaction
# poly(x, degree): polynomial
# etc

test_that("feglm works with formula operators of the form y ~ I(x1 * x2)", {
  m1 <- feglm(log(mpg) ~ I(wt * hp), data = mtcars)
  m1_no_space <- feglm(log(mpg) ~ I(wt*hp), data = mtcars)
  m2 <- glm(log(mpg) ~ I(wt * hp), data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(coef(m1_no_space), coef(m2), tolerance = 1e-6)
})

test_that("feglm with x1 * x2 expands to x1 + x2 + x1:x2", {
  # x1 * x2 expands to x1 + x2 + x1:x2
  m1 <- feglm(log(mpg) ~ wt * hp, data = mtcars)
  m2 <- glm(log(mpg) ~ wt * hp, data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

test_that("feglm with three-way interaction operator * expands correctly", {
  m1 <- feglm(log(mpg) ~ wt * hp * drat, data = mtcars)
  m2 <- glm(log(mpg) ~ wt * hp * drat, data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

test_that("feglm with x1:x2 creates only the interaction term", {
  m1 <- feglm(log(mpg) ~ wt + hp + wt:hp, data = mtcars)
  m2 <- glm(log(mpg) ~ wt + hp + wt:hp, data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm works with interaction + removed term", {
  m1 <- feglm(log(mpg) ~ wt * hp - hp, data = mtcars)
  m2 <- glm(log(mpg) ~ wt * hp - hp, data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm works with intercept removal", {
  m1 <- feglm(log(mpg) ~ wt + hp - 1, data = mtcars)
  m2 <- glm(log(mpg) ~ wt + hp - 1, data = mtcars)
  m3 <- feglm(log(mpg) ~ 0 + wt + hp, data = mtcars)
  m4 <- glm(log(mpg) ~ 0 + wt + hp, data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(coef(m3), coef(m4), tolerance = 1e-6)
  expect_equal(coef(m1), coef(m3), tolerance = 1e-6)
})

test_that("feglm with (x1 + x2 + x3)^2 expands to all main effects and two-way interactions", {
    m1 <- feglm(log(mpg) ~ (wt + hp + drat)^2, data = mtcars)
    m2 <- glm(log(mpg) ~ (wt + hp + drat)^2, data = mtcars)

    expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
    expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm with (x1 + x2)^3 expands to all main effects, two-way and three-way interactions", {
  m1 <- feglm(log(mpg) ~ (wt + hp)^3, data = mtcars)
  m2 <- glm(log(mpg) ~ (wt + hp)^3, data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm works with poly() for polynomial terms", {
  m1 <- feglm(log(mpg) ~ poly(wt, 2), data = mtcars)
  m2 <- glm(log(mpg) ~ poly(wt, 2), data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm works with poly() degree 3", {
  m1 <- feglm(log(mpg) ~ poly(wt, 3), data = mtcars)
  m2 <- glm(log(mpg) ~ poly(wt, 3), data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm works with poly() raw = TRUE", {
  m1 <- feglm(log(mpg) ~ poly(wt, 2, raw = TRUE), data = mtcars)
  m2 <- glm(log(mpg) ~ poly(wt, 2, raw = TRUE), data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm works with factor interactions", {
  # Create a factor variable
  mtcars_with_factor <- mtcars
  mtcars_with_factor$cyl_factor <- factor(mtcars_with_factor$cyl)

  m1 <- feglm(log(mpg) ~ wt * cyl_factor, data = mtcars_with_factor)
  m2 <- glm(log(mpg) ~ wt * cyl_factor, data = mtcars_with_factor)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm works with factor-only interaction", {
  mtcars_with_factors <- mtcars
  mtcars_with_factors$cyl_factor <- factor(mtcars_with_factors$cyl)
  mtcars_with_factors$vs_factor <- factor(mtcars_with_factors$vs)

  m1 <- feglm(log(mpg) ~ cyl_factor:vs_factor, data = mtcars_with_factors)
  m2 <- glm(log(mpg) ~ cyl_factor:vs_factor, data = mtcars_with_factors)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

test_that("feglm works with transformations in formula", {
  m1 <- feglm(log(mpg) ~ log(wt) + sqrt(hp), data = mtcars)
  m2 <- glm(log(mpg) ~ log(wt) + sqrt(hp), data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

test_that("feglm works with I() for multiple transformations", {
  m1 <- feglm(log(mpg) ~ I(wt^2) + I(hp^2), data = mtcars)
  m2 <- glm(log(mpg) ~ I(wt^2) + I(hp^2), data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

test_that("feglm works with I() for division", {
  m1 <- feglm(log(mpg) ~ I(wt / hp), data = mtcars)
  m2 <- glm(log(mpg) ~ I(wt / hp), data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

test_that("feglm works with I() for addition and subtraction", {
  m1 <- feglm(log(mpg) ~ I(wt + hp) + I(drat - qsec), data = mtcars)
  m2 <- glm(log(mpg) ~ I(wt + hp) + I(drat - qsec), data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm works with complex formula combining multiple operators", {
  m1 <- feglm(log(mpg) ~ wt * hp + I(drat^2) + log(qsec), data = mtcars)
  m2 <- glm(log(mpg) ~ wt * hp + I(drat^2) + log(qsec), data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm works with . operator (all variables)", {
  # Create a smaller dataset for this test
  small_mtcars <- mtcars[, c("mpg", "wt", "hp", "drat")]

  m1 <- feglm(log(mpg) ~ ., data = small_mtcars)
  m2 <- glm(log(mpg) ~ ., data = small_mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm works with . operator and subtraction", {
  small_mtcars <- mtcars[, c("mpg", "wt", "hp", "drat", "qsec")]

  m1 <- feglm(log(mpg) ~ . - qsec, data = small_mtcars)
  m2 <- glm(log(mpg) ~ . - qsec, data = small_mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm works with nested interactions using /", {
  # x1/x2 expands to x1 + x1:x2
  mtcars_with_factor <- mtcars
  mtcars_with_factor$cyl_factor <- factor(mtcars_with_factor$cyl)

  m1 <- feglm(log(mpg) ~ wt/cyl_factor, data = mtcars_with_factor)
  m2 <- glm(log(mpg) ~ wt/cyl_factor, data = mtcars_with_factor)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm works with %in% operator for nesting", {
  # x2 %in% x1 is equivalent to x1/x2 - x1
  mtcars_with_factor <- mtcars
  mtcars_with_factor$cyl_factor <- factor(mtcars_with_factor$cyl)
  mtcars_with_factor$vs_factor <- factor(mtcars_with_factor$vs)

  m1 <- feglm(log(mpg) ~ wt + vs_factor %in% cyl_factor, data = mtcars_with_factor)
  m2 <- glm(log(mpg) ~ wt + vs_factor %in% cyl_factor, data = mtcars_with_factor)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm works with interaction between transformation and variable", {
  m1 <- feglm(log(mpg) ~ log(wt) * hp, data = mtcars)
  m2 <- glm(log(mpg) ~ log(wt) * hp, data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm works with poly() in interaction", {
  m1 <- feglm(log(mpg) ~ poly(wt, 2) * hp, data = mtcars)
  m2 <- glm(log(mpg) ~ poly(wt, 2) * hp, data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm works with as.factor() in formula", {
  m1 <- feglm(log(mpg) ~ wt + as.factor(cyl), data = mtcars)
  m2 <- glm(log(mpg) ~ wt + as.factor(cyl), data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})

# TODO: fix this one
test_that("feglm works with cut() to create factor from continuous variable", {
  m1 <- feglm(log(mpg) ~ cut(wt, breaks = 3), data = mtcars)
  m2 <- glm(log(mpg) ~ cut(wt, breaks = 3), data = mtcars)

  expect_equal(coef(m1), coef(m2), tolerance = 1e-6)
  expect_equal(names(coef(m1)), names(coef(m2)))
})
