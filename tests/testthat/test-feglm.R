#' srr_stats (tests)
#' 
#' @srrstatsVerbose TRUE
#' 
#' @srrstats {G5.0} *Where applicable or practicable, tests should use standard data sets with known properties (for example, the [NIST Standard Reference Datasets](https://www.itl.nist.gov/div898/strd/), or data sets provided by other widely-used R packages).*
#' @srrstats {G5.1} *Data sets created within, and used to test, a package should be exported (or otherwise made generally available) so that users can confirm tests and run examples.*
#' @srrstats {G5.2} *Appropriate error and warning behaviour of all functions should be explicitly demonstrated through tests. In particular,*
#' @srrstats {G5.2b} *Explicit tests should demonstrate conditions which trigger every one of those messages, and should compare the result with expected values.*
#' @srrstats {G5.4b} *For new implementations of existing methods, correctness tests should include tests against previous implementations. Such testing may explicitly call those implementations in testing, preferably from fixed-versions of other software, or use stored outputs from those where that is not possible.*
#' @srrstats {G5.8} **Edge condition tests** *to test that these conditions produce expected behaviour such as clear warnings or errors when confronted with data with extreme properties including but not limited to:*
#' @srrstats {G5.8a} *Zero-length data*
#' @srrstats {G5.8b} *Data of unsupported types (e.g., character or complex numbers in for functions designed only for numeric data)*
#' @srrstats {G5.8c} *Data with all-`NA` fields or columns or all identical fields or columns*
#' @srrstats {G5.8d} *Data outside the scope of the algorithm (for example, data with more fields (columns) than observations (rows) for some regression algorithms)*
#' @srrstats {G5.9} **Noise susceptibility tests** *Packages should test for expected stochastic behaviour, such as through the following conditions:*
#' @srrstats {G5.9a} *Adding trivial noise (for example, at the scale of `.Machine$double.eps`) to data does not meaningfully change results*
#' @srrstats {G5.10} *Extended tests should included and run under a common framework with other tests but be switched on by flags such as as a `<MYPKG>_EXTENDED_TESTS="true"` environment variable.* - The extended tests can be then run automatically by GitHub Actions for example by adding the following to the `env` section of the workflow:
#' @srrstats {G5.11} *Where extended tests require large data sets or other assets, these should be provided for downloading and fetched as part of the testing workflow.*
#' @srrstats {RE7.0} *Tests with noiseless, exact relationships between predictor (independent) data.*
#' @srrstats {RE7.0a} In particular, these tests should confirm ability to reject perfectly noiseless input data.
#' @srrstats {RE7.1} *Tests with noiseless, exact relationships between predictor (independent) and response (dependent) data.*
#' @srrstats {RE7.1a} *In particular, these tests should confirm that model fitting is at least as fast or (preferably) faster than testing with equivalent noisy data (see RE2.4b).*
#' @srrstats {RE7.2} Demonstrate that output objects retain aspects of input data such as row or case names (see **RE1.3**).
#' @srrstats {RE7.3} Demonstrate and test expected behaviour when objects returned from regression software are submitted to the accessor methods of **RE4.2**--**RE4.7**.
#' @srrstats {RE7.4} Extending directly from **RE4.15**, where appropriate, tests should demonstrate and confirm that forecast errors, confidence intervals, or equivalent values increase with forecast horizons.
#' 
#' @noRd
NULL

test_that("feglm is similar to glm", {
  # Gaussian ----

  # see felm

  # Poisson

  # see fepoisson

  # Binomial ----

  mod <- feglm(
    am ~ wt + mpg | cyl,
    mtcars,
    family = binomial()
  )

  mod_base <- glm(
    am ~ wt + mpg + as.factor(cyl),
    mtcars,
    family = binomial()
  )

  expect_equal(unname(round(coef(mod) - coef(mod_base)[2:3], 3)), rep(0, 2))

  fe <- unname(drop(fixed_effects(mod)$cyl))
  fe_base <- coef(mod_base)[c(1, 4, 5)]
  fe_base <- unname(fe_base + c(0, rep(fe_base[1], 2)))

  expect_equal(round(fe - fe_base, 2), rep(0, 3))

  # Gamma ----

  mod <- feglm(
    mpg ~ wt + am | cyl,
    mtcars,
    family = Gamma()
  )

  mod_base <- glm(
    mpg ~ wt + am + as.factor(cyl),
    mtcars,
    family = Gamma()
  )

  expect_equal(unname(round(coef(mod) - coef(mod_base)[2:3], 3)), rep(0, 2))

  fe <- unname(drop(fixed_effects(mod)$cyl))
  fe_base <- coef(mod_base)[c(1, 4, 5)]
  fe_base <- unname(fe_base + c(0, rep(fe_base[1], 2)))

  expect_equal(round(fe - fe_base, 2), rep(0, 3))

  # Inverse Gaussian ----

  mod <- feglm(
    mpg ~ wt + am | cyl,
    mtcars,
    family = inverse.gaussian()
  )

  mod_base <- glm(
    mpg ~ wt + am + as.factor(cyl),
    mtcars,
    family = inverse.gaussian()
  )

  expect_equal(unname(round(coef(mod) - coef(mod_base)[2:3], 3)), rep(0, 2))

  fe <- unname(drop(fixed_effects(mod)$cyl))
  fe_base <- coef(mod_base)[c(1, 4, 5)]
  fe_base <- unname(fe_base + c(0, rep(fe_base[1], 2)))

  expect_equal(round(fe - fe_base, 2), rep(0, 3))
})
