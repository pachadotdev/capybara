#' srr_stats (tests)
#' @srrstats {RE3.1} Validates consistency between `fepoisson` and other established R models like `glm` with comparable families.
#' @srrstats {RE3.2} Compares coefficients produced by `fepoisson` with those from base R models to validate similarity.
#' @srrstats {RE4.3} Ensures stable estimates when adding negligible noise to the data.
#' @srrstats {RE5.1} Validates proper output generation for the model summary and printing methods.
#' @srrstats {RE7.3} Confirms that estimated coefficients are within a reasonable variation threshold compared to baseline models.
#' @srrstats {G5.10} The CAPYBARA_EXTENDED_TESTS environment variable can be set to true to run extended tests.
#' @srrstats {G5.11} The extended tests do not require additional downloads.
#' @srrstats {G5.11a} As for G5.11., the extended tests do not require additional downloads.
#' @srrstats {G5.12} The extended tests verify that the algorithm fitting time is robust to noise. This has to be tested with a larger dataset to see that time(clean) <= time(noisy).
#' @noRd
NULL

test_that("fepoisson responds well to noise", {
  skip_on_cran()

  if (Sys.getenv("CAPYBARA_EXTENDED_TESTS") == "yes") {
    n <- 10e5

    set.seed(123)

    d <- data.frame(
      x = rnorm(n),
      y = rpois(n, 1),
      f = factor(rep(1:10, 10e4))
    )

    d$y2 <- d$y + pmax(rnorm(nrow(d)), 0) * .Machine$double.eps

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
  } else {
    expect_true(TRUE) # just to avoid testthat note
  }
})

test_that("fepoisson returns the same RTA coef as alpaca and fixest", {
  skip_on_cran()

  if (Sys.getenv("CAPYBARA_EXTENDED_TESTS") == "yes") {
    # Filter where importer != exporter
    form <- trade ~ log_dist + cntg + lang + clny + rta | exp_year + imp_year

    if (!requireNamespace("alpaca", quietly = TRUE)) {
      skip("alpaca package is not available")
    }

    if (!requireNamespace("fixest", quietly = TRUE)) {
      skip("fixest package is not available")
    }

    rta_coefs <- c(
      "alpaca" = round(alpaca::feglm(form, trade_panel, family = poisson())$coefficients["rta"], 2),
      "fixest" = round(fixest::fepois(form, trade_panel)$coefficients["rta"], 2),
      "capybara" = round(fepoisson(form, trade_panel)$coefficients["rta"], 2)
    )

    expect_equal(
      unname(rta_coefs["alpaca"]),
      unname(rta_coefs["capybara"]),
      tolerance = 0.01
    )

    expect_equal(
      unname(rta_coefs["fixest"]),
      unname(rta_coefs["capybara"]),
      tolerance = 0.01
    )
  } else {
    expect_true(TRUE) # just to avoid testthat note
  }
})
