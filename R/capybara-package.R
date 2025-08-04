#' srr_stats
#' @srrstats {G1.1} The algorithm is a full refactor with memory and speed
#'  improvements for a previous R implementation (Stammann, 2018). The reference
#'  is Stammann (2018) <https://arxiv.org/abs/1707.01815> for GLMs, and also
#'  Gaure (2013) <https://dx.doi.org/10.1016/j.csda.2013.03.024> for LMs.
#' @srrstats {G1.2} This describes the current and anticipated future states of
#'  development.
#' @srrstats {G1.3} For fixed effects, I mean the "c" coeffients in the model
#'  mpg_i = a + b * wt_i + c * cyl_i + e_i with the variables from the mtcars
#'  dataset. The model notation for this example is mpg ~ wt | cyl.
#' @srrstats {G1.4} The package uses roxygen2.
#' @srrstats {G1.4a} All internal (non-exported) functions are documented. See
#'  the `*_helpers.R` files.
#' @srrstats {G1.5} The test include examples to verify the speed gains
#'  in this implementation compare to base R.
#' @srrstats {G1.6} To keep dependencies minimal, we compare against base R in
#'  the tests. An alternative would be to compare against alpaca.
#' @srrstats {RE4.12} The link and inverse link functions are written in C++
#'  to use those with the Armadillo library. This is in the file
#'  `src/05_glm_fit.cpp`.
#' @noRd
NULL

#' NA_standards
#' @srrstatsNA {G2.6} Only some model parameters can be unidimensional. To fit
#'  a regression we need at least two observations and two variables.
#' @srrstatsNA {G5.6b} No randomness is needed for the in fixed effects
#'  estimation. With the model slopes, recovering the fixed effects is a
#'  deterministic process.
#' @srrstatsNA {G2.9} Conversion of variables from factor to character is not
#'  conducted and the original input data is not modified.
#' @srrstatsNA {G2.12} `data.frame`-like tabular objects which have list
#'  columns cannot be used as input data. This behaviour should be tested.
#' @srrstatsNA {G2.14c} Missing data is not replaced with imputed values.
#' @srrstatsNA {G2.14c} Replacing data with imputed values bias the estimation.
#'  This is not done in the package, and it is left to the user to decide
#'  when processing the data.
#' @srrstatsNA {RE7.0a} No cross-validation implemented in this package.
#' @noRd
NULL

#' @title Generalized Linear Models (GLMs) with high-dimensional k-way fixed
#'  effects
#'
#' @description
#' Provides a routine to partial out factors with many levels during the
#' optimization of the log-likelihood function of the corresponding GLM. The
#' package is based on the algorithm described in Stammann (2018). It also
#' offers an efficient algorithm to recover estimates of the fixed effects in a
#' post-estimation routine and includes robust and multi-way clustered standard
#' errors. Further the package provides analytical bias corrections for binary
#' choice models derived by Fernández-Val and Weidner (2016) and Hinz, Stammann,
#' and Wanner (2020). This package is a ground up rewrite with multiple
#' refactors, optimizations, and new features compared to the original package
#' `alpaca`. In its current state, the package is stable and future changes will
#' be limited to bug fixes and improvements, but not to altering the functions'
#' arguments or outputs.
#'
#' @name capybara-package
#' @importFrom data.table setDT := .SD
#' @importFrom Formula Formula
#' @importFrom ggplot2 ggplot aes geom_point geom_errorbar labs theme_minimal
#'  coord_flip autoplot
#' @importFrom kendallknight kendall_cor kendall_cor_test
#' @importFrom MASS negative.binomial theta.ml
#' @importFrom stats as.formula fitted.values gaussian model.matrix na.omit
#'  pnorm poisson predict printCoefmat qnorm terms vcov
#' @importFrom utils combn
#' @useDynLib capybara, .registration = TRUE
"_PACKAGE"

#' srr_stats (tests)
#' @srrstats {G5.1} The panel is exported and used in the package examples.
#' @noRd
NULL

#' Trade Panel 1986-2006
#'
#' Aggregated exports at origin-destination-year level for 1986-2006.
#'
#' @format ## `trade_panel`
#' A data frame with 14,285 rows and 7 columns:
#' \describe{
#'   \item{trade}{Nominal trade flows in current US dollars}
#'   \item{dist}{Population-weighted bilateral distance between country 'i' and
#'    'j', in kilometers}
#'   \item{cntg}{Indicator. Equal to 1 if country 'i' and 'j' share a common
#'    border}
#'   \item{lang}{Indicator. Equal to 1 if country 'i' and 'j' speak the same
#'    official language}
#'   \item{clny}{Indicator. Equal to 1 if country 'i' and 'j' share a colonial
#'    relationship}
#'   \item{year}{Year of observation}
#'   \item{exp_year}{Exporter ISO country code and year}
#'   \item{imp_year}{Importer ISO country code and year}
#' }
#'
#' @source Advanced Guide to Trade Policy Analysis (ISBN: 978-92-870-4367-2)
"trade_panel"
