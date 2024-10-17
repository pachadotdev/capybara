#' @title Generalized Linear Models (GLMs) with high-dimensional k-way fixed
#'  effects
#'
#' @srrstats {G1.1} *Statistical Software should document whether the algorithm(s) it implements are:* - *The first implementation of a novel algorithm*; or - *The first implementation within **R** of an algorithm which has previously been implemented in other languages or contexts*; or - *An improvement on other implementations of similar algorithms in **R***.
#' @srrstats {G1.2} *Statistical Software should include a* Life Cycle Statement *describing current and anticipated future states of development.*
#' @srrstats {G1.4} *Software should use [`roxygen2`](https://roxygen2.r-lib.org/) to document all functions.*
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
#' @importFrom dplyr across all_of filter group_by mutate pull select summarise
#'  ungroup vars
#' @importFrom Formula Formula
#' @importFrom ggplot2 ggplot aes geom_point geom_errorbar labs theme_minimal
#'  coord_flip autoplot
#' @importFrom kendallknight kendall_cor kendall_cor_test
#' @importFrom MASS negative.binomial theta.ml
#' @importFrom rlang sym :=
#' @importFrom stats as.formula fitted.values gaussian model.matrix na.omit
#'  pnorm poisson predict printCoefmat qnorm terms vcov
#' @importFrom utils combn
#' @useDynLib capybara, .registration = TRUE
"_PACKAGE"

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
#' @source Advanced Guide to Trade Policy Analysis (ISBN: 978-92-870-4367-2)
"trade_panel"
