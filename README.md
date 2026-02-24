
<!-- README.md is generated from README.Rmd. Please edit that file -->

# capybara <img src="man/figures/logo.svg" align="right" height="139" alt="" />

<!-- badges: start -->

[![R-CMD-check](https://github.com/pachadotdev/capybara/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/pachadotdev/capybara/actions/workflows/R-CMD-check.yaml)
[![codecov](https://app.codecov.io/gh/pachadotdev/capybara/graph/badge.svg?token=kDP0pWmfRk)](https://app.codecov.io/gh/pachadotdev/capybara)
[![BuyMeACoffee](https://raw.githubusercontent.com/pachadotdev/buymeacoffee-badges/main/bmc-donate-yellow.svg)](https://buymeacoffee.com/pacha)
[![Lifecycle:
stable](https://img.shields.io/badge/lifecycle-stable-brightgreen.svg)](https://lifecycle.r-lib.org/articles/stages.html#stable)
[![CRAN
status](https://www.r-pkg.org/badges/version/capybara)](https://CRAN.R-project.org/package=capybara)
<!-- badges: end -->

## About

Please read my article for the full details about capybara 1.x.y (Open
Access):

> Vargas Sepulveda, Mauricio. 2025. ‘Capybara: Efficient Estimation of
> Generalized Linear Models with High-Dimensional Fixed Effects’. *PLOS
> ONE* 20 (9): e0331178. <https://doi.org/10.1371/journal.pone.0331178>.

I am preparing the manuscript for capybara 2.x.y that is about speed and
memory improvements.

If you have a 2-4GB dataset and you need to estimate a (generalized)
linear model with a large number of fixed effects, this package is for
you. It works with larger datasets as well and facilites computing
clustered standard errors.

‘capybara’ is a fast and small footprint software that provides
efficient functions for demeaning variables before conducting a GLM
estimation. This technique is particularly useful when estimating linear
models with multiple group fixed effects. It is a fork of the excellent
Alpaca package created and maintained by [Dr. Amrei
Stammann](https://github.com/amrei-stammann). The software can estimate
Exponential Family models (e.g., Poisson) and Negative Binomial models.

Traditional QR estimation on the full design matrix can be unfeasible
due to additional memory requirements. The method, which is based on
Halperin (1962) vector projections offers important time and memory
savings without compromising numerical stability in the estimation
process.

The software heavily borrows from Gaure (2013), Stammann (2018) and
Berge (2018) works on OLS and GLM estimation with large fixed effects
implemented in the ‘lfe’, ‘alpaca’ and ‘fixest’ packages. The
differences are that ‘capybara’ does not use C nor Rcpp code, instead it
uses cpp4r and
[armadillo4r](https://github.com/pachadotdev/armadillo4r).

The summary tables borrow from Stata outputs. I have also provided
integrations with ‘broom’ to facilitate the inclusion of statistical
tables in Quarto/Jupyter notebooks.

If this software is useful to you, please consider donating on [Buy Me A
Coffee](https://buymeacoffee.com/pacha). All donations will be used to
continue improving `capybara`.

## Installation

You can install the development version of capybara like so:

``` r
install.packages("capybara", repos = "https://cloud.r-project.org/")
```

You can install the development version of capybara like so:

``` r
remotes::install_github("pachadotdev/capybara")
```

## Examples

See the documentation: <https://pacha.dev/capybara/>.

Here is simple example of estimating a linear model and a Poisson model
with fixed effects:

``` r
m1 <- felm(mpg ~ wt | cyl, mtcars)
m2 <- fepoisson(mpg ~ wt | cyl, mtcars)
summary_table(m1, m2, model_names = c("Linear", "Poisson"))

|     Variable     |       Linear        |      Poisson      |
|------------------|---------------------|-------------------|
| wt               |           -3.206*** |           -0.180* |
|                  |             (0.295) |           (0.072) |
|                  |                     |                   |
| Fixed effects    |                     |                   |
| cyl              |                 Yes |               Yes |
|                  |                     |                   |
| N                |                  32 |                32 |
| R-squared        |               0.837 |             0.616 |

Standard errors in parenthesis
Significance levels: *** p < 0.001; ** p < 0.01; * p < 0.05; . p < 0.1
```

## Installing with compiler optimizations

CRAN packages are built with the `-O2` compiler flag, which is
sufficient for most packages, including capybara. However, if you want
to use the maximum compiler optimizations, you can do so by setting the
`-O3` compiler flag.

To do that, create a user Makevars file in your home directory
(`~/.R/Makevars`) and add the following lines:

``` makefile
# Copy to ~/.R/Makevars if you want to override R's default optimization
CXXFLAGS = -O3
CXX11FLAGS = -O3
CXX14FLAGS = -O3
CXX17FLAGS = -O3
CXX20FLAGS = -O3
```

Additional optimizations can be enabled by setting the
`CAPYBARA_OPTIMIZATIONS` environment variable to “yes” and choosing the
number of cores to use for parallel processing (the default is to use
50% of the available cores). You can do this in your R session like so:

``` r
# Install local version

Sys.setenv(CAPYBARA_OPTIMIZATIONS = "yes")
Sys.setenv(CAPYBARA_CORES = 4) # Set the number of cores to use (optional)

install.packages(".", repos = NULL, type = "source")
# or
devtools::install()
```

This will determine if your hardware allows hardware-specific compiler
flags that provide significant performance improvements (sometimes 2-4x
faster than just using portable flags).

## Code of Conduct

Please note that the capybara project is released with a [Contributor
Code of
Conduct](https://contributor-covenant.org/version/2/1/CODE_OF_CONDUCT.html).
By contributing to this project, you agree to abide by its terms.

## Acknowledgements

Thanks a lot to [Prof. Yoto Yotov](https://yotoyotov.com/) for reviewing
the summary functions.
