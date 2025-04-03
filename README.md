
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

tldr; If you have a 2-4GB dataset and you need to estimate a
(generalized) linear model with a large number of fixed effects, this
package is for you. It works with larger datasets as well and facilites
computing clustered standard errors.

‘capybara’ is a fast and small footprint software that provides
efficient functions for demeaning variables before conducting a GLM
estimation. This technique is particularly useful when estimating linear
models with multiple group fixed effects. It is a fork of the excellent
Alpaca package created and maintained by [Dr. Amrei
Stammann](https://github.com/amrei-stammann). The software can estimate
Exponential Family models (e.g., Poisson) and Negative Binomial models.

Traditional QR estimation can be unfeasible due to additional memory
requirements. The method, which is based on Halperin (1962) vector
projections offers important time and memory savings without
compromising numerical stability in the estimation process.

The software heavily borrows from Gaure (2013) and Stammann (2018) works
on OLS and GLM estimation with large fixed effects implemented in the
‘lfe’ and ‘alpaca’ packages. The differences are that ‘capybara’ does
not use C nor Rcpp code, instead it uses cpp11 and
[cpp11armadillo](https://github.com/pachadotdev/cpp11armadillo).

The summary tables borrow from Stata outputs. I have also provided
integrations with ‘broom’ to facilitate the inclusion of statistical
tables in Quarto/Jupyter notebooks.

If this software is useful to you, please consider donating on [Buy Me A
Coffee](https://buymeacoffee.com/pacha). All donations will be used to
continue improving `capybara`.

## Installation

You can install the development version of capybara like so:

``` r
remotes::install_github("pachadotdev/capybara")
```

## Examples

See the documentation in progress: <https://pacha.dev/capybara/>.

## Design choices

Capybara is full of trade-offs. I have used ‘data.table’ to benefit from
in-place modifications. The model fitting is done on C++ side. While the
code aims to be fast, I prefer to have some bottlenecks instead of low
numerical stability. The principle was: “He who gives up code safety for
code speed deserves neither.” (Wickham, 2014).

## Benchmarks

Median time and memory footprint for the different models in the book
[An Advanced Guide to Trade Policy
Analysis](https://www.wto.org/english/res_e/publications_e/advancedguide2016_e.htm).

| Model             | Package  | Median Time   | Memory        |
| :---------------- | :------- | :------------ | :------------ |
| PPML              | Alpaca   | 822.14 ms - 3 | 302.62 MB - 3 |
| PPML              | Base R   | 45.43 s - 4   | 2.73 GB - 4   |
| PPML              | Capybara | 404.27 ms - 2 | 23.91 MB - 1  |
| PPML              | Fixest   | 140.87 ms - 1 | 44.59 MB - 2  |
|                   |          |               |               |
| Trade Diversion   | Alpaca   | 3.73 s - 3    | 339.79 MB - 3 |
| Trade Diversion   | Base R   | 45.91 s - 4   | 2.6 GB - 4    |
| Trade Diversion   | Capybara | 929.89 ms - 1 | 30.77 MB - 1  |
| Trade Diversion   | Fixest   | 1.01 s - 2    | 36.59 MB - 2  |
|                   |          |               |               |
| Endogeneity       | Alpaca   | 2.9 s - 3     | 306.27 MB - 3 |
| Endogeneity       | Base R   | 12.19 m - 4   | 11.94 GB - 4  |
| Endogeneity       | Capybara | 1.3 s - 2     | 16.81 MB - 1  |
| Endogeneity       | Fixest   | 247.72 ms - 1 | 28.08 MB - 2  |
|                   |          |               |               |
| Reverse Causality | Alpaca   | 3.7 s - 3     | 335.61 MB - 3 |
| Reverse Causality | Base R   | 12.23 m - 4   | 11.94 GB - 4  |
| Reverse Causality | Capybara | 1.36 s - 2    | 19.86 MB - 1  |
| Reverse Causality | Fixest   | 329.78 ms - 1 | 32.43 MB - 2  |
|                   |          |               |               |
| Phasing Effects   | Alpaca   | 4.78 s - 3    | 393.86 MB - 3 |
| Phasing Effects   | Base R   | 12.18 m - 4   | 11.95 GB - 4  |
| Phasing Effects   | Capybara | 1.49 s - 2    | 25.95 MB - 1  |
| Phasing Effects   | Fixest   | 525.04 ms - 1 | 41.12 MB - 2  |
|                   |          |               |               |
| Globalization     | Alpaca   | 7.97 s - 3    | 539.49 MB - 3 |
| Globalization     | Base R   | 11.59 m - 4   | 11.97 GB - 4  |
| Globalization     | Capybara | 1.94 s - 2    | 41.19 MB - 1  |
| Globalization     | Fixest   | 914.51 ms - 1 | 62.87 MB - 2  |

## Changing the number of cores

Note that you can edit the `Makevars` file to change the number of cores
that capybara uses, here is an example of how it affects the performance

| cores | PPML | Trade Diversion |
| :---- | ---: | --------------: |
| 2     | 1.8s |           16.2s |
| 4     | 1.5s |           14.0s |
| 6     | 0.8s |            2.4s |
| 8     | 0.4s |            0.9s |

## Testing and debugging

## Testing

I use `testthat` (e.g., `devtools::test()`) to compare the results with
base R. These tests are about the correctness of the results.

### Debuging

I run `r_valgrind "dev/valgrind-kendall-correlation.r"` or the
corresponding test from the project’s root in a new terminal (bash)
after running `devtools::install()`. These tests are about memory leaks
(e.g., I use repeteated computations and sometimes things such as “pi =
3”).

This works because I previously defined this in `.bashrc`, to make it
work you need to run `source ~/.bashrc` or reboot your computer.

    function r_debug_symbols () {
        # if src/Makevars does not exist, exit
        if [ ! -f src/Makevars ]; then
            echo "File src/Makevars does not exist"
            return 1
        fi
    
        # if src/Makevars contains a line that says "PKG_CPPFLAGS"
        # but there is no "-UDEBUG -g" on it
        # then add "PKG_CPPFLAGS += -UDEBUG -g" at the end
        if grep -q "PKG_CPPFLAGS" src/Makevars; then
            if ! grep -q "PKG_CPPFLAGS.*-UDEBUG.*-g" src/Makevars; then
                echo "PKG_CPPFLAGS += -UDEBUG -g" >> src/Makevars
            fi
        fi
    
        # if src/Makevars does not contain a line that reads
        # PKG_CPPFLAGS ...something... -UDEBUG -g ...something...
        # then add PKG_CPPFLAGS = -UDEBUG -g to it
        if ! grep -q "PKG_CPPFLAGS.*-UDEBUG.*-g" src/Makevars; then
            echo "PKG_CPPFLAGS = -UDEBUG -g" >> src/Makevars
        fi
    }
    
    function r_valgrind () {
        # if no argument is provided, ask for a file
        if [ -z "$1" ]; then
            read -p "Enter the script to debug: " script
        else
            script=$1
        fi
    
        # if no output file is provided, use the same filename but ended in txt
        if [ -z "$2" ]; then
            output="${script%.*}.txt"
        else
            output=$2
        fi
    
        # if the file does not exist, exit
        if [ ! -f "$script" ]; then
            echo "File $script does not exist"
            return 1
        fi
    
        # if the file does not end in .R/.r, exit
        shopt -s nocasematch
        if [[ "$script" != *.R ]]; then
            echo "File $script does not end in .R or .r"
            return 1
        fi
        shopt -u nocasematch
    
        # run R in debug mode, but after that we compiled with debug symbols
        # see https://reside-ic.github.io/blog/debugging-memory-errors-with-valgrind-and-gdb/
        # R -d 'valgrind -s --leak-check=full --show-leak-kinds=all --track-origins=yes' -f $script 2>&1 | tee valgrind.txt
        R --vanilla -d 'valgrind -s --track-origins=yes' -f $script 2>&1 | tee $output
    }
    
    # create an alias for R
    alias r="R"
    alias rvalgrind="R --vanilla -d 'valgrind -s --track-origins=yes'"

`r_debug_symbols` makes everything slower, but makes sure that all
compiler optimizations are disabled and then valgrind will point us to
the lines that create memory leaks.

`r_valgrind` will run an R script and use Linux system tools to test for
initialized values and all kinds of problems that result in memory
leaks.

When you are ready testing, you need to remove `-UDEBUG` from
`src/Makevars`.

## Code of Conduct

Please note that the capybara project is released with a [Contributor
Code of
Conduct](https://contributor-covenant.org/version/2/1/CODE_OF_CONDUCT.html).
By contributing to this project, you agree to abide by its terms.

## Acknowledgements

Thanks a lot to [Prof. Yoto Yotov](https://yotoyotov.com/) for reviewing
the summary functions.
