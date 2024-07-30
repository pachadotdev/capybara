
<!-- README.md is generated from README.Rmd. Please edit that file -->

# capybara <img src="man/figures/logo.svg" align="right" height="139" alt="" />

<!-- badges: start -->

[![R-CMD-check](https://github.com/pachadotdev/capybara/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/pachadotdev/capybara/actions/workflows/R-CMD-check.yaml)
[![codecov](https://codecov.io/gh/pachadotdev/capybara/graph/badge.svg?token=kDP0pWmfRk)](https://codecov.io/gh/pachadotdev/capybara)
[![BuyMeACoffee](https://raw.githubusercontent.com/pachadotdev/buymeacoffee-badges/main/bmc-donate-yellow.svg)](https://www.buymeacoffee.com/pacha)
[![Lifecycle:
stable](https://img.shields.io/badge/lifecycle-stable-brightgreen.svg)](https://lifecycle.r-lib.org/articles/stages.html#stable)
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

See the documentation in progress: <https://pacha.dev/capybara>.

## Design choices

Capybara uses C++ and vectorized R operations to address bottlenecks
where possible. Some parts of the code use ‘dplyr’, which allows me to
write code that is easier to understand and it works well to performed
grouped operations. The intensive computations are done on C++ side. I
tried to implement this idea from v0.2 and onwards: “He who gives up
code safety for code speed deserves neither.” (Wickham, 2014).

I know some parts of the code are not particularly easy to understand.
For example, such as my implementation of Kendall’s Tau (or Kendall’s
correlation) with a time complexity of O(n \* log(n)) instead of O(n^2).
I still did my best to write a straightforward code.

Capybara is full of trade-offs. I have a branch where I used ‘dplyr’ and
‘dtplyr’ to help myself with the ‘data.table’ syntax, otherwise there is
no way to use in-place modification of data. Because ‘data.table’
modifies the original data (e.g., it converts ‘data.frame’ and ‘tibble’
structures into ‘data.table’ structures), the main branch uses ‘dplyr’
to avoid side effects.

In my research I use ‘SQL’ because I have over 200 GB of international
trade data, where ‘dplyr’ helps a lot because it allows me to query
‘SQL’ directly from R mand just using ‘dplyr’ syntax, something
impossible with ‘data.table’, which requires me to go to the ‘SQL’
editor en export my queries in CSV format and then import them in R. The
downside is that ‘dplyr’ is slower than ‘data.table’ and uses more
memory.

I think with my design choices I accomplished my goal of fitting models
in my laptop instead of relying on UofT’s servers.

## Future plans

I will add a RESET test.

There are a few tests but these have to be expanded.

## Benchmarks

Median time for the different models in the book [An Advanced Guide to
Trade Policy
Analysis](https://www.wto.org/english/res_e/publications_e/advancedguide2016_e.htm).

| package      |   PPML | Trade Diversion | Endogeneity | Reverse Causality | Non-linear/Phasing Effects | Globalization |
| :----------- | -----: | --------------: | ----------: | ----------------: | -------------------------: | ------------: |
| Alpaca       |   0.4s |            2.6s |        1.6s |              2.0s |                       3.1s |          5.3s |
| Base R       | 120.0s |            2.0m |     1380.0s |           1440.0s |                    1380.0s |       1500.0s |
| **Capybara** |   0.3s |            2.0s |        1.2s |              1.4s |                       1.7s |          3.4s |
| Fixest       |   0.1s |            0.5s |        0.1s |              0.2s |                       0.3s |          0.5s |

Memory allocation for the same models

| package      |   PPML | Trade Diversion | Endogeneity | Reverse Causality | Non-linear/Phasing Effects | Globalization |
| :----------- | -----: | --------------: | ----------: | ----------------: | -------------------------: | ------------: |
| Alpaca       |  307MB |           341MB |       306MB |             336MB |                      395MB |         541MB |
| Base R       | 3000MB |          3000MB |     12000MB |           12000GB |                    12000GB |       12000MB |
| **Capybara** |   27MB |            32MB |        20MB |              23MB |                       29MB |          43MB |
| Fixest       |   44MB |            36MB |        27MB |              32MB |                       41MB |          63MB |

# Debugging

I run `r_valgrind "dev/test_get_alpha.r"` or the corresponding test from
the project’s root in a new terminal (bash).

This works because I previously defined this in `.bashrc`, to make it
work you need to run `source ~/.bashrc` or reboot:

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
