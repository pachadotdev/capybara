
<!-- README.md is generated from README.Rmd. Please edit that file -->

# capybara <img src="man/figures/logo.svg" align="right" height="139" alt="" />

<!-- badges: start -->

[![R-CMD-check](https://github.com/pachadotdev/capybara/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/pachadotdev/capybara/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

## About

Capybara is a fast and small footprint software that provides efficient
functions for demeaning variables before conducting a GLM estimation via
Iteratively Weighted Least Squares (IWLS). This technique is
particularly useful when estimating linear models with multiple group
fixed effects.

The software can estimate GLMs from the Exponential Family and also
Negative Binomial models but the focus will be the Poisson estimator
because it is the one used for structural counterfactual analysis in
International Trade. It is relevant to add that the IWLS estimator is
equivalent with the PPML estimator from Santos-Silva et al. 2006

Tradition QR estimation can be unfeasible due to additional memory
requirements. The method, which is based on Halperin 1962 article on
vector projections offers important time and memory savings without
compromising numerical stability in the estimation process.

The software heavily borrows from Gaure 20213 and Stammann 2018 works on
the OLS and IWLS estimator with large k-way fixed effects (i.e., the Lfe
and Alpaca packages). The differences are that Capybara uses an
elementary approach and uses a minimal C++ code without parallelization,
which achieves very good results considering its simplicity. I hope it
is east to maintain.

The summary tables are nothing like R’s default and borrow from the
Broom package and Stata outputs. The default summary from this package
is a Markdown table that you can insert in RMarkdown/Quarto or copy and
paste to Jupyter.

## Installation

You can install the development version of capybara like so:

``` r
remotes::install_github("pachadotdev/capybara")
```

## Examples

See the documentation in progress: <https://pacha.dev/capybara>.

## Benchmarks

Median time for the different models in the book [An Advanced Guide to
Trade Policy
Analysis](https://www.wto.org/english/res_e/publications_e/advancedguide2016_e.htm).

| package  |    PPML | Trade Diversion | Endogeneity | Reverse Causality | Non-linear/Phasing Effects | Globalization |
| :------- | ------: | --------------: | ----------: | ----------------: | -------------------------: | ------------: |
| Alpaca   |   282ms |           1.78s |        1.1s |             1.34s |                      2.18s |         4.48s |
| Base R   |   36.2s |          36.87s |       9.81m |            10.03m |                     10.41m |         10.4m |
| Capybara | 159.2ms |         97.96ms |     81.38ms |           86.77ms |                   104.69ms |      130.22ms |
| Fixest   |  33.6ms |        191.04ms |     64.38ms |            75.2ms |                   102.18ms |      162.28ms |

Memory allocation for the same models

| package  |     PPML | Trade Diversion | Endogeneity | Reverse Causality | Non-linear/Phasing Effects | Globalization |
| :------- | -------: | --------------: | ----------: | ----------------: | -------------------------: | ------------: |
| Alpaca   | 282.78MB |         321.5MB |     270.4MB |             308MB |                    366.5MB |       512.1MB |
| Base R   |   2.73GB |           2.6GB |      11.9GB |            11.9GB |                     11.9GB |          12GB |
| Capybara | 339.13MB |         196.3MB |     162.6MB |           169.1MB |                    181.1MB |       239.9MB |
| Fixest   |  44.79MB |          36.6MB |      28.1MB |            32.4MB |                     41.1MB |        62.9MB |

# Debugging

*This debugging is about code quality, not about statistical quality.*
*There is a full set of numerical tests for testthat to check the math.*
*In this section of the test, I can write pi = 3 and if there are no
memory leaks, it will pass the test.*

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

`r_debug_symbols()` makes everything slower, but makes sure that all
compiler optimizations are disabled and then valgrind will point us to
the lines that create memory leaks.

`r_valgrind()` will run an R script and use Linux system tools to test
for initialized values and all kinds of problems that result in memory
leaks.

When you are ready testing, you need to remove `-UDEGUG` from
`src/Makevars`.
