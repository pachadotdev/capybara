
<!-- README.md is generated from README.Rmd. Please edit that file -->

# capybara <img src="man/figures/logo.svg" align="right" height="139" alt="" />

<!-- badges: start -->

[![R-CMD-check](https://github.com/pachadotdev/capybara/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/pachadotdev/capybara/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

## About

tldr; If you have a 2-4GB dataset and you need to estimate a
(generalized) linear model with a large number of fixed effects, this
package is for you.

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
elementary approach and uses a very naive code without parallelization,
which achieves very good results considering its simplicity. I hope it
is easy to maintain.

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

## Design choices

Capybara uses C++ to address some bottlenecks. It also uses dplyr
because it allows me to use an expressive syntax.

For v0.1, the benchmark showed that Capybara was much faster, but then I
realized it provided correct estimates for 2 fixed effects. Since v0.2,
the memory footprint is bigger, but at the same time I made the code
more explicit and comments, such that it should be much easier to
address issues.

I tried to implement an important lesson between v0.1 and v0.2: “He who
gives up \[code\] safety for \[code\] speed deserves neither.” (Wickham,
2014).

I know some parts of the code are not particularly easy to understand.
For example, I initially wrote an implementation for Kendall’s Tau (or
Kendall’s correlation) with a time complexity of O(n^2). Posterior work
on it, allowed me to solve the bottleneck that computation created, and
I was able to reduce it to a time complexity of O(n \* log(n)) at the
expense of making the code harder to understand, but I still did my best
to write a straightforward code.

Capybara is full of trade-offs. Using dplyr means there is no way to use
in-place modification of data. This is something intentional in dplyr,
and I prefer to have a slightly bigger memory footprint than do have
side effects (like changing the input data).

With data.table you get faster computation and a slightly reduced use of
memory. The problem is to understand the code when you are familiar with
data.table (I am not). data.table uses in-place modification, which
gives me the heebie-jeebies. I think data.table is great from what I
heard from its users, but I think with my design choices I accomplished
my goal of making model estimation feasible, and now I can run models on
my laptop without relying on UofT’s servers a bit less.

Finally, I am not using parallelization on purpose. I might add this
feature later, but I have colleagues who use Macs, and OMP is very hard
to install when you use the M1/M2 processor.

## Future plans

The other nice thing about dplyr is directly connecting to SQL
databases, something that I shall implement for v0.3 to move the data
filtering to server-side, as I am already using SQL to organize the
around 200 GB of trade datasets that I use.

I will also work on adding a RESET test to summaries as default and make
clustered standard errors computation a bit easier.

This also needs lots of tests. There are a few in the dev folder, but I
need to test with testthat.

## Benchmarks

Median time for the different models in the book [An Advanced Guide to
Trade Policy
Analysis](https://www.wto.org/english/res_e/publications_e/advancedguide2016_e.htm).

| package      |    PPML | Trade Diversion | Endogeneity | Reverse Causality | Non-linear/Phasing Effects | Globalization |
| :----------- | ------: | --------------: | ----------: | ----------------: | -------------------------: | ------------: |
| Alpaca       | 213.4ms |            2.3s |       1.35s |             1.86s |                      2.59s |         4.96s |
| Base R       |    1.5m |           1.53m |      23.43m |            23.52m |                     23.16m |        24.85m |
| **Capybara** | 400.4ms |           4.23s |       1.87s |             2.46s |                      3.33s |         6.25s |
| Fixest       |  67.4ms |        477.08ms |     95.88ms |          136.21ms |                   206.12ms |      415.31ms |

Memory allocation for the same models

| package      |     PPML | Trade Diversion | Endogeneity | Reverse Causality | Non-linear/Phasing Effects | Globalization |
| :----------- | -------: | --------------: | ----------: | ----------------: | -------------------------: | ------------: |
| Alpaca       |  304.8MB |         339.8MB |     306.3MB |          335.61MB |                   393.86MB |      539.49MB |
| Base R       |   2.73GB |           2.6GB |      11.9GB |           11.94GB |                    11.95GB |       11.97GB |
| **Capybara** | 454.18MB |          2.31GB |     965.1MB |            1.17GB |                     1.62GB |        2.88GB |
| Fixest       |  44.59MB |         36.59MB |      28.1MB |           32.43MB |                    41.12MB |       62.87MB |

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

## Donating

If this software is useful to you, please consider donating on [Buy Me A
Coffee](https://buymeacoffee.com/pacha). All donations will be used to
continue improving `capybara`.
