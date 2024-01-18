
<!-- README.md is generated from README.Rmd. Please edit that file -->

# capybara

<!-- badges: start -->

<!-- badges: end -->

The goal of capybara is to …

## Installation

You can install the development version of capybara like so:

``` r
remotes::install_github("pachadotdev/capybara")
```

## Examples

See the documentation in progress: <https://pacha.dev/capybara>

# Debugging

*This debugging is about code quality, not about statistical quality.*
*There is a full set of numerical tests for testthat to check the math.*
*In this section of the test, I can write pi = 3 and if there are no
memory leaks, it will pass the test.*

I run `r_valgrind "dev/test_get_alpha.r" "dev/test_get_alpha.txt"` or
the corresponding test from the project’s root in a new terminal (bash).

This works because I previously defined this in `.bashrc`, to make it
work you need to run `source ~/.bashrc` or reboot:

    # create an alias for R
    alias r="R"
    
    alias rvalgrind="R --vanilla -d 'valgrind -s --track-origins=yes'"
    
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
    
        # if no output file is provided, use dev/valgrind.txt
        if [ -z "$2" ]; then
            output="dev/valgrind.txt"
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

`r_debug_symbols()` makes everything slower, but makes sure that all
compiler optimizations are disabled and then valgrind will point us to
the lines that create memory leaks.

`r_valgrind()` will run an R script and use Linux system tools to test
for initialized values and all kinds of problems that result in memory
leaks.

When you are ready testing, you need to remove `-UDEGUG` from
`src/Makevars`.
