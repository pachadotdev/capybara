# capybara 0.4.6

* Fixes notes from tidyselect regarding the use of `all_of()`.
* The C++ code follows a more consistent style.
* The GH-Actions do not test gcc 4.8 anymore.

# capybara 0.4.5

* Ungroups the data to avoid issues with the model matrix

# capybara 0.4

* Uses R's C API efficiently to add a bit more of memory optimizations

# capybara 0.3.5

* Uses Mat<T> consistently for all matrix operations (avoids vectors)

# capybara 0.3

* Reduces memory footprint ~45% by moving some computation to Armadillo's side

# capybara 0.2

* Includes pseudo R2 (same as Stata) for Poisson models

# capybara 0.1

* Initial CRAN submission.
