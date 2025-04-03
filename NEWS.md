# capybara 0.9.3

* Fixes the `tidy()` method for linear models (`felm` class). Now it does not
  require to load the `tibble` package to work.
* Adds a wrapper to present multiple models into a single table with the option
  to export to LaTeX.
  
# capybara 0.9.2

* Implements Irons and Tuck acceleration for fast convergence.

# capybara 0.9.1

* Fixes a minor uninitialized variable in the C++ code used for a conditional
  check.
  
# capybara 0.9

* First CRAN version

* Refactored functions to avoid data copies:
  * center variables
  * crossprod
  * GLM and LM fit
  * get alpha
  * group sums
  * mu eta
  * variance

* `iter_center_max` and `iter_inner_max` now can be modified in
  `feglm_control()`.

# capybara 0.8.0

* Dedicated functions for linear models to avoid the overhead of running
  the GLM function with a Gaussian link.

# capybara 0.7.0

* The predict method now allows to pass new data to predict the outcome.
* Fully documented code and tests according to rOpenSci standards.

# capybara 0.6.0

* Moves all the heavy computation to C++ using Armadillo and it exports the 
  results to R. Previously, there were multiple data copies between R and C++
  that added overhead to the computations.
* The previous versions returned MX by default, now it has to be specified.
* Adds code to extract the fixed effects with `felm` objects.

# capybara 0.5.2

* Uses an O(n log(n)) algorithm to compute the Kendall correlation for the
  pseudo-R2 in the Poisson model.

# capybara 0.5.1

* Using `arma::field` consistently instead of `std::vector<std::vector<>>` for indices.
* Linear algebra changes, such as using `arma::inv` instead of solving `arma::qr` for the inverse.
* Replaces multiple for loops with dedicated Armadillo functions.

# capybara 0.5.0

* Avoids for loops in the C++ code, and instead uses Armadillo's functions.
* O(n) computations in C++ access data directly by using pointers.

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
