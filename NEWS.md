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
