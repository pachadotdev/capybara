#include <armadillo.hpp>
#include <cpp11.hpp>
#include <cpp11armadillo.hpp>

// #include <iostream>

using namespace arma;
using namespace cpp11;

// used across the scripts

Mat<double> center_variables_(const Mat<double> &V, const Col<double> &w,
                              const list &klist, const double &tol,
                              const int &maxiter);

Col<double> solve_beta_(const Mat<double> &MX, const Mat<double> &MNU,
                        const Col<double> &w);

Col<double> solve_eta_(const Mat<double> &MX, const Mat<double> &MNU,
                       const Col<double> &nu, const Col<double> &beta);

Mat<double> crossprod_(const Mat<double> &X, const Col<double> &w, const int &n,
                       const int &p, const bool &weighted,
                       const bool &root_weights);

std::string tidy_family_(const std::string &family);

Col<double> link_inv_(const Col<double> &eta, const std::string &fam);

double dev_resids_(const Col<double> &y, const Col<double> &mu,
                   const double &theta, const Col<double> &wt,
                   const std::string &fam);

Col<double> mu_eta_(Col<double> &eta, const std::string &fam);

Col<double> variance_(const Col<double> &mu, const double &theta,
                      const std::string &fam);

bool valid_eta_(const Col<double> &eta, const std::string &fam);

bool valid_mu_(const Col<double> &mu, const std::string &fam);
