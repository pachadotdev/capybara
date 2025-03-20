#include <armadillo.hpp>
#include <cpp11.hpp>
#include <cpp11armadillo.hpp>
#include <regex>
#include <unordered_map>

using namespace arma;
using namespace cpp11;

// used across the scripts

void center_variables_(mat &V, const vec &w, const list &klist,
                       const double &tol, const int &max_iter,
                       const int &iter_interrupt);

vec solve_beta_(mat MX, const mat &MNU, const vec &w);

vec solve_eta_(const mat &MX, const mat &MNU, const vec &nu, const vec &beta);

mat crossprod_(const mat &X, const vec &w);

std::string tidy_family_(const std::string &family);

vec link_inv_(const vec &eta, const std::string &fam);

double dev_resids_(const vec &y, const vec &mu, const double &theta,
                   const vec &wt, const std::string &fam);

vec mu_eta_(const vec &eta, const std::string &fam);

vec variance_(const vec &mu, const double &theta, const std::string &fam);

bool valid_eta_(const vec &eta, const std::string &fam);

bool valid_mu_(const vec &mu, const std::string &fam);
