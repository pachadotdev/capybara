#include <cpp11.hpp>
#include <cpp11armadillo.hpp>
#include <regex>
#include <unordered_map>

using namespace arma;
using namespace cpp11;

// used across the scripts

void center_variables_(mat &V, const vec &w, const list &klist,
                       const double &tol, const size_t &maxiter,
                       const size_t &interrupt_iter);

vec solve_beta_(mat MX, const mat &MNU, const vec &w);

vec solve_eta_(const mat &MX, const mat &MNU, const vec &nu, const vec &beta);

mat crossprod_(const mat &X, const vec &w);

// Enum for GLM family types
enum FamilyType {
  GAUSSIAN,
  POISSON,
  BINOMIAL,
  GAMMA,
  INV_GAUSSIAN,
  NEG_BIN,
  UNKNOWN
};

std::string tidy_family_(const std::string &family);

FamilyType get_family_type(const std::string &fam);

vec link_inv_(const vec &eta, const FamilyType &fam);

double dev_resids_(const vec &y, const vec &mu, const double &theta,
                   const vec &wt, const FamilyType &fam);

vec mu_eta_(const vec &eta, const FamilyType &fam);

vec variance_(const vec &mu, const double &theta, const FamilyType &fam);

bool valid_eta_(const vec &eta, const FamilyType &fam);

bool valid_mu_(const vec &mu, const FamilyType &fam);
