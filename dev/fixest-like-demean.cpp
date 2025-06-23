/*
     _                                       _
    | |                                     (_)
  __| |  ___  _ __ ___    ___   __ _  _ __   _  _ __    __ _
 / _` | / _ \| '_ ` _ \  / _ \ / _` || '_ \ | || '_ \  / _` |
| (_| ||  __/| | | | | ||  __/| (_| || | | || || | | || (_| |
 \__,_| \___||_| |_| |_| \___| \__,_||_| |_||_||_| |_| \__, |
                                                        __/ |
                                                       |___/

Original Author: Laurent R. Berge
Refactored by Mauricio "Pacha" Vargas Sepulveda starting in Jun 2022

Workhorse for feols and feglm.

It demeans any variable given in input, the algortihm is not
a real demeaning algorithm.
It is in fact identical as obtaining the optimal set of
cluster coefficients (i.e. fixed-effects) in a ML context with
a Gaussian likelihood (as described in Berge, 2018).

This way we can leverage the powerful Irons and Tuck acceleration
algorithm. For simple cases, this doesn't matter so much.
But for messy data (employee-company for instance), it matters
quite a bit.

In terms of functionality it accommodate weights and coefficients
with varying slopes.

Of course any input is **strongly** checked before getting into
this function.

I had to apply a trick to accommodate user interrupt in a
parallel setup. It costs a bit, but it's clearly worth it.
*/

/*
CONVENTIONS

Suffixes
_Q: the vector is of length Q
_I (for Identifier): the vector is of length the total number of FE identifiers
_T: a scalar representing the Total of something, usually the sum of a _Q object
_C: the vector is of length the number of coefficients
VS_C: the vector is of length the number of coefficients for the varying slopes
noVS_C: the vector is of length the number of coefficients for regular
fixed-effects (no VS!) _N: the vector is of length n_obs

PREFIXES
p_ means a pointer
nb_id: refers to the fixed-effects IDs
nb_coef: refers to the fixed-effects coefficients. Coef and id are identical in
**absence** of varying slopes. vs: means varying slopes
 */

// TODO: Next version => clean c++ code, use only sMat

#include "00_common.hpp"

#pragma once

// DEMEANING

// classes

// class that handles varying types of SEXP and behaves as a
// regular matrix
class sVec {
  double *p_dbl = nullptr;
  int *p_int = nullptr;

public:
  bool is_int = false;

  sVec(){};
  sVec(SEXP);
  sVec(double *p_x) : p_dbl(p_x), is_int(false){};
  sVec(int *p_x) : p_int(p_x), is_int(true){};
  sVec(std::nullptr_t){};

  double operator[](int i) {
    if (is_int)
      return static_cast<double>(p_int[i]);
    return p_dbl[i];
  }
};

class sMat {
  std::vector<sVec> p_sVec;
  int n = 0;
  int K = 0;

  sMat() = delete;

public:
  sMat(SEXP, bool);

  int nrow() { return n; };
  int ncol() { return K; };

  sVec operator[](int);
  double operator()(int, int);
};

class FEClass {
  int Q;
  int n_obs;
  bool is_weight;
  bool is_slope;

  // Dense vectors that we populate in the class, and their associated pointers
  std::vector<double> eq_systems_VS_C;
  std::vector<double *> p_eq_systems_VS_C;

  std::vector<double> sum_weights_noVS_C;
  std::vector<double *> p_sum_weights_noVS_C;

  // p_fe_id: pointers to the fe_id vectors
  // p_vs_vars: pointers to the VS variables
  // p_weights: pointer to the weight vector
  // eq_systems_VS_C: vector stacking all the systems of equations (each system
  // is of size n_coef * n_vs * n_vs) p_eq_systems_VS_C: pointer to the right
  // equation system. Of length Q.
  std::vector<int *> p_fe_id;
  std::vector<sVec> p_vs_vars;
  double *p_weights = nullptr;

  std::vector<bool> is_slope_Q;
  std::vector<bool> is_slope_fe_Q;

  std::vector<int> nb_vs_Q;
  std::vector<int> nb_vs_noFE_Q;

  int *nb_id_Q;

  std::vector<int> coef_start_Q;

  // internal functions
  void compute_fe_coef_internal(int, double *, bool, sVec, double *, double *);
  void compute_fe_coef_2_internal(double *, double *, double *, bool);
  void add_wfe_coef_to_mu_internal(int, double *, double *, bool);

public:
  // Utility class: Facilitates the access to the VS variables
  class simple_mat_of_vs_vars {
    int K_fe;
    std::vector<sVec> pvars;

  public:
    simple_mat_of_vs_vars(const FEClass *, int);
    double operator()(int, int);
  };

  int nb_coef_T;
  std::vector<int> nb_coef_Q;

  // constructor:
  FEClass(int n_obs, int Q, SEXP r_weights, SEXP fe_id_list, SEXP r_nb_id_Q,
          SEXP table_id_I, SEXP slope_flag_Q, SEXP slope_vars_list);

  // functions
  void compute_fe_coef(double *fe_coef, sVec &mu_in_N);
  void compute_fe_coef(int q, double *fe_coef, double *sum_other_coef_N,
                       double *in_out_C);

  void add_wfe_coef_to_mu(int q, double *fe_coef_C, double *out_N);
  void add_fe_coef_to_mu(int q, double *fe_coef_C, double *out_N);

  void compute_fe_coef_2(double *fe_coef_in_C, double *fe_coef_out_C,
                         double *fe_coef_tmp, double *in_out_C);

  void add_2_fe_coef_to_mu(double *fe_coef_a, double *fe_coef_b,
                           double *in_out_C, double *out_N, bool update_beta);

  void compute_in_out(int q, double *in_out_C, sVec &in_N, double *out_N);
};

// Now we start a big chunk => computing the varying slopes coefficients
// That's a big job. To simplify it, I created the class FEClass that takes care
// of it.
class simple_mat_with_id {
  // => Access to one of the n_coef matrices of size n_vs x n_vs; all stacked in
  // a single vector
  //
  // px0: origin of the vector (which is of length n_coef * n_vs * n_vs)
  // px_current: working location of the n_vs x n_vs matrix
  // n: n_vs
  // n2: explicit
  // id_current: current identifier. The identifiers range from 0 to (n_coef -
  // 1)

  simple_mat_with_id() = delete;

  double *px0;
  double *px_current;
  int nrow, ncol, n_total, id_current = 0;

public:
  simple_mat_with_id(double *px_in, int nrow_in)
      : px0(px_in), px_current(px_in), nrow(nrow_in), ncol(nrow_in),
        n_total(nrow * ncol){};
  simple_mat_with_id(double *px_in, int nrow_in, int ncol_in)
      : px0(px_in), px_current(px_in), nrow(nrow_in), ncol(ncol_in),
        n_total(nrow * ncol){};
  double &operator()(int id, int i, int j);
  double &operator()(int id, int i);
};

// list of objects, used to
// lighten the writting of the functions
struct PARAM_DEMEAN {
  int n_obs;
  int Q;
  int nb_coef_T;
  int iterMax;
  double diffMax;

  int algo_extraProj;
  int algo_iter_warmup;
  int algo_iter_projAfterAcc;
  int algo_iter_grandAcc;

  // iterations
  int *p_iterations_all;

  // vectors of pointers
  std::vector<sVec> p_input;
  std::vector<double *> p_output;

  // saving the fixed effects
  bool save_fixef;
  double *fixef_values;

  // FE information
  FEClass *p_FE_info;

  // stopflag
  bool *stopnow;
  int *jobdone;
};

// functions

std::vector<int> set_parallel_scheme_ter(int N, int nthreads);

void compute_fe_gnl(double *p_fe_coef_origin, double *p_fe_coef_destination,
                    double *p_sum_other_means, double *p_sum_in_out,
                    PARAM_DEMEAN *args);

void stayIdleCheckingInterrupt(bool *stopnow, std::vector<int> &jobdone,
                               int n_vars, int *counterInside);

sVec::sVec(SEXP x) {
  if (TYPEOF(x) == REALSXP) {
    is_int = false;
    p_dbl = REAL(x);
  } else if (TYPEOF(x) == INTSXP) {
    is_int = true;
    p_int = INTEGER(x);
  } else {
    stop("The current SEXP type is not supported by the sVec class.");
  }
}

sMat::sMat(SEXP x, bool single_obs = false) {
  if (TYPEOF(x) == VECSXP) {
    // x can be a list of either vectors or matrices

    int L = Rf_length(x);

    for (int l = 0; l < L; ++l) {
      SEXP xx = VECTOR_ELT(x, l);
      SEXP dim = Rf_getAttrib(xx, R_DimSymbol);

      int n_tmp = 0, K_tmp = 0;

      if (Rf_length(dim) == 0) {
        // vector
        n_tmp = Rf_length(xx);
        K_tmp = 1;
      } else {
        int *pdim = INTEGER(dim);
        n_tmp = pdim[0];
        K_tmp = pdim[1];
      }

      // we set the number of rows at the first iteration
      if (l == 0) {
        n = n_tmp;
      } else {
        if (n != n_tmp)
          stop("When setting up the class sMat: The number of observations in "
               "the list is not coherent across columns.");
      }

      K += K_tmp;

      if (TYPEOF(xx) == REALSXP) {
        double *p_x = REAL(xx);
        for (int k = 0; k < K_tmp; ++k) {
          p_sVec.push_back(sVec(p_x));
          if (k + 1 < K_tmp)
            p_x += n;
        }
      } else if (TYPEOF(xx) == INTSXP) {
        int *p_x = INTEGER(xx);
        for (int k = 0; k < K_tmp; ++k) {
          p_sVec.push_back(sVec(p_x));
          if (k + 1 < K_tmp)
            p_x += n;
        }
      } else {
        stop("The current SEXP type is not supported by the sMat class.");
      }
    }
  } else {
    // Matrix or vector

    SEXP dim = Rf_getAttrib(x, R_DimSymbol);

    if (Rf_length(dim) == 0) {
      // vector
      n = Rf_length(x);
      K = 1;
    } else {
      const int *pdim = INTEGER(dim);
      n = pdim[0];
      K = pdim[1];
    }

    if (!single_obs && (n == 1 && K == 1)) {
      // => absence of data
      n = 0;
      K = 0;
    } else if (TYPEOF(x) == REALSXP) {
      double *p_x = REAL(x);
      for (int k = 0; k < K; ++k) {
        p_sVec.push_back(sVec(p_x));
        if (k + 1 < K)
          p_x += n;
      }
    } else if (TYPEOF(x) == INTSXP) {
      int *p_x = INTEGER(x);
      for (int k = 0; k < K; ++k) {
        p_sVec.push_back(sVec(p_x));
        if (k + 1 < K)
          p_x += n;
      }
    } else {
      stop("The current SEXP type is not supported by the sMat class.");
    }
  }
}

FEClass::simple_mat_of_vs_vars::simple_mat_of_vs_vars(const FEClass *FE_info,
                                                      int q) {
  // We set up the matrix
  int start = 0;
  for (int l = 0; l < q; ++l) {
    start += FE_info->nb_vs_noFE_Q[l];
  }

  int K = FE_info->nb_vs_noFE_Q[q];
  pvars.resize(K);
  for (int k = 0; k < K; ++k) {
    pvars[k] = FE_info->p_vs_vars[start + k];
  }

  K_fe = FE_info->is_slope_fe_Q[q] ? K : -1;
}

sVec sMat::operator[](int k) { return p_sVec[k]; }

double sMat::operator()(int i, int k) { return p_sVec[k][i]; }

double &simple_mat_with_id::operator()(int id, int i, int j) {
  if (id != id_current) {
    id_current = id;
    px_current = px0 + n_total * id;
  }

  return px_current[i + nrow * j];
}

double &simple_mat_with_id::operator()(int id, int i) {
  if (id != id_current) {
    id_current = id;
    px_current = px0 + n_total * id;
  }

  return px_current[i];
}

double FEClass::simple_mat_of_vs_vars::operator()(int i, int k) {
  if (k == K_fe) {
    return 1;
  }

  return pvars[k][i];
}

#include "02_0_demeaning.hpp"

std::vector<int> set_parallel_scheme_ter(int N, int nthreads) {
  // => this concerns only the parallel application on a 1-Dimensional matrix
  // takes in the nber of observations of the vector and the nber of threads
  // gives back a vector of the length the nber of threads + 1 giving the
  // start/stop of each threads

  std::vector<int> res(nthreads + 1, 0);
  double N_rest = N;

  for (int i = 0; i < nthreads; ++i) {
    res[i + 1] = std::ceil(N_rest / (nthreads - i));
    N_rest -= res[i + 1];
    res[i + 1] += res[i];
  }

  return res;
}

[[cpp11::register]] list cpp_which_na_inf_(SEXP x, int nthreads) {
  // x: vector, matrix, data.frame // double or integer

  /*
   This function takes a matrix and looks at whether it contains NA or infinite
   values return: flag for na/inf + logical vector of obs that are Na/inf
   std::isnan, std::isinf are OK since cpp11 required
   do_any_na_inf: if high suspicion of NA present: we go directly constructing
   the vector is_na_inf in the "best" case (default expected), we need not
   construct is_na_inf
   */

  sMat mat(x, false);

  int nobs = mat.nrow();
  int K = mat.ncol();
  bool anyNAInf = false;
  bool any_na = false;  // return value
  bool any_inf = false; // return value

  /*
   we make parallel the anyNAInf loop
   why? because we want that when there's no NA (default) it works as fast as
   possible if there are NAs, single threaded mode is faster, but then we
   circumvent with the do_any_na_inf flag
   */

  // no need to care about the race condition
  // "trick" to make a break in a multi-threaded section

  std::vector<int> bounds = set_parallel_scheme_ter(nobs, nthreads);

#pragma omp parallel for num_threads(nthreads)
  for (int t = 0; t < nthreads; ++t) {
    for (int k = 0; k < K; ++k) {
      for (int i = bounds[t]; i < bounds[t + 1] && !anyNAInf; ++i) {
        if (mat[k].is_int) {
          if (mat(i, k) == -2147483648.0) {
            anyNAInf = true;
          }
        } else if (std::isnan(mat(i, k)) || std::isinf(mat(i, k))) {
          anyNAInf = true;
        }
      }
    }
  }

  // object to return: is_na_inf
  int is_na_inf_size = anyNAInf ? nobs : 1;
  writable::logicals is_na_inf(is_na_inf_size);
  for (int i = 0; i < is_na_inf_size; ++i) {
    is_na_inf[i] = false;
  }

  if (anyNAInf) {
#pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nobs; ++i) {
      double x_tmp = 0;
      for (int k = 0; k < K; ++k) {
        x_tmp = mat(i, k);
        if (mat[k].is_int) {
          if (mat(i, k) == -2147483648.0) {
            is_na_inf[i] = true;
            any_na = true;
            break;
          }
        } else if (std::isnan(x_tmp)) {
          is_na_inf[i] = true;
          any_na = true;
          break;
        } else if (std::isinf(x_tmp)) {
          is_na_inf[i] = true;
          any_inf = true;
          break;
        }
      }
    }
  }

  // Return
  writable::list res;
  res.push_back({"any_na"_nm = any_na});
  res.push_back({"any_inf"_nm = any_inf});
  res.push_back({"any_na_inf"_nm = any_na || any_inf});
  res.push_back({"is_na_inf"_nm = is_na_inf});

  return res;
}

// Demeans each variable in input
// The method is based on obtaining the optimal cluster coefficients
// Works but only the master thread can call that function!
// What happens if the master thread has finished its job but the lower thread
// is in an "infinite" loop? this is tricky, as such we cannot stop it.
// Solution: create a function keeping the threads idle waiting for the complete
// job to be done BUT I need to add static allocation of threads => performance
// cost

// List of objects, used to
// lighten the writting of the functions

#include "02_0_demeaning.hpp"

void FEClass::add_2_fe_coef_to_mu(double *fe_coef_a, double *fe_coef_b,
                                  double *in_out_C, double *out_N,
                                  bool update_beta = true) {
  // We add the value of the FE coefficients to each observation

  // Step 1: we update the coefficients of b

  if (update_beta) {
    compute_fe_coef_2_internal(fe_coef_a, fe_coef_b, in_out_C, out_N);
  }

  // Step 2: we add the value of each coef

  for (int q = 0; q < 2; ++q) {
    double *my_fe_coef = q == 0 ? fe_coef_a : fe_coef_b;
    int *my_fe = p_fe_id[q];
    bool is_slope = is_slope_Q[q];
    int V = nb_vs_Q[q];

    simple_mat_of_vs_vars VS_mat(this, q);
    simple_mat_with_id my_vs_coef(my_fe_coef, nb_vs_Q[q], 1);

    for (int i = 0; i < n_obs; ++i) {
      if (is_slope) {
        for (int v = 0; v < V; ++v) {
          out_N[i] += my_vs_coef(my_fe[i] - 1, v) * VS_mat(i, v);
        }
      } else {
        out_N[i] += my_fe_coef[my_fe[i] - 1];
      }
    }
  }
}

void compute_fe(int Q, double *p_fe_coef_origin, double *p_fe_coef_destination,
                double *p_sum_other_means, double *p_sum_in_out,
                PARAM_DEMEAN *args) {
  if (Q == 2) {
    FEClass &FE_info = *(args->p_FE_info);
    FE_info.compute_fe_coef_2(p_fe_coef_origin, p_fe_coef_destination,
                              p_sum_other_means, p_sum_in_out);
  } else {
    compute_fe_gnl(p_fe_coef_origin, p_fe_coef_destination, p_sum_other_means,
                   p_sum_in_out, args);
  }
}

bool dm_update_X_IronsTuck(int nb_coef_no_Q, std::vector<double> &X,
                           const std::vector<double> &GX,
                           const std::vector<double> &GGX,
                           std::vector<double> &delta_GX,
                           std::vector<double> &delta2_X) {
  for (int i = 0; i < nb_coef_no_Q; ++i) {
    double GX_tmp = GX[i];
    delta_GX[i] = GGX[i] - GX_tmp;
    delta2_X[i] = delta_GX[i] - GX_tmp + X[i];
  }

  double vprod = 0, ssq = 0;
  for (int i = 0; i < nb_coef_no_Q; ++i) {
    double delta2_X_tmp = delta2_X[i];
    vprod += delta_GX[i] * delta2_X_tmp;
    ssq += delta2_X_tmp * delta2_X_tmp;
  }

  bool res = false;

  if (ssq == 0) {
    res = true;
  } else {
    double coef = vprod / ssq;

    // update of X:
    for (int i = 0; i < nb_coef_no_Q; ++i) {
      X[i] = GGX[i] - coef * delta_GX[i];
    }
  }

  return res;
}

void demean_single_1(int v, PARAM_DEMEAN *args) {
  // v: variable identifier to demean

  // Q == 1: nothing to say, just compute the closed form

  // loading the data
  int nb_coef_T = args->nb_coef_T;

  std::vector<sVec> &p_input = args->p_input;
  std::vector<double *> &p_output = args->p_output;

  // fe_info
  FEClass &FE_info = *(args->p_FE_info);

  // vector of fixed-effects coefficients initialized at 0
  std::vector<double> fe_coef(nb_coef_T, 0);
  double *p_fe_coef = fe_coef.data();

  // interruption handling
  // bool isMaster = omp_get_thread_num() == 0;
  // bool *pStopNow = args->stopnow;
  // if (isMaster) {
  //   if (pending_interrupt()) {
  //     *pStopNow = true;
  //   }
  // }

  // the input & output
  sVec &input = p_input[v];
  double *output = p_output[v];

  // We compute the FEs
  FE_info.compute_fe_coef(p_fe_coef, input);

  // Output:
  FE_info.add_fe_coef_to_mu(0, p_fe_coef, output);

  // saving the fixef coefs
  double *fixef_values = args->fixef_values;
  if (args->save_fixef) {
    for (int m = 0; m < nb_coef_T; ++m) {
      fixef_values[m] = fe_coef[m];
    }
  }
}

bool demean_acc_gnl(int v, int iterMax, PARAM_DEMEAN *args,
                    bool two_fe = false) {
  //
  // data
  //

  // fe_info
  FEClass &FE_info = *(args->p_FE_info);

  // algo info
  const int n_extraProj = args->algo_extraProj;
  const int iter_projAfterAcc = args->algo_iter_projAfterAcc;
  const int iter_grandAcc = args->algo_iter_grandAcc;

  int n_obs = args->n_obs;
  int nb_coef_T = args->nb_coef_T;
  int Q = args->Q;
  double diffMax = args->diffMax;

  int nb_coef_all = nb_coef_T;
  const bool two_fe_algo = two_fe || Q == 2;
  if (two_fe_algo) {
    Q = 2;
    // special case: 2 FEs, below will contain only the size of the first FE
    //   this is because 2-FE is a special case which allows to avoid creating
    //   a N-size vector
    nb_coef_T = FE_info.nb_coef_Q[0];
    // nb_coef_all: only used in in-out
    nb_coef_all = FE_info.nb_coef_Q[0] + FE_info.nb_coef_Q[1];
  }

  // input output
  std::vector<sVec> &p_input = args->p_input;
  std::vector<double *> &p_output = args->p_output;
  sVec &input = p_input[v];
  double *output = p_output[v];

  // temp var:
  int size_other_means = two_fe_algo ? FE_info.nb_coef_Q[1] : n_obs;
  std::vector<double> sum_other_means_or_second_coef(size_other_means);
  double *p_sum_other_means = sum_other_means_or_second_coef.data();

  // conditional sum of input minus output
  std::vector<double> sum_input_output(nb_coef_all, 0);
  double *p_sum_in_out = sum_input_output.data();

  for (int q = 0; q < Q; ++q) {
    FE_info.compute_in_out(q, p_sum_in_out, input, output);
  }

  // interruption handling
  // bool isMaster = omp_get_thread_num() == 0;
  // bool *pStopNow = args->stopnow;
  // I overcast to remember the lesson
  // rough estimate nber operation per iter
  // double flop = 4.0 * (5 + 12 * (Q - 1) + 4 * (Q - 1) * (Q - 1)) *
  //               static_cast<double>(n_obs);
  // if (two_fe_algo) {
  //   flop = 20.0 * static_cast<double>(n_obs);
  // }
  // int iterSecond =
  //     ceil(2000000000 / flop / 5);  // nber iter per 1/5 second at 2GHz

  //
  // IT iteration (preparation)
  //

  // variables on 1:nb_coef
  // note that in the case of two_fe_algo, the length is equal to the 1st FE
  // coefs
  std::vector<double> X(nb_coef_T, 0);
  std::vector<double> GX(nb_coef_T);
  std::vector<double> GGX(nb_coef_T);
  // pointers:
  double *p_X = X.data();
  double *p_GX = GX.data();
  double *p_GGX = GGX.data();

  // variables on 1:(Q-1)
  int nb_coef_no_Q = 0;
  for (int q = 0; q < (Q - 1); ++q) {
    nb_coef_no_Q += FE_info.nb_coef_Q[q];
  }
  std::vector<double> delta_GX(nb_coef_no_Q);
  std::vector<double> delta2_X(nb_coef_no_Q);

  // additional vectors used to save the coefs
  std::vector<double> Y(nb_coef_T);
  std::vector<double> GY(nb_coef_T);
  std::vector<double> GGY(nb_coef_T);
  // pointers:
  double *p_Y = Y.data();
  double *p_GY = GY.data();
  double *p_GGY = GGY.data();

  int grand_acc = 0;

  //
  // the main loop
  //

  // first iteration
  compute_fe(Q, p_X, p_GX, p_sum_other_means, p_sum_in_out, args);

  // check whether we should go into the loop
  bool keepGoing = false;
  for (int i = 0; i < nb_coef_T; ++i) {
    if (continue_criterion(X[i], GX[i], diffMax)) {
      keepGoing = true;
      break;
    }
  }

  // For the stopping criterion on total addition
  double ssr = 0;

  int iter = 0;
  bool numconv = false;
  // while (!*pStopNow && keepGoing && iter < iterMax) {
  while (keepGoing && iter < iterMax) {
    // if (isMaster && iter % iterSecond == 0) {
    //   if (pending_interrupt()) {
    //     *pStopNow = true;
    //     break;
    //   }
    // }

    iter++;

    for (int rep = 0; rep < n_extraProj; ++rep) {
      // simple projections, at the request of the user
      // may be useful to hasten convergence on special cases
      // default is 0
      compute_fe(Q, p_GX, p_GGX, p_sum_other_means, p_sum_in_out, args);
      compute_fe(Q, p_GGX, p_X, p_sum_other_means, p_sum_in_out, args);
      compute_fe(Q, p_X, p_GX, p_sum_other_means, p_sum_in_out, args);
    }

    // origin: GX, destination: GGX
    compute_fe(Q, p_GX, p_GGX, p_sum_other_means, p_sum_in_out, args);

    // X: outcome of the acceleration
    numconv =
        dm_update_X_IronsTuck(nb_coef_no_Q, X, GX, GGX, delta_GX, delta2_X);
    if (numconv)
      break;

    if (iter >= iter_projAfterAcc) {
      memcpy(p_Y, p_X, nb_coef_T * sizeof(double));
      compute_fe(Q, p_Y, p_X, p_sum_other_means, p_sum_in_out, args);
    }

    // origin: X, destination: GX
    compute_fe(Q, p_X, p_GX, p_sum_other_means, p_sum_in_out, args);

    keepGoing = false;
    for (int i = 0; i < nb_coef_no_Q; ++i) {
      if (continue_criterion(X[i], GX[i], diffMax)) {
        keepGoing = true;
        break;
      }
    }

    if (iter % iter_grandAcc == 0) {
      ++grand_acc;
      if (grand_acc == 1) {
        memcpy(p_Y, p_GX, nb_coef_T * sizeof(double));
      } else if (grand_acc == 2) {
        memcpy(p_GY, p_GX, nb_coef_T * sizeof(double));
      } else {
        memcpy(p_GGY, p_GX, nb_coef_T * sizeof(double));
        numconv =
            dm_update_X_IronsTuck(nb_coef_no_Q, Y, GY, GGY, delta_GX, delta2_X);
        if (numconv)
          break;
        compute_fe(Q, p_Y, p_GX, p_sum_other_means, p_sum_in_out, args);
        grand_acc = 0;
      }
    }

    // Other stopping criterion: change to SSR very small
    if (iter % 40 == 0) {
      // mu_current is the vector of means
      std::vector<double> mu_current(n_obs, 0);
      double *p_mu = mu_current.data();

      if (two_fe_algo) {
        FE_info.add_2_fe_coef_to_mu(p_GX, p_sum_other_means, p_sum_in_out,
                                    p_mu);
      } else {
        for (int q = 0; q < Q; ++q) {
          FE_info.add_fe_coef_to_mu(q, p_GX, p_mu);
        }
      }

      double ssr_old = ssr;

      // we compute the new SSR
      ssr = 0;
      double resid;
      for (int i = 0; i < n_obs; ++i) {
        resid = input[i] - mu_current[i];
        ssr += resid * resid;
      }

      if (stopping_criterion(ssr_old, ssr, diffMax)) {
        break;
      }
    }
  }

  //
  // Updating the output
  //

  double *p_beta_final = nullptr;
  if (two_fe_algo) {
    // we end with a last iteration
    double *p_alpha_final = p_GX;
    p_beta_final = p_sum_other_means;

    FE_info.compute_fe_coef_2(p_alpha_final, p_alpha_final, p_beta_final,
                              p_sum_in_out);
    FE_info.add_2_fe_coef_to_mu(p_alpha_final, p_beta_final, p_sum_in_out,
                                output, false);
  } else {
    for (int q = 0; q < Q; ++q) {
      FE_info.add_fe_coef_to_mu(q, p_GX, output);
    }
  }

  // keeping track of iterations
  int *iterations_all = args->p_iterations_all;
  iterations_all[v] += iter;

  // saving the fixef coefs
  double *fixef_values = args->fixef_values;
  if (args->save_fixef) {
    for (int m = 0; m < nb_coef_T; ++m) {
      fixef_values[m] += GX[m];
    }

    if (two_fe_algo) {
      // we add the other coefficients
      int n_coefs_FE1 = nb_coef_T;
      int n_coefs_FE2 = size_other_means;
      for (int m = 0; m < n_coefs_FE2; ++m) {
        fixef_values[n_coefs_FE1 + m] += p_beta_final[m];
      }
    }
  }

  bool conv = iter == iterMax ? false : true;

  return (conv);
}

void demean_single_gnl(int v, PARAM_DEMEAN *args) {
  // v: integer identifying the variable to demean

  // Algorithm to quickly get the means
  // Q >= 3 => acceleration for 15 iter
  // if no convergence: conv 2 FEs
  // then acceleration again

  // data
  int iterMax = args->iterMax;

  // Note that the historical default was 15 iterations (still the case on
  // 2024-02-14)
  int iter_warmup = args->algo_iter_warmup;
  int Q = args->Q;

  if (Q == 2) {
    demean_acc_gnl(v, iterMax, args);
  } else {
    bool conv = false;

    // iter_warmup <= 0 means no warmup and we start directly with 2FE algo
    if (iter_warmup > 0) {
      conv = demean_acc_gnl(v, iter_warmup, args);
      iter_warmup = 0;
    }

    if (conv == false && iterMax > iter_warmup) {
      // convergence for the first 2 FEs
      int iter_max_2FE = iterMax / 2 - iter_warmup;
      if (iter_max_2FE > 0) {
        demean_acc_gnl(v, iter_max_2FE, args, true);
      }

      // re-acceleration
      int iter_previous = args->p_iterations_all[v];
      demean_acc_gnl(v, iterMax - iter_previous, args);
    }
  }

  int *jobdone = args->jobdone;
  jobdone[v] = 1;
}

// Loop over demean_single
[[cpp11::register]] list
cpp_demean_(SEXP y, SEXP X_raw, SEXP r_weights, int iterMax, double diffMax,
            SEXP r_nb_id_Q, SEXP fe_id_list, SEXP table_id_I, SEXP slope_flag_Q,
            SEXP slope_vars_list, SEXP r_init, int nthreads, int algo_extraProj,
            int algo_iter_warmup, int algo_iter_projAfterAcc,
            int algo_iter_grandAcc, bool save_fixef) {
  // main fun that calls demean_single
  // preformats all the information needed on the fixed-effects
  // y: the dependent variable
  // X_raw: the matrix of the explanatory variables -- can be "empty"

  // when including weights: recreate table values
  // export weights and is_weight bool

  // slope_flag: whether a FE is a varying slope
  // slope_var: the associated variables with varying slopes

  // initial variables
  int Q = Rf_length(r_nb_id_Q);

  // info on y
  sMat m_y(y, false);
  int n_vars_y = m_y.ncol();
  bool useY = n_vars_y > 0;
  bool is_y_list = n_vars_y > 1 || TYPEOF(y) == VECSXP;
  int n_obs = m_y.nrow();

  // info on X
  sMat m_X(X_raw, false);
  int n_vars_X = m_X.ncol();
  if (useY == false) {
    n_obs = m_X.nrow();
  }
  bool useX = n_vars_X > 0;

  if (n_obs == 0 || n_obs == 1) {
    // The data set is of length 1!!!!
    n_obs = 1;

    m_y = sMat(y, true);
    n_vars_y = m_y.ncol();
    useY = true;

    m_X = sMat(X_raw, true);
    n_vars_X = m_X.ncol();
    useX = n_vars_X > 0;
  }

  int n_vars = n_vars_y + n_vars_X;

  // initialisation if needed (we never initialize when only one FE, except if
  // asked explicitly)
  bool isInit = Rf_xlength(r_init) != 1 && Q > 1;
  double *init = REAL(r_init);
  bool saveInit = ((isInit || init[0] != 0) && Q > 1) || init[0] == 666;

  // Creating the object containing all information on the FEs
  FEClass FE_info(n_obs, Q, r_weights, fe_id_list, r_nb_id_Q, table_id_I,
                  slope_flag_Q, slope_vars_list);
  int nb_coef_T = FE_info.nb_coef_T;

  // output vector: (Note that if the means are provided, we use that vector and
  // will modify it in place)
  int64_t n_total = static_cast<int64_t>(n_obs) * n_vars;
  SEXP output_values = PROTECT(Rf_allocVector(REALSXP, isInit ? 1 : n_total));

  double *p_output_origin = isInit ? init : REAL(output_values);
  if (isInit == false) {
    std::fill_n(p_output_origin, n_total, 0);
  }

  // vector of pointers: input/output

  std::vector<double *> p_output(n_vars);
  p_output[0] = p_output_origin;
  for (int v = 1; v < n_vars; v++) {
    p_output[v] = p_output[v - 1] + n_obs;
  }

  std::vector<sVec> p_input(n_vars);

  for (int k = 0; k < n_vars_X; ++k) {
    p_input[k] = m_X[k];
  }

  for (int k = 0; k < n_vars_y; ++k) {
    p_input[n_vars_X + k] = m_y[k];
  }

  // keeping track of iterations
  std::vector<int> iterations_all(n_vars, 0);
  int *p_iterations_all = iterations_all.data();

  // save fixef option
  if (useX && save_fixef) {
    stop("save_fixef can only be used when there is no Xs.");
  }

  std::vector<double> fixef_values(save_fixef ? nb_coef_T : 1, 0);
  double *p_fixef_values = fixef_values.data();

  // Sending variables to envir

  PARAM_DEMEAN args;

  args.n_obs = n_obs;
  args.iterMax = iterMax;
  args.diffMax = diffMax;
  args.Q = Q;
  args.nb_coef_T = nb_coef_T;
  args.p_input = p_input;
  args.p_output = p_output;
  args.p_iterations_all = p_iterations_all;
  args.algo_extraProj = algo_extraProj;
  args.algo_iter_warmup = algo_iter_warmup;
  args.algo_iter_projAfterAcc = algo_iter_projAfterAcc;
  // negative number or 0 means never
  args.algo_iter_grandAcc =
      algo_iter_grandAcc <= 0 ? 1000000 : algo_iter_grandAcc;

  // save FE_info
  args.p_FE_info = &FE_info;

  // save fixef:
  args.save_fixef = save_fixef;
  args.fixef_values = p_fixef_values;

  // stopping flag + indicator that job is finished
  bool stopnow = false;
  args.stopnow = &stopnow;
  std::vector<int> jobdone(n_vars, 0);
  int *pjobdone = jobdone.data();
  args.jobdone = pjobdone;

  int counter = 0;
  int *pcounter = &counter;

  // main loop

  int nthreads_current = nthreads > n_vars ? n_vars : nthreads;

#pragma omp parallel for num_threads(nthreads_current) schedule(static, 1)
  for (int v = 0; v < (n_vars + nthreads_current); ++v) {
    // demean_single is the workhorse
    // you get the "mean"

    if (!*(args.stopnow)) {
      if (v < n_vars) {
        if (Q == 1) {
          demean_single_1(v, &args);
        } else {
          demean_single_gnl(v, &args);
        }
      } else if (true && Q != 1) {
        // stayIdleCheckingInterrupt(&stopnow, jobdone, n_vars, pcounter);
      }
    }
  }

  if (*(args.stopnow)) {
    UNPROTECT(1);
    stop("cpp_demean: User interrupt.");
  }

  // save

  writable::list res; // a vector and a matrix

  int nrow = useX ? n_obs : 1;
  int ncol = useX ? n_vars_X : 1;
  writable::doubles_matrix<> X_demean(nrow, ncol);

  sVec p_input_tmp;
  double *p_output_tmp;
  for (int k = 0; k < n_vars_X; ++k) {
    p_input_tmp = p_input[k];
    p_output_tmp = p_output[k];

    for (int i = 0; i < n_obs; ++i) {
      X_demean(i, k) = p_input_tmp[i] - p_output_tmp[i];
    }
  }

  res.push_back({"X_demean"_nm = X_demean});

  if (is_y_list && useY) {
    writable::list y_demean(n_vars_y);

    for (int v = 0; v < n_vars_y; ++v) {
      p_input_tmp = p_input[n_vars_X + v];
      p_output_tmp = p_output[n_vars_X + v];

      writable::doubles y_demean_tmp(n_obs);
      for (int i = 0; i < n_obs; ++i) {
        y_demean_tmp[i] = p_input_tmp[i] - p_output_tmp[i];
      }

      y_demean[v] = y_demean_tmp;
    }

    res.push_back({"y_demean"_nm = y_demean});
  } else {
    writable::doubles y_demean(useY ? n_obs : 1);
    if (useY) {
      // y is always the last variable
      p_input_tmp = p_input[n_vars - 1];
      p_output_tmp = p_output[n_vars - 1];
      for (int i = 0; i < n_obs; ++i) {
        y_demean[i] = p_input_tmp[i] - p_output_tmp[i];
      }
    }
    res.push_back({"y_demean"_nm = y_demean});
  }

  // iterations
  writable::integers iter_final(n_vars);
  for (int v = 0; v < n_vars; ++v) {
    iter_final[v] = p_iterations_all[v];
  }

  res.push_back({"iterations"_nm = iter_final});

  // if save is requested
  if (saveInit) {
    if (isInit) {
      res.push_back({"means"_nm = r_init});
    } else {
      res.push_back({"means"_nm = output_values});
    }
  } else {
    res.push_back({"means"_nm = 0.0});
  }

  // save fixef coef
  if (save_fixef) {
    res.push_back({"fixef_coef"_nm = fixef_values});
  }

  UNPROTECT(1);

  return res;
}
