#ifndef CAPYBARA_CENTER
#define CAPYBARA_CENTER

// Stopping/continuing criteria from fixest demeaning.cpp - exact implementation
inline bool continue_crit(double a, double b, double diffMax) {
  // continuing criterion of the algorithm
  double diff = std::abs(a - b);
  return ((diff > diffMax) && (diff / (0.1 + std::abs(a)) > diffMax));
}

inline bool stopping_crit(double a, double b, double diffMax) {
  // stopping criterion of the algorithm
  double diff = std::abs(a - b);
  return ((diff < diffMax) || (diff / (0.1 + std::abs(a)) < diffMax));
}

// Forward declaration
class FEClass;

// Parameter structure for demeaning algorithm
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

  // iterations tracking
  int *p_iterations_all;

  // input/output data
  std::vector<vec> p_input;
  std::vector<double *> p_output;

  // saving the fixed effects
  bool save_fixef;
  double *fixef_values;

  // FE information
  FEClass *p_FE_info;

  // stop flags
  bool *stopnow;
  int *jobdone;
};

// Main FE class
class FEClass {
  int Q;
  int n_obs;
  bool is_weight;
  bool is_slope;

  // Dense vectors and their pointers
  std::vector<double> eq_systems_VS_C;
  std::vector<double *> p_eq_systems_VS_C;

  std::vector<double> sum_weights_noVS_C;
  std::vector<double *> p_sum_weights_noVS_C;

  // FE structure
  std::vector<int *> p_fe_id; // pointers to FE id vectors
  std::vector<vec> p_vs_vars; // varying slope variables
  double *p_weights;          // pointer to weights

  std::vector<bool> is_slope_Q;
  std::vector<bool> is_slope_fe_Q;
  std::vector<int> nb_vs_Q;
  std::vector<int> nb_vs_noFE_Q;

  int *nb_id_Q;
  std::vector<int> coef_start_Q;

  // Armadillo storage for FE IDs
  std::vector<std::vector<int>> fe_id_storage;

  // Storage for converted int arrays
  std::vector<int> nb_id_Q_storage;
  std::vector<int> table_id_storage;

  // Internal helper class for accessing VS variables
  class simple_mat_of_vs_vars {
    int K_fe;
    std::vector<vec> pvars;

  public:
    simple_mat_of_vs_vars(const FEClass *, int);
    double operator()(int, int);
  };

  // Internal functions
  void compute_fe_coef_internal(int, double *, bool, vec *, double *, double *);
  void compute_fe_coef_2_internal(double *, double *, double *, bool);
  void add_wfe_coef_to_mu_internal(int, double *, double *, bool);

public:
  int nb_coef_T;
  std::vector<int> nb_coef_Q;

  // Constructor
  FEClass(int n_obs, int Q, const vec &weights, const field<ivec> &fe_id_list,
          const ivec &r_nb_id_Q, const ivec &table_id_I,
          const ivec &slope_flag_Q, const field<mat> &slope_vars_list);

  // Core functions
  void compute_fe_coef(double *fe_coef_C, vec &mu_in_N);
  void compute_fe_coef(int q, double *fe_coef_C, double *sum_other_coef_N,
                       double *in_out_C);

  void add_wfe_coef_to_mu(int q, double *fe_coef_C, double *out_N);
  void add_fe_coef_to_mu(int q, double *fe_coef_C, double *out_N);

  void compute_fe_coef_2(double *fe_coef_in_C, double *fe_coef_out_C,
                         double *fe_coef_tmp, double *in_out_C);

  void add_2_fe_coef_to_mu(double *fe_coef_a, double *fe_coef_b,
                           double *in_out_C, double *out_N,
                           bool update_beta = true);

  void compute_in_out(int q, double *in_out_C, vec &in_N, double *out_N);
};

// Constructor implementation
inline FEClass::FEClass(int n_obs, int Q, const vec &weights,
                        const field<ivec> &fe_id_list, const ivec &r_nb_id_Q,
                        const ivec &table_id_I, const ivec &slope_flag_Q,
                        const field<mat> &slope_vars_list) {
  this->n_obs = n_obs;
  this->Q = Q;

  // Store FE IDs and set up pointers - convert from sword to int
  fe_id_storage.resize(Q);
  p_fe_id.resize(Q);
  for (int q = 0; q < Q; ++q) {
    const ivec &fe_q = fe_id_list(q);
    fe_id_storage[q].resize(fe_q.n_elem);
    for (size_t i = 0; i < fe_q.n_elem; ++i) {
      fe_id_storage[q][i] = static_cast<int>(fe_q(i));
    }
    p_fe_id[q] = fe_id_storage[q].data();
  }

  // Copy nb_id_Q data - convert from sword to int and store as member
  nb_id_Q_storage.resize(Q);
  for (int q = 0; q < Q; ++q) {
    nb_id_Q_storage[q] = static_cast<int>(r_nb_id_Q(q));
  }
  nb_id_Q = nb_id_Q_storage.data();

  // Initialize weights
  is_weight = weights.n_elem > 1;
  if (is_weight) {
    // Note: this creates a potential dangling pointer issue
    // In practice, caller must ensure weights lifetime
    p_weights = const_cast<double *>(weights.memptr());
  } else {
    p_weights = nullptr;
  }

  // Slope flags
  int nb_slopes = 0;
  std::vector<bool> is_slope_Q_vec(Q, false);
  std::vector<bool> is_slope_fe_Q_vec(Q, false);
  std::vector<int> nb_vs_Q_vec(Q, 0);
  std::vector<int> nb_vs_noFE_Q_vec(Q, 0);
  std::vector<int> nb_coef_Q_vec(Q);
  int nb_coef_T_tmp = 0;

  for (int q = 0; q < Q; ++q) {
    int sf = static_cast<int>(slope_flag_Q(q));
    if (sf != 0) {
      nb_slopes += std::abs(sf);
      is_slope_Q_vec[q] = true;
      nb_vs_Q_vec[q] = std::abs(sf);
      nb_vs_noFE_Q_vec[q] = std::abs(sf);

      if (sf > 0) {
        ++nb_vs_Q_vec[q];
        is_slope_fe_Q_vec[q] = true;
      }

      nb_coef_Q_vec[q] = nb_vs_Q_vec[q] * nb_id_Q_storage[q];
    } else {
      nb_coef_Q_vec[q] = nb_id_Q_storage[q];
    }

    nb_coef_T_tmp += nb_coef_Q_vec[q];
  }

  // Coefficient starting positions
  std::vector<int> coef_start_Q_vec(Q, 0);
  for (int q = 1; q < Q; ++q) {
    coef_start_Q_vec[q] = coef_start_Q_vec[q - 1] + nb_coef_Q_vec[q - 1];
  }

  // Copy to member variables
  this->is_slope_Q = is_slope_Q_vec;
  this->is_slope_fe_Q = is_slope_fe_Q_vec;
  this->nb_vs_Q = nb_vs_Q_vec;
  this->nb_vs_noFE_Q = nb_vs_noFE_Q_vec;
  this->nb_coef_Q = nb_coef_Q_vec;
  this->nb_coef_T = nb_coef_T_tmp;
  this->coef_start_Q = coef_start_Q_vec;

  is_slope = nb_slopes > 0;

  // Non-slope coefficients
  int nb_coef_noVS_T = 0;
  for (int q = 0; q < Q; ++q) {
    if (!is_slope_Q_vec[q]) {
      nb_coef_noVS_T += nb_id_Q_storage[q];
    }
  }

  sum_weights_noVS_C.resize(nb_coef_noVS_T > 0 ? nb_coef_noVS_T : 1);
  std::fill(sum_weights_noVS_C.begin(), sum_weights_noVS_C.end(), 0);

  p_sum_weights_noVS_C.resize(Q);
  p_sum_weights_noVS_C[0] = sum_weights_noVS_C.data();
  for (int q = 1; q < Q; ++q) {
    p_sum_weights_noVS_C[q] =
        p_sum_weights_noVS_C[q - 1] +
        (is_slope_Q_vec[q - 1] ? 0 : nb_id_Q_storage[q - 1]);
  }

  // Table pointers
  table_id_storage.resize(table_id_I.n_elem);
  for (size_t i = 0; i < table_id_I.n_elem; ++i) {
    table_id_storage[i] = static_cast<int>(table_id_I(i));
  }

  std::vector<int *> p_table_id_I(Q);
  p_table_id_I[0] = table_id_storage.data();
  for (int q = 1; q < Q; ++q) {
    p_table_id_I[q] = p_table_id_I[q - 1] + nb_id_Q_storage[q - 1];
  }

  // Sum of weights for non-slope FEs
  for (int q = 0; q < Q; ++q) {
    if (is_slope_Q_vec[q])
      continue;

    double *my_SW = p_sum_weights_noVS_C[q];

    if (is_weight) {
      int *my_fe = p_fe_id[q];
      for (int obs = 0; obs < n_obs; ++obs) {
        my_SW[my_fe[obs] - 1] += p_weights[obs];
      }
    } else {
      int nb_coef = nb_id_Q_storage[q];
      int *my_table = p_table_id_I[q];
      for (int i = 0; i < nb_coef; ++i) {
        my_SW[i] = my_table[i];
      }
    }
  }

  if (is_weight && nb_coef_noVS_T > 0) {
    // Check for zero weights
    for (int c = 0; c < nb_coef_noVS_T; ++c) {
      if (sum_weights_noVS_C[c] == 0) {
        sum_weights_noVS_C[c] = 1;
      }
    }
  }

  // Setup varying slopes if needed
  if (nb_slopes > 0) {
    p_vs_vars.resize(nb_slopes);
    for (int v = 0; v < nb_slopes; ++v) {
      p_vs_vars[v] = vectorise(slope_vars_list(v));
    }

    // Setup equation systems (simplified version)
    // TODO: Full slope setup implementation would follow original pattern
    // but is complex and omitted for brevity
    int nb_vs_coef_T = 0;
    for (int q = 0; q < Q; ++q) {
      nb_vs_coef_T += nb_vs_Q_vec[q] * nb_vs_Q_vec[q] * nb_id_Q_storage[q];
    }

    eq_systems_VS_C.resize(nb_vs_coef_T);
    std::fill(eq_systems_VS_C.begin(), eq_systems_VS_C.end(), 0);

    p_eq_systems_VS_C.resize(Q);
    p_eq_systems_VS_C[0] = eq_systems_VS_C.data();
    for (int q = 1; q < Q; ++q) {
      p_eq_systems_VS_C[q] =
          p_eq_systems_VS_C[q - 1] +
          nb_vs_Q_vec[q - 1] * nb_vs_Q_vec[q - 1] * nb_id_Q_storage[q - 1];
    }
  }
}

// Internal helper for VS variable access
inline FEClass::simple_mat_of_vs_vars::simple_mat_of_vs_vars(
    const FEClass *FE_info, int q) {
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

inline double FEClass::simple_mat_of_vs_vars::operator()(int i, int k) {
  if (k == K_fe) {
    return 1.0;
  }
  return pvars[k](i);
}

// Core FE computation functions
inline void FEClass::compute_fe_coef(double *fe_coef_C, vec &mu_in_N) {
  compute_fe_coef_internal(0, fe_coef_C, true, &mu_in_N, nullptr, nullptr);
}

inline void FEClass::compute_fe_coef(int q, double *fe_coef_C,
                                     double *sum_other_coef_N,
                                     double *in_out_C) {
  compute_fe_coef_internal(q, fe_coef_C, false, nullptr, sum_other_coef_N,
                           in_out_C);
}

inline void FEClass::compute_fe_coef_internal(int q, double *fe_coef_C,
                                              bool is_single, vec *mu_in_N,
                                              double *sum_other_coef_N,
                                              double *in_out_C) {
  int *my_fe = p_fe_id[q];
  int nb_coef = nb_coef_Q[q];
  double *my_fe_coef = fe_coef_C + coef_start_Q[q];

  if (!is_slope_Q[q]) {
    double *my_SW = p_sum_weights_noVS_C[q];

    if (is_single) {
      std::fill_n(my_fe_coef, nb_coef, 0.0);

      const double *mu_ptr = mu_in_N->memptr();
      for (int obs = 0; obs < n_obs; ++obs) {
        if (is_weight) {
          my_fe_coef[my_fe[obs] - 1] += p_weights[obs] * mu_ptr[obs];
        } else {
          my_fe_coef[my_fe[obs] - 1] += mu_ptr[obs];
        }
      }
    } else {
      const double *sum_in_out = in_out_C + coef_start_Q[q];

      // Cluster coefficients
      for (int m = 0; m < nb_coef; ++m) {
        my_fe_coef[m] = sum_in_out[m];
      }

      // Subtract other FE contributions
      for (int i = 0; i < n_obs; ++i) {
        my_fe_coef[my_fe[i] - 1] -= sum_other_coef_N[i];
      }
    }

    // Normalize by sum of weights
    for (int m = 0; m < nb_coef; ++m) {
      my_fe_coef[m] /= my_SW[m];
    }
  } else {
    // Varying slopes case - simplified implementation
    // TODO: Full implementation would follow original pattern with equation
    // solving
    std::fill_n(my_fe_coef, nb_coef, 0.0);
  }
}

inline void FEClass::add_fe_coef_to_mu(int q, double *fe_coef_C,
                                       double *out_N) {
  add_wfe_coef_to_mu_internal(q, fe_coef_C, out_N, false);
}

inline void FEClass::add_wfe_coef_to_mu(int q, double *fe_coef_C,
                                        double *out_N) {
  add_wfe_coef_to_mu_internal(q, fe_coef_C, out_N, true);
}

inline void FEClass::add_wfe_coef_to_mu_internal(int q, double *fe_coef_C,
                                                 double *out_N,
                                                 bool add_weights) {
  int *my_fe = p_fe_id[q];
  double *my_fe_coef = fe_coef_C + coef_start_Q[q];
  const bool use_weights = add_weights && is_weight;

  if (!is_slope_Q[q]) {
    for (int i = 0; i < n_obs; ++i) {
      if (use_weights) {
        out_N[i] += my_fe_coef[my_fe[i] - 1] * p_weights[i];
      } else {
        out_N[i] += my_fe_coef[my_fe[i] - 1];
      }
    }
  } else {
    // Varying slopes case
    for (int i = 0; i < n_obs; ++i) {
      if (use_weights) {
        out_N[i] += my_fe_coef[my_fe[i] - 1] * p_weights[i];
      } else {
        out_N[i] += my_fe_coef[my_fe[i] - 1];
      }
    }
  }
}

inline void FEClass::compute_in_out(int q, double *in_out_C, vec &in_N,
                                    double *out_N) {
  double *sum_in_out = in_out_C + coef_start_Q[q];
  int *my_fe = p_fe_id[q];

  const double *in_ptr = in_N.memptr();

  if (!is_slope_Q[q]) {
    for (int i = 0; i < n_obs; ++i) {
      if (is_weight) {
        sum_in_out[my_fe[i] - 1] += (in_ptr[i] - out_N[i]) * p_weights[i];
      } else {
        sum_in_out[my_fe[i] - 1] += (in_ptr[i] - out_N[i]);
      }
    }
  } else {
    // Varying slopes case
    // TODO: this was simplified
    for (int i = 0; i < n_obs; ++i) {
      if (is_weight) {
        sum_in_out[my_fe[i] - 1] += (in_ptr[i] - out_N[i]) * p_weights[i];
      } else {
        sum_in_out[my_fe[i] - 1] += (in_ptr[i] - out_N[i]);
      }
    }
  }
}

inline void FEClass::compute_fe_coef_2(double *fe_coef_in_C,
                                       double *fe_coef_out_C,
                                       double *fe_coef_tmp, double *in_out_C) {
  // Step 1: Update 2nd FE
  compute_fe_coef_2_internal(fe_coef_in_C, fe_coef_tmp, in_out_C, false);

  // Step 2: Update 1st FE
  compute_fe_coef_2_internal(fe_coef_out_C, fe_coef_tmp, in_out_C, true);
}

inline void FEClass::compute_fe_coef_2_internal(double *fe_coef_in_out_C,
                                                double *fe_coef_tmp,
                                                double *in_out_C, bool step_2) {
  // Simplified 2-FE implementation
  // TODO: check later
  int index_a = step_2 ? 1 : 0;
  int index_b = step_2 ? 0 : 1;

  double *my_fe_coef_a = step_2 ? fe_coef_tmp : fe_coef_in_out_C;
  double *my_fe_coef_b = step_2 ? fe_coef_in_out_C : fe_coef_tmp;

  int *my_fe_a = p_fe_id[index_a];
  int *my_fe_b = p_fe_id[index_b];
  int nb_coef_b = nb_coef_Q[index_b];
  double *my_in_out_b = in_out_C + coef_start_Q[index_b];

  // Initialize coefficients
  for (int m = 0; m < nb_coef_b; ++m) {
    my_fe_coef_b[m] = my_in_out_b[m];
  }

  // Subtract first FE contributions
  for (int i = 0; i < n_obs; ++i) {
    if (is_weight) {
      my_fe_coef_b[my_fe_b[i] - 1] -=
          my_fe_coef_a[my_fe_a[i] - 1] * p_weights[i];
    } else {
      my_fe_coef_b[my_fe_b[i] - 1] -= my_fe_coef_a[my_fe_a[i] - 1];
    }
  }

  // Normalize
  double *my_SW = p_sum_weights_noVS_C[index_b];
  for (int m = 0; m < nb_coef_b; ++m) {
    my_fe_coef_b[m] /= my_SW[m];
  }
}

inline void FEClass::add_2_fe_coef_to_mu(double *fe_coef_a, double *fe_coef_b,
                                         double *in_out_C, double *out_N,
                                         bool update_beta) {
  if (update_beta) {
    compute_fe_coef_2_internal(fe_coef_a, fe_coef_b, in_out_C, false);
  }

  // Add contributions from both FEs
  for (int q = 0; q < 2; ++q) {
    double *my_fe_coef = (q == 0) ? fe_coef_a : fe_coef_b;
    int *my_fe = p_fe_id[q];

    for (int i = 0; i < n_obs; ++i) {
      out_N[i] += my_fe_coef[my_fe[i] - 1];
    }
  }
}

// Irons-Tuck acceleration
inline bool dm_update_X_IronsTuck(int nb_coef_no_Q, std::vector<double> &X,
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

  if (ssq == 0) {
    return true; // Failed
  }

  double coef = vprod / ssq;

  // Update X
  for (int i = 0; i < nb_coef_no_Q; ++i) {
    X[i] = GGX[i] - coef * delta_GX[i];
  }

  return false; // Success
}

// General FE computation for Q >= 3
inline void compute_fe_gnl(double *p_fe_coef_origin,
                           double *p_fe_coef_destination,
                           double *p_sum_other_means, double *p_sum_in_out,
                           PARAM_DEMEAN *args) {
  int n_obs = args->n_obs;
  int Q = args->Q;
  FEClass &FE_info = *(args->p_FE_info);

  for (int q = Q - 1; q >= 0; q--) {
    // Zero out sum of other means
    std::fill_n(p_sum_other_means, n_obs, 0);

    // Compute sum of other FE contributions
    for (int h = 0; h < Q; h++) {
      if (h == q)
        continue;

      double *my_fe_coef = (h < q) ? p_fe_coef_origin : p_fe_coef_destination;
      FE_info.add_wfe_coef_to_mu(h, my_fe_coef, p_sum_other_means);
    }

    // Compute FE coefficients
    FE_info.compute_fe_coef(q, p_fe_coef_destination, p_sum_other_means,
                            p_sum_in_out);
  }
}

// Dispatcher for FE computation
inline void compute_fe(int Q, double *p_fe_coef_origin,
                       double *p_fe_coef_destination, double *p_sum_other_means,
                       double *p_sum_in_out, PARAM_DEMEAN *args) {
  if (Q == 2) {
    FEClass &FE_info = *(args->p_FE_info);
    FE_info.compute_fe_coef_2(p_fe_coef_origin, p_fe_coef_destination,
                              p_sum_other_means, p_sum_in_out);
  } else {
    compute_fe_gnl(p_fe_coef_origin, p_fe_coef_destination, p_sum_other_means,
                   p_sum_in_out, args);
  }
}

// Single FE case (closed form)
inline void demean_single_1(int v, PARAM_DEMEAN *args) {
  int nb_coef_T = args->nb_coef_T;
  FEClass &FE_info = *(args->p_FE_info);

  std::vector<double> fe_coef(nb_coef_T, 0);
  double *p_fe_coef = fe_coef.data();

  vec &input = args->p_input[v];
  double *output = args->p_output[v];

  // Compute FE coefficients
  FE_info.compute_fe_coef(p_fe_coef, input);

  // Apply to output
  FE_info.add_fe_coef_to_mu(0, p_fe_coef, output);

  // Save fixed effects if requested
  if (args->save_fixef) {
    double *fixef_values = args->fixef_values;
    for (int m = 0; m < nb_coef_T; ++m) {
      fixef_values[m] = fe_coef[m];
    }
  }
}

// Main acceleration algorithm
inline bool demean_acc_gnl(int v, int iterMax, PARAM_DEMEAN *args,
                           bool two_fe = false) {
  FEClass &FE_info = *(args->p_FE_info);

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
    nb_coef_T = FE_info.nb_coef_Q[0];
    nb_coef_all = FE_info.nb_coef_Q[0] + FE_info.nb_coef_Q[1];
  }

  // Input/output
  vec &input = args->p_input[v];
  double *output = args->p_output[v];

  // Working vectors
  int size_other_means = two_fe_algo ? FE_info.nb_coef_Q[1] : n_obs;
  std::vector<double> sum_other_means_or_second_coef(size_other_means);
  double *p_sum_other_means = sum_other_means_or_second_coef.data();

  std::vector<double> sum_input_output(nb_coef_all, 0);
  double *p_sum_in_out = sum_input_output.data();

  // Compute initial in_out for all FEs
  for (int q = 0; q < Q; ++q) {
    FE_info.compute_in_out(q, p_sum_in_out, input, output);
  }

  // Iteration vectors
  std::vector<double> X(nb_coef_T, 0);
  std::vector<double> GX(nb_coef_T);
  std::vector<double> GGX(nb_coef_T);
  std::vector<double> Y(nb_coef_T);
  std::vector<double> GY(nb_coef_T);
  std::vector<double> GGY(nb_coef_T);

  double *p_X = X.data();
  double *p_GX = GX.data();
  double *p_GGX = GGX.data();
  double *p_Y = Y.data();
  double *p_GY = GY.data();
  double *p_GGY = GGY.data();

  int nb_coef_no_Q = 0;
  for (int q = 0; q < (Q - 1); ++q) {
    nb_coef_no_Q += FE_info.nb_coef_Q[q];
  }
  std::vector<double> delta_GX(nb_coef_no_Q);
  std::vector<double> delta2_X(nb_coef_no_Q);

  int grand_acc = 0;

  // First iteration
  compute_fe(Q, p_X, p_GX, p_sum_other_means, p_sum_in_out, args);

  // Check convergence
  bool keepGoing = false;
  for (int i = 0; i < nb_coef_T; ++i) {
    if (continue_crit(X[i], GX[i], diffMax)) {
      keepGoing = true;
      break;
    }
  }

  // Main iteration loop
  double ssr = 0;
  int iter = 0;
  bool numconv = false;

  while (!*(args->stopnow) && keepGoing && iter < iterMax) {
    iter++;

    // Extra projections
    for (int rep = 0; rep < n_extraProj; ++rep) {
      compute_fe(Q, p_GX, p_GGX, p_sum_other_means, p_sum_in_out, args);
      compute_fe(Q, p_GGX, p_X, p_sum_other_means, p_sum_in_out, args);
      compute_fe(Q, p_X, p_GX, p_sum_other_means, p_sum_in_out, args);
    }

    // Main projection
    compute_fe(Q, p_GX, p_GGX, p_sum_other_means, p_sum_in_out, args);

    // Irons-Tuck acceleration
    numconv =
        dm_update_X_IronsTuck(nb_coef_no_Q, X, GX, GGX, delta_GX, delta2_X);
    if (numconv)
      break;

    if (iter >= iter_projAfterAcc) {
      std::memcpy(p_Y, p_X, nb_coef_T * sizeof(double));
      compute_fe(Q, p_Y, p_X, p_sum_other_means, p_sum_in_out, args);
    }

    // Next iteration
    compute_fe(Q, p_X, p_GX, p_sum_other_means, p_sum_in_out, args);

    // Check convergence
    keepGoing = false;
    for (int i = 0; i < nb_coef_no_Q; ++i) {
      if (continue_crit(X[i], GX[i], diffMax)) {
        keepGoing = true;
        break;
      }
    }

    // Grand acceleration
    if (iter % iter_grandAcc == 0) {
      ++grand_acc;
      if (grand_acc == 1) {
        std::memcpy(p_Y, p_GX, nb_coef_T * sizeof(double));
      } else if (grand_acc == 2) {
        std::memcpy(p_GY, p_GX, nb_coef_T * sizeof(double));
      } else {
        std::memcpy(p_GGY, p_GX, nb_coef_T * sizeof(double));
        numconv =
            dm_update_X_IronsTuck(nb_coef_no_Q, Y, GY, GGY, delta_GX, delta2_X);
        if (numconv)
          break;
        compute_fe(Q, p_Y, p_GX, p_sum_other_means, p_sum_in_out, args);
        grand_acc = 0;
      }
    }

    // SSR stopping criterion
    if (iter % 40 == 0) {
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
      ssr = 0;
      const double *input_ptr = input.memptr();
      for (int i = 0; i < n_obs; ++i) {
        double resid = input_ptr[i] - mu_current[i];
        ssr += resid * resid;
      }

      if (stopping_crit(ssr_old, ssr, diffMax)) {
        break;
      }
    }
  }

  // Update output
  double *p_beta_final = nullptr;
  if (two_fe_algo) {
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

  // Track iterations
  args->p_iterations_all[v] += iter;

  // Save fixed effects
  if (args->save_fixef) {
    double *fixef_values = args->fixef_values;
    for (int m = 0; m < nb_coef_T; ++m) {
      fixef_values[m] += GX[m];
    }

    if (two_fe_algo) {
      int n_coefs_FE1 = nb_coef_T;
      int n_coefs_FE2 = size_other_means;
      for (int m = 0; m < n_coefs_FE2; ++m) {
        fixef_values[n_coefs_FE1 + m] += p_beta_final[m];
      }
    }
  }

  return iter < iterMax;
}

// High-level single variable demeaning
inline void demean_single_gnl(int v, PARAM_DEMEAN *args) {
  int iterMax = args->iterMax;
  int iter_warmup = args->algo_iter_warmup;
  int Q = args->Q;

  if (Q == 2) {
    demean_acc_gnl(v, iterMax, args);
  } else {
    bool conv = false;

    if (iter_warmup > 0) {
      conv = demean_acc_gnl(v, iter_warmup, args);
    }

    if (!conv && iterMax > iter_warmup) {
      int iter_max_2FE = iterMax / 2 - iter_warmup;
      if (iter_max_2FE > 0) {
        demean_acc_gnl(v, iter_max_2FE, args, true);
      }

      int iter_previous = args->p_iterations_all[v];
      demean_acc_gnl(v, iterMax - iter_previous, args);
    }
  }

  args->jobdone[v] = 1;
}

#endif // CAPYBARA_CENTER
