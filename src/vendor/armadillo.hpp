// cpp11armadillo version: 0.1.2
// vendored on: 2024-02-18
// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#ifndef ARMA_INCLUDES
#define ARMA_INCLUDES

// NOTE: functions that are designed to be user accessible are described in the
// documentation (docs.html). NOTE: all other functions and classes (ie. not
// explicitly described in the documentation) NOTE: are considered as internal
// implementation details, and may be changed or removed without notice.

// clang-format off

// workaround to avoid R check() notes about std::cerr
#include "armadillo/r_compatible_messages.hpp"

#include "armadillo/config.hpp"
#include "armadillo/compiler_check.hpp"

#include <cstdlib>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <climits>
#include <cstdint>
#include <cmath>
#include <ctime>

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <new>
#include <limits>
#include <algorithm>
#include <complex>
#include <vector>
#include <utility>
#include <map>
#include <initializer_list>
#include <random>
#include <functional>
#include <chrono>
#include <atomic>

#if !defined(ARMA_DONT_USE_STD_MUTEX)
  #include <mutex>
#endif

// #if defined(ARMA_HAVE_CXX17)
//   #include <charconv>
//   #include <system_error>
// #endif

#if ( defined(__unix__) || defined(__unix) || defined(_POSIX_C_SOURCE) || (defined(__APPLE__) && defined(__MACH__)) ) && !defined(_WIN32)
  #include <unistd.h>
#endif

#if defined(ARMA_USE_TBB_ALLOC)
  #if defined(__has_include)
    #if __has_include(<tbb/scalable_allocator.h>)
      #include <tbb/scalable_allocator.h>
    #else
      #undef ARMA_USE_TBB_ALLOC
      #pragma message ("WARNING: use of TBB alloc disabled; tbb/scalable_allocator.h header not found")
    #endif
  #else
    #include <tbb/scalable_allocator.h>
  #endif
#endif

#if defined(ARMA_USE_MKL_ALLOC)
  #if defined(__has_include)
    #if __has_include(<mkl_service.h>)
      #include <mkl_service.h>
    #else
      #undef ARMA_USE_MKL_ALLOC
      #pragma message ("WARNING: use of MKL alloc disabled; mkl_service.h header not found")
    #endif
  #else
    #include <mkl_service.h>
  #endif
#endif


#include "armadillo/compiler_setup.hpp"


#if defined(ARMA_USE_OPENMP)
  #if defined(__has_include)
    #if __has_include(<omp.h>)
      #include <omp.h>
    #else
      #undef ARMA_USE_OPENMP
      #pragma message ("WARNING: use of OpenMP disabled; omp.h header not found")
    #endif
  #else
    #include <omp.h>
  #endif
#endif


#include "armadillo/include_hdf5.hpp"
#include "armadillo/include_superlu.hpp"


//! \namespace arma namespace for Armadillo classes and functions
namespace arma
  {
  
  // preliminaries
  
  #include "armadillo/arma_forward.hpp"
  #include "armadillo/arma_static_check.hpp"
  #include "armadillo/typedef_elem.hpp"
  #include "armadillo/typedef_elem_check.hpp"
  #include "armadillo/typedef_mat.hpp"
  #include "armadillo/arma_str.hpp"
  #include "armadillo/arma_version.hpp"
  #include "armadillo/arma_config.hpp"
  #include "armadillo/traits.hpp"
  #include "armadillo/promote_type.hpp"
  #include "armadillo/upgrade_val.hpp"
  #include "armadillo/restrictors.hpp"
  #include "armadillo/access.hpp"
  #include "armadillo/span.hpp"
  #include "armadillo/distr_param.hpp"
  #include "armadillo/constants.hpp"
  #include "armadillo/constants_old.hpp"
  #include "armadillo/mp_misc.hpp"
  #include "armadillo/arma_rel_comparators.hpp"
  #include "armadillo/fill.hpp"
  
  #if defined(ARMA_RNG_ALT)
    #include ARMA_INCFILE_WRAP(ARMA_RNG_ALT)
  #else
    #include "armadillo/arma_rng_cxx03.hpp"
  #endif
  
  #include "armadillo/arma_rng.hpp"
  
  
  //
  // class prototypes
  
  #include "armadillo/Base_bones.hpp"
  #include "armadillo/BaseCube_bones.hpp"
  #include "armadillo/SpBase_bones.hpp"
  
  #include "armadillo/def_blas.hpp"
  #include "armadillo/def_atlas.hpp"
  #include "armadillo/def_lapack.hpp"
  #include "armadillo/def_arpack.hpp"
  #include "armadillo/def_superlu.hpp"
  #include "armadillo/def_fftw3.hpp"
  
  #include "armadillo/translate_blas.hpp"
  #include "armadillo/translate_atlas.hpp"
  #include "armadillo/translate_lapack.hpp"
  #include "armadillo/translate_arpack.hpp"
  #include "armadillo/translate_superlu.hpp"
  #include "armadillo/translate_fftw3.hpp"
  
  #include "armadillo/cond_rel_bones.hpp"
  #include "armadillo/arrayops_bones.hpp"
  #include "armadillo/podarray_bones.hpp"
  #include "armadillo/auxlib_bones.hpp"
  #include "armadillo/sp_auxlib_bones.hpp"
  
  #include "armadillo/injector_bones.hpp"
  
  #include "armadillo/Mat_bones.hpp"
  #include "armadillo/Col_bones.hpp"
  #include "armadillo/Row_bones.hpp"
  #include "armadillo/Cube_bones.hpp"
  #include "armadillo/xvec_htrans_bones.hpp"
  #include "armadillo/xtrans_mat_bones.hpp"
  #include "armadillo/SizeMat_bones.hpp"
  #include "armadillo/SizeCube_bones.hpp"
    
  #include "armadillo/SpValProxy_bones.hpp"
  #include "armadillo/SpMat_bones.hpp"
  #include "armadillo/SpCol_bones.hpp"
  #include "armadillo/SpRow_bones.hpp"
  #include "armadillo/SpSubview_bones.hpp"
  #include "armadillo/SpSubview_col_list_bones.hpp"
  #include "armadillo/spdiagview_bones.hpp"
  #include "armadillo/MapMat_bones.hpp"
  
  #include "armadillo/typedef_mat_fixed.hpp"
  
  #include "armadillo/field_bones.hpp"
  #include "armadillo/subview_bones.hpp"
  #include "armadillo/subview_elem1_bones.hpp"
  #include "armadillo/subview_elem2_bones.hpp"
  #include "armadillo/subview_field_bones.hpp"
  #include "armadillo/subview_cube_bones.hpp"
  #include "armadillo/diagview_bones.hpp"
  #include "armadillo/subview_each_bones.hpp"
  #include "armadillo/subview_cube_each_bones.hpp"
  #include "armadillo/subview_cube_slices_bones.hpp"
  
  #include "armadillo/hdf5_name.hpp"
  #include "armadillo/csv_name.hpp"
  #include "armadillo/diskio_bones.hpp"
  #include "armadillo/wall_clock_bones.hpp"
  #include "armadillo/running_stat_bones.hpp"
  #include "armadillo/running_stat_vec_bones.hpp"
  
  #include "armadillo/Op_bones.hpp"
  #include "armadillo/CubeToMatOp_bones.hpp"
  #include "armadillo/OpCube_bones.hpp"
  #include "armadillo/SpOp_bones.hpp"
  #include "armadillo/SpToDOp_bones.hpp"
  
  #include "armadillo/eOp_bones.hpp"
  #include "armadillo/eOpCube_bones.hpp"
  
  #include "armadillo/mtOp_bones.hpp"
  #include "armadillo/mtOpCube_bones.hpp"
  #include "armadillo/mtSpOp_bones.hpp"
  
  #include "armadillo/Glue_bones.hpp"
  #include "armadillo/eGlue_bones.hpp"
  #include "armadillo/mtGlue_bones.hpp"
  #include "armadillo/SpGlue_bones.hpp"
  #include "armadillo/mtSpGlue_bones.hpp"
  #include "armadillo/SpToDGlue_bones.hpp"
  
  #include "armadillo/GlueCube_bones.hpp"
  #include "armadillo/eGlueCube_bones.hpp"
  #include "armadillo/mtGlueCube_bones.hpp"
  
  #include "armadillo/eop_core_bones.hpp"
  #include "armadillo/eglue_core_bones.hpp"
  
  #include "armadillo/Gen_bones.hpp"
  #include "armadillo/GenCube_bones.hpp"
  
  #include "armadillo/op_diagmat_bones.hpp"
  #include "armadillo/op_diagvec_bones.hpp"
  #include "armadillo/op_dot_bones.hpp"
  #include "armadillo/op_det_bones.hpp"
  #include "armadillo/op_log_det_bones.hpp"
  #include "armadillo/op_inv_gen_bones.hpp"
  #include "armadillo/op_inv_spd_bones.hpp"
  #include "armadillo/op_htrans_bones.hpp"
  #include "armadillo/op_max_bones.hpp"
  #include "armadillo/op_min_bones.hpp"
  #include "armadillo/op_index_max_bones.hpp"
  #include "armadillo/op_index_min_bones.hpp"
  #include "armadillo/op_mean_bones.hpp"
  #include "armadillo/op_median_bones.hpp"
  #include "armadillo/op_sort_bones.hpp"
  #include "armadillo/op_sort_index_bones.hpp"
  #include "armadillo/op_sum_bones.hpp"
  #include "armadillo/op_stddev_bones.hpp"
  #include "armadillo/op_strans_bones.hpp"
  #include "armadillo/op_var_bones.hpp"
  #include "armadillo/op_repmat_bones.hpp"
  #include "armadillo/op_repelem_bones.hpp"
  #include "armadillo/op_reshape_bones.hpp"
  #include "armadillo/op_vectorise_bones.hpp"
  #include "armadillo/op_resize_bones.hpp"
  #include "armadillo/op_cov_bones.hpp"
  #include "armadillo/op_cor_bones.hpp"
  #include "armadillo/op_shift_bones.hpp"
  #include "armadillo/op_shuffle_bones.hpp"
  #include "armadillo/op_prod_bones.hpp"
  #include "armadillo/op_pinv_bones.hpp"
  #include "armadillo/op_dotext_bones.hpp"
  #include "armadillo/op_flip_bones.hpp"
  #include "armadillo/op_reverse_bones.hpp"
  #include "armadillo/op_princomp_bones.hpp"
  #include "armadillo/op_misc_bones.hpp"
  #include "armadillo/op_orth_null_bones.hpp"
  #include "armadillo/op_relational_bones.hpp"
  #include "armadillo/op_find_bones.hpp"
  #include "armadillo/op_find_unique_bones.hpp"
  #include "armadillo/op_chol_bones.hpp"
  #include "armadillo/op_cx_scalar_bones.hpp"
  #include "armadillo/op_trimat_bones.hpp"
  #include "armadillo/op_cumsum_bones.hpp"
  #include "armadillo/op_cumprod_bones.hpp"
  #include "armadillo/op_symmat_bones.hpp"
  #include "armadillo/op_hist_bones.hpp"
  #include "armadillo/op_unique_bones.hpp"
  #include "armadillo/op_toeplitz_bones.hpp"
  #include "armadillo/op_fft_bones.hpp"
  #include "armadillo/op_any_bones.hpp"
  #include "armadillo/op_all_bones.hpp"
  #include "armadillo/op_normalise_bones.hpp"
  #include "armadillo/op_clamp_bones.hpp"
  #include "armadillo/op_expmat_bones.hpp"
  #include "armadillo/op_nonzeros_bones.hpp"
  #include "armadillo/op_diff_bones.hpp"
  #include "armadillo/op_norm_bones.hpp"
  #include "armadillo/op_vecnorm_bones.hpp"
  #include "armadillo/op_norm2est_bones.hpp"
  #include "armadillo/op_sqrtmat_bones.hpp"
  #include "armadillo/op_logmat_bones.hpp"
  #include "armadillo/op_range_bones.hpp"
  #include "armadillo/op_chi2rnd_bones.hpp"
  #include "armadillo/op_wishrnd_bones.hpp"
  #include "armadillo/op_roots_bones.hpp"
  #include "armadillo/op_cond_bones.hpp"
  #include "armadillo/op_rcond_bones.hpp"
  #include "armadillo/op_sp_plus_bones.hpp"
  #include "armadillo/op_sp_minus_bones.hpp"
  #include "armadillo/op_powmat_bones.hpp"
  #include "armadillo/op_rank_bones.hpp"
  #include "armadillo/op_row_as_mat_bones.hpp"
  #include "armadillo/op_col_as_mat_bones.hpp"
  
  #include "armadillo/glue_times_bones.hpp"
  #include "armadillo/glue_times_misc_bones.hpp"
  #include "armadillo/glue_mixed_bones.hpp"
  #include "armadillo/glue_cov_bones.hpp"
  #include "armadillo/glue_cor_bones.hpp"
  #include "armadillo/glue_kron_bones.hpp"
  #include "armadillo/glue_cross_bones.hpp"
  #include "armadillo/glue_join_bones.hpp"
  #include "armadillo/glue_relational_bones.hpp"
  #include "armadillo/glue_solve_bones.hpp"
  #include "armadillo/glue_conv_bones.hpp"
  #include "armadillo/glue_toeplitz_bones.hpp"
  #include "armadillo/glue_hist_bones.hpp"
  #include "armadillo/glue_histc_bones.hpp"
  #include "armadillo/glue_max_bones.hpp"
  #include "armadillo/glue_min_bones.hpp"
  #include "armadillo/glue_trapz_bones.hpp"
  #include "armadillo/glue_atan2_bones.hpp"
  #include "armadillo/glue_hypot_bones.hpp"
  #include "armadillo/glue_polyfit_bones.hpp"
  #include "armadillo/glue_polyval_bones.hpp"
  #include "armadillo/glue_intersect_bones.hpp"
  #include "armadillo/glue_affmul_bones.hpp"
  #include "armadillo/glue_mvnrnd_bones.hpp"
  #include "armadillo/glue_quantile_bones.hpp"
  #include "armadillo/glue_powext_bones.hpp"
  
  #include "armadillo/gmm_misc_bones.hpp"
  #include "armadillo/gmm_diag_bones.hpp"
  #include "armadillo/gmm_full_bones.hpp"
  
  #include "armadillo/spop_max_bones.hpp"
  #include "armadillo/spop_min_bones.hpp"
  #include "armadillo/spop_sum_bones.hpp"
  #include "armadillo/spop_strans_bones.hpp"
  #include "armadillo/spop_htrans_bones.hpp"
  #include "armadillo/spop_misc_bones.hpp"
  #include "armadillo/spop_diagmat_bones.hpp"
  #include "armadillo/spop_mean_bones.hpp"
  #include "armadillo/spop_var_bones.hpp"
  #include "armadillo/spop_trimat_bones.hpp"
  #include "armadillo/spop_symmat_bones.hpp"
  #include "armadillo/spop_normalise_bones.hpp"
  #include "armadillo/spop_reverse_bones.hpp"
  #include "armadillo/spop_repmat_bones.hpp"
  #include "armadillo/spop_vectorise_bones.hpp"
  #include "armadillo/spop_norm_bones.hpp"
  #include "armadillo/spop_vecnorm_bones.hpp"
  
  #include "armadillo/spglue_plus_bones.hpp"
  #include "armadillo/spglue_minus_bones.hpp"
  #include "armadillo/spglue_schur_bones.hpp"
  #include "armadillo/spglue_times_bones.hpp"
  #include "armadillo/spglue_join_bones.hpp"
  #include "armadillo/spglue_kron_bones.hpp"
  #include "armadillo/spglue_min_bones.hpp"
  #include "armadillo/spglue_max_bones.hpp"
  #include "armadillo/spglue_merge_bones.hpp"
  #include "armadillo/spglue_relational_bones.hpp"
  
  #include "armadillo/spsolve_factoriser_bones.hpp"
  
  #if defined(ARMA_USE_NEWARP)
    #include "armadillo/newarp_EigsSelect.hpp"
    #include "armadillo/newarp_DenseGenMatProd_bones.hpp"
    #include "armadillo/newarp_SparseGenMatProd_bones.hpp"
    #include "armadillo/newarp_SparseGenRealShiftSolve_bones.hpp"
    #include "armadillo/newarp_DoubleShiftQR_bones.hpp"
    #include "armadillo/newarp_GenEigsSolver_bones.hpp"
    #include "armadillo/newarp_SymEigsSolver_bones.hpp"
    #include "armadillo/newarp_SymEigsShiftSolver_bones.hpp"
    #include "armadillo/newarp_TridiagEigen_bones.hpp"
    #include "armadillo/newarp_UpperHessenbergEigen_bones.hpp"
    #include "armadillo/newarp_UpperHessenbergQR_bones.hpp"
  #endif
  
  
  //
  // low-level debugging and memory handling functions
  
  #include "armadillo/debug.hpp"
  #include "armadillo/memory.hpp"
  
  //
  // wrappers for various cmath functions
  
  #include "armadillo/arma_cmath.hpp"
  
  //
  // classes that underlay metaprogramming 
  
  #include "armadillo/unwrap.hpp"
  #include "armadillo/unwrap_cube.hpp"
  #include "armadillo/unwrap_spmat.hpp"
  
  #include "armadillo/Proxy.hpp"
  #include "armadillo/ProxyCube.hpp"
  #include "armadillo/SpProxy.hpp"
  
  #include "armadillo/diagmat_proxy.hpp"

  #include "armadillo/strip.hpp"
  
  #include "armadillo/eop_aux.hpp"
  
  //
  // ostream
  
  #include "armadillo/arma_ostream_bones.hpp"
  #include "armadillo/arma_ostream_meat.hpp"
  
  //
  // n_unique, which is used by some sparse operators

  #include "armadillo/fn_n_unique.hpp"
  
  //
  // operators
  
  #include "armadillo/operator_plus.hpp"
  #include "armadillo/operator_minus.hpp"
  #include "armadillo/operator_times.hpp"
  #include "armadillo/operator_schur.hpp"
  #include "armadillo/operator_div.hpp"
  #include "armadillo/operator_relational.hpp"
  
  #include "armadillo/operator_cube_plus.hpp"
  #include "armadillo/operator_cube_minus.hpp"
  #include "armadillo/operator_cube_times.hpp"
  #include "armadillo/operator_cube_schur.hpp"
  #include "armadillo/operator_cube_div.hpp"
  #include "armadillo/operator_cube_relational.hpp"
  
  #include "armadillo/operator_ostream.hpp"
  
  //
  // user accessible functions
  
  // the order of the fn_*.hpp include files matters,
  // as some files require functionality given in preceding files
  
  #include "armadillo/fn_conv_to.hpp"
  #include "armadillo/fn_max.hpp"
  #include "armadillo/fn_min.hpp"
  #include "armadillo/fn_index_max.hpp"
  #include "armadillo/fn_index_min.hpp"
  #include "armadillo/fn_accu.hpp"
  #include "armadillo/fn_sum.hpp"
  #include "armadillo/fn_diagmat.hpp"
  #include "armadillo/fn_diagvec.hpp"
  #include "armadillo/fn_inv.hpp"
  #include "armadillo/fn_inv_sympd.hpp"
  #include "armadillo/fn_trace.hpp"
  #include "armadillo/fn_trans.hpp"
  #include "armadillo/fn_det.hpp"
  #include "armadillo/fn_log_det.hpp"
  #include "armadillo/fn_eig_gen.hpp"
  #include "armadillo/fn_eig_sym.hpp"
  #include "armadillo/fn_eig_pair.hpp"
  #include "armadillo/fn_lu.hpp"
  #include "armadillo/fn_zeros.hpp"
  #include "armadillo/fn_ones.hpp"
  #include "armadillo/fn_eye.hpp"
  #include "armadillo/fn_misc.hpp"
  #include "armadillo/fn_orth_null.hpp"
  #include "armadillo/fn_regspace.hpp"
  #include "armadillo/fn_find.hpp"
  #include "armadillo/fn_find_unique.hpp"
  #include "armadillo/fn_elem.hpp"
  #include "armadillo/fn_approx_equal.hpp"
  #include "armadillo/fn_norm.hpp"
  #include "armadillo/fn_vecnorm.hpp"
  #include "armadillo/fn_dot.hpp"
  #include "armadillo/fn_randu.hpp"
  #include "armadillo/fn_randn.hpp"
  #include "armadillo/fn_trig.hpp"
  #include "armadillo/fn_mean.hpp"
  #include "armadillo/fn_median.hpp"
  #include "armadillo/fn_stddev.hpp"
  #include "armadillo/fn_var.hpp"
  #include "armadillo/fn_sort.hpp"
  #include "armadillo/fn_sort_index.hpp"
  #include "armadillo/fn_strans.hpp"
  #include "armadillo/fn_chol.hpp"
  #include "armadillo/fn_qr.hpp"
  #include "armadillo/fn_svd.hpp"
  #include "armadillo/fn_solve.hpp"
  #include "armadillo/fn_repmat.hpp"
  #include "armadillo/fn_repelem.hpp"
  #include "armadillo/fn_reshape.hpp"
  #include "armadillo/fn_vectorise.hpp"
  #include "armadillo/fn_resize.hpp"
  #include "armadillo/fn_cov.hpp"
  #include "armadillo/fn_cor.hpp"
  #include "armadillo/fn_shift.hpp"
  #include "armadillo/fn_shuffle.hpp"
  #include "armadillo/fn_prod.hpp"
  #include "armadillo/fn_eps.hpp"
  #include "armadillo/fn_pinv.hpp"
  #include "armadillo/fn_rank.hpp"
  #include "armadillo/fn_kron.hpp"
  #include "armadillo/fn_flip.hpp"
  #include "armadillo/fn_reverse.hpp"
  #include "armadillo/fn_as_scalar.hpp"
  #include "armadillo/fn_princomp.hpp"
  #include "armadillo/fn_cross.hpp"
  #include "armadillo/fn_join.hpp"
  #include "armadillo/fn_conv.hpp"
  #include "armadillo/fn_trunc_exp.hpp"
  #include "armadillo/fn_trunc_log.hpp"
  #include "armadillo/fn_toeplitz.hpp"
  #include "armadillo/fn_trimat.hpp"
  #include "armadillo/fn_trimat_ind.hpp"
  #include "armadillo/fn_cumsum.hpp"
  #include "armadillo/fn_cumprod.hpp"
  #include "armadillo/fn_symmat.hpp"
  #include "armadillo/fn_sylvester.hpp"
  #include "armadillo/fn_hist.hpp"
  #include "armadillo/fn_histc.hpp"
  #include "armadillo/fn_unique.hpp"
  #include "armadillo/fn_fft.hpp"
  #include "armadillo/fn_fft2.hpp"
  #include "armadillo/fn_any.hpp"
  #include "armadillo/fn_all.hpp"
  #include "armadillo/fn_size.hpp"
  #include "armadillo/fn_numel.hpp"
  #include "armadillo/fn_inplace_strans.hpp"
  #include "armadillo/fn_inplace_trans.hpp"
  #include "armadillo/fn_randi.hpp"
  #include "armadillo/fn_randg.hpp"
  #include "armadillo/fn_cond_rcond.hpp"
  #include "armadillo/fn_normalise.hpp"
  #include "armadillo/fn_clamp.hpp"
  #include "armadillo/fn_expmat.hpp"
  #include "armadillo/fn_nonzeros.hpp"
  #include "armadillo/fn_interp1.hpp"
  #include "armadillo/fn_interp2.hpp"
  #include "armadillo/fn_qz.hpp"
  #include "armadillo/fn_diff.hpp"
  #include "armadillo/fn_hess.hpp"
  #include "armadillo/fn_schur.hpp"
  #include "armadillo/fn_kmeans.hpp"
  #include "armadillo/fn_sqrtmat.hpp"
  #include "armadillo/fn_logmat.hpp"
  #include "armadillo/fn_trapz.hpp"
  #include "armadillo/fn_range.hpp"
  #include "armadillo/fn_polyfit.hpp"
  #include "armadillo/fn_polyval.hpp"
  #include "armadillo/fn_intersect.hpp"
  #include "armadillo/fn_normpdf.hpp"
  #include "armadillo/fn_log_normpdf.hpp"
  #include "armadillo/fn_normcdf.hpp"
  #include "armadillo/fn_mvnrnd.hpp"
  #include "armadillo/fn_chi2rnd.hpp"
  #include "armadillo/fn_wishrnd.hpp"
  #include "armadillo/fn_roots.hpp"
  #include "armadillo/fn_randperm.hpp"
  #include "armadillo/fn_quantile.hpp"
  #include "armadillo/fn_powmat.hpp"
  #include "armadillo/fn_powext.hpp"
  #include "armadillo/fn_diags_spdiags.hpp"
  
  #include "armadillo/fn_speye.hpp"
  #include "armadillo/fn_spones.hpp"
  #include "armadillo/fn_sprandn.hpp"
  #include "armadillo/fn_sprandu.hpp"
  #include "armadillo/fn_eigs_sym.hpp"
  #include "armadillo/fn_eigs_gen.hpp"
  #include "armadillo/fn_spsolve.hpp"
  #include "armadillo/fn_svds.hpp"
  
  //
  // misc stuff
  
  #include "armadillo/hdf5_misc.hpp"
  #include "armadillo/fft_engine_kissfft.hpp"
  #include "armadillo/fft_engine_fftw3.hpp"
  #include "armadillo/band_helper.hpp"
  #include "armadillo/sym_helper.hpp"
  #include "armadillo/trimat_helper.hpp"
  
  //
  // classes implementing various forms of dense matrix multiplication
  
  #include "armadillo/mul_gemv.hpp"
  #include "armadillo/mul_gemm.hpp"
  #include "armadillo/mul_gemm_mixed.hpp"
  #include "armadillo/mul_syrk.hpp"
  #include "armadillo/mul_herk.hpp"
  
  //
  // class meat
  
  #include "armadillo/Op_meat.hpp"
  #include "armadillo/CubeToMatOp_meat.hpp"
  #include "armadillo/OpCube_meat.hpp"
  #include "armadillo/SpOp_meat.hpp"
  #include "armadillo/SpToDOp_meat.hpp"
  
  #include "armadillo/mtOp_meat.hpp"
  #include "armadillo/mtOpCube_meat.hpp"
  #include "armadillo/mtSpOp_meat.hpp"
  
  #include "armadillo/Glue_meat.hpp"
  #include "armadillo/GlueCube_meat.hpp"
  #include "armadillo/SpGlue_meat.hpp"
  #include "armadillo/mtSpGlue_meat.hpp"
  #include "armadillo/SpToDGlue_meat.hpp"
  
  #include "armadillo/eOp_meat.hpp"
  #include "armadillo/eOpCube_meat.hpp"
  
  #include "armadillo/eGlue_meat.hpp"
  #include "armadillo/eGlueCube_meat.hpp"

  #include "armadillo/mtGlue_meat.hpp"
  #include "armadillo/mtGlueCube_meat.hpp"
  
  #include "armadillo/Base_meat.hpp"
  #include "armadillo/BaseCube_meat.hpp"
  #include "armadillo/SpBase_meat.hpp"
  
  #include "armadillo/Gen_meat.hpp"
  #include "armadillo/GenCube_meat.hpp"
  
  #include "armadillo/eop_core_meat.hpp"
  #include "armadillo/eglue_core_meat.hpp"
  
  #include "armadillo/cond_rel_meat.hpp"
  #include "armadillo/arrayops_meat.hpp"
  #include "armadillo/podarray_meat.hpp"
  #include "armadillo/auxlib_meat.hpp"
  #include "armadillo/sp_auxlib_meat.hpp"
  
  #include "armadillo/injector_meat.hpp"
  
  #include "armadillo/Mat_meat.hpp"
  #include "armadillo/Col_meat.hpp"
  #include "armadillo/Row_meat.hpp"
  #include "armadillo/Cube_meat.hpp"
  #include "armadillo/xvec_htrans_meat.hpp"
  #include "armadillo/xtrans_mat_meat.hpp"
  #include "armadillo/SizeMat_meat.hpp"
  #include "armadillo/SizeCube_meat.hpp"
  
  #include "armadillo/field_meat.hpp"
  #include "armadillo/subview_meat.hpp"
  #include "armadillo/subview_elem1_meat.hpp"
  #include "armadillo/subview_elem2_meat.hpp"
  #include "armadillo/subview_field_meat.hpp"
  #include "armadillo/subview_cube_meat.hpp"
  #include "armadillo/diagview_meat.hpp"
  #include "armadillo/subview_each_meat.hpp"
  #include "armadillo/subview_cube_each_meat.hpp"
  #include "armadillo/subview_cube_slices_meat.hpp"

  #include "armadillo/SpValProxy_meat.hpp"
  #include "armadillo/SpMat_meat.hpp"
  #include "armadillo/SpMat_iterators_meat.hpp"
  #include "armadillo/SpCol_meat.hpp"
  #include "armadillo/SpRow_meat.hpp"
  #include "armadillo/SpSubview_meat.hpp"
  #include "armadillo/SpSubview_iterators_meat.hpp"
  #include "armadillo/SpSubview_col_list_meat.hpp"
  #include "armadillo/spdiagview_meat.hpp"
  #include "armadillo/MapMat_meat.hpp"
  
  #include "armadillo/diskio_meat.hpp"
  #include "armadillo/wall_clock_meat.hpp"
  #include "armadillo/running_stat_meat.hpp"
  #include "armadillo/running_stat_vec_meat.hpp"
  
  #include "armadillo/op_diagmat_meat.hpp"
  #include "armadillo/op_diagvec_meat.hpp"
  #include "armadillo/op_dot_meat.hpp"
  #include "armadillo/op_det_meat.hpp"
  #include "armadillo/op_log_det_meat.hpp"
  #include "armadillo/op_inv_gen_meat.hpp"
  #include "armadillo/op_inv_spd_meat.hpp"
  #include "armadillo/op_htrans_meat.hpp"
  #include "armadillo/op_max_meat.hpp"
  #include "armadillo/op_index_max_meat.hpp"
  #include "armadillo/op_index_min_meat.hpp"
  #include "armadillo/op_min_meat.hpp"
  #include "armadillo/op_mean_meat.hpp"
  #include "armadillo/op_median_meat.hpp"
  #include "armadillo/op_sort_meat.hpp"
  #include "armadillo/op_sort_index_meat.hpp"
  #include "armadillo/op_sum_meat.hpp"
  #include "armadillo/op_stddev_meat.hpp"
  #include "armadillo/op_strans_meat.hpp"
  #include "armadillo/op_var_meat.hpp"
  #include "armadillo/op_repmat_meat.hpp"
  #include "armadillo/op_repelem_meat.hpp"
  #include "armadillo/op_reshape_meat.hpp"
  #include "armadillo/op_vectorise_meat.hpp"
  #include "armadillo/op_resize_meat.hpp"
  #include "armadillo/op_cov_meat.hpp"
  #include "armadillo/op_cor_meat.hpp"
  #include "armadillo/op_shift_meat.hpp"
  #include "armadillo/op_shuffle_meat.hpp"
  #include "armadillo/op_prod_meat.hpp"
  #include "armadillo/op_pinv_meat.hpp"
  #include "armadillo/op_dotext_meat.hpp"
  #include "armadillo/op_flip_meat.hpp"
  #include "armadillo/op_reverse_meat.hpp"
  #include "armadillo/op_princomp_meat.hpp"
  #include "armadillo/op_misc_meat.hpp"
  #include "armadillo/op_orth_null_meat.hpp"
  #include "armadillo/op_relational_meat.hpp"
  #include "armadillo/op_find_meat.hpp"
  #include "armadillo/op_find_unique_meat.hpp"
  #include "armadillo/op_chol_meat.hpp"
  #include "armadillo/op_cx_scalar_meat.hpp"
  #include "armadillo/op_trimat_meat.hpp"
  #include "armadillo/op_cumsum_meat.hpp"
  #include "armadillo/op_cumprod_meat.hpp"
  #include "armadillo/op_symmat_meat.hpp"
  #include "armadillo/op_hist_meat.hpp"
  #include "armadillo/op_unique_meat.hpp"
  #include "armadillo/op_toeplitz_meat.hpp"
  #include "armadillo/op_fft_meat.hpp"
  #include "armadillo/op_any_meat.hpp"
  #include "armadillo/op_all_meat.hpp"
  #include "armadillo/op_normalise_meat.hpp"
  #include "armadillo/op_clamp_meat.hpp"
  #include "armadillo/op_expmat_meat.hpp"
  #include "armadillo/op_nonzeros_meat.hpp"
  #include "armadillo/op_diff_meat.hpp"
  #include "armadillo/op_norm_meat.hpp"
  #include "armadillo/op_vecnorm_meat.hpp"
  #include "armadillo/op_norm2est_meat.hpp"
  #include "armadillo/op_sqrtmat_meat.hpp"
  #include "armadillo/op_logmat_meat.hpp"
  #include "armadillo/op_range_meat.hpp"
  #include "armadillo/op_chi2rnd_meat.hpp"
  #include "armadillo/op_wishrnd_meat.hpp"
  #include "armadillo/op_roots_meat.hpp"
  #include "armadillo/op_cond_meat.hpp"
  #include "armadillo/op_rcond_meat.hpp"
  #include "armadillo/op_sp_plus_meat.hpp"
  #include "armadillo/op_sp_minus_meat.hpp"
  #include "armadillo/op_powmat_meat.hpp"
  #include "armadillo/op_rank_meat.hpp"
  #include "armadillo/op_row_as_mat_meat.hpp"
  #include "armadillo/op_col_as_mat_meat.hpp"
  
  #include "armadillo/glue_times_meat.hpp"
  #include "armadillo/glue_times_misc_meat.hpp"
  #include "armadillo/glue_mixed_meat.hpp"
  #include "armadillo/glue_cov_meat.hpp"
  #include "armadillo/glue_cor_meat.hpp"
  #include "armadillo/glue_kron_meat.hpp"
  #include "armadillo/glue_cross_meat.hpp"
  #include "armadillo/glue_join_meat.hpp"
  #include "armadillo/glue_relational_meat.hpp"
  #include "armadillo/glue_solve_meat.hpp"
  #include "armadillo/glue_conv_meat.hpp"
  #include "armadillo/glue_toeplitz_meat.hpp"
  #include "armadillo/glue_hist_meat.hpp"
  #include "armadillo/glue_histc_meat.hpp"
  #include "armadillo/glue_max_meat.hpp"
  #include "armadillo/glue_min_meat.hpp"
  #include "armadillo/glue_trapz_meat.hpp"
  #include "armadillo/glue_atan2_meat.hpp"
  #include "armadillo/glue_hypot_meat.hpp"
  #include "armadillo/glue_polyfit_meat.hpp"
  #include "armadillo/glue_polyval_meat.hpp"
  #include "armadillo/glue_intersect_meat.hpp"
  #include "armadillo/glue_affmul_meat.hpp"
  #include "armadillo/glue_mvnrnd_meat.hpp"
  #include "armadillo/glue_quantile_meat.hpp"
  #include "armadillo/glue_powext_meat.hpp"
  
  #include "armadillo/gmm_misc_meat.hpp"
  #include "armadillo/gmm_diag_meat.hpp"
  #include "armadillo/gmm_full_meat.hpp"
  
  #include "armadillo/spop_max_meat.hpp"
  #include "armadillo/spop_min_meat.hpp"
  #include "armadillo/spop_sum_meat.hpp"
  #include "armadillo/spop_strans_meat.hpp"
  #include "armadillo/spop_htrans_meat.hpp"
  #include "armadillo/spop_misc_meat.hpp"
  #include "armadillo/spop_diagmat_meat.hpp"
  #include "armadillo/spop_mean_meat.hpp"
  #include "armadillo/spop_var_meat.hpp"
  #include "armadillo/spop_trimat_meat.hpp"
  #include "armadillo/spop_symmat_meat.hpp"
  #include "armadillo/spop_normalise_meat.hpp"
  #include "armadillo/spop_reverse_meat.hpp"
  #include "armadillo/spop_repmat_meat.hpp"
  #include "armadillo/spop_vectorise_meat.hpp"
  #include "armadillo/spop_norm_meat.hpp"
  #include "armadillo/spop_vecnorm_meat.hpp"
  
  #include "armadillo/spglue_plus_meat.hpp"
  #include "armadillo/spglue_minus_meat.hpp"
  #include "armadillo/spglue_schur_meat.hpp"
  #include "armadillo/spglue_times_meat.hpp"
  #include "armadillo/spglue_join_meat.hpp"
  #include "armadillo/spglue_kron_meat.hpp"
  #include "armadillo/spglue_min_meat.hpp"
  #include "armadillo/spglue_max_meat.hpp"
  #include "armadillo/spglue_merge_meat.hpp"
  #include "armadillo/spglue_relational_meat.hpp"
  
  #include "armadillo/spsolve_factoriser_meat.hpp"
  
  #if defined(ARMA_USE_NEWARP)
    #include "armadillo/newarp_cx_attrib.hpp"
    #include "armadillo/newarp_SortEigenvalue.hpp"
    #include "armadillo/newarp_DenseGenMatProd_meat.hpp"
    #include "armadillo/newarp_SparseGenMatProd_meat.hpp"
    #include "armadillo/newarp_SparseGenRealShiftSolve_meat.hpp"
    #include "armadillo/newarp_DoubleShiftQR_meat.hpp"
    #include "armadillo/newarp_GenEigsSolver_meat.hpp"
    #include "armadillo/newarp_SymEigsSolver_meat.hpp"
    #include "armadillo/newarp_SymEigsShiftSolver_meat.hpp"
    #include "armadillo/newarp_TridiagEigen_meat.hpp"
    #include "armadillo/newarp_UpperHessenbergEigen_meat.hpp"
    #include "armadillo/newarp_UpperHessenbergQR_meat.hpp"
  #endif
  }



#include "armadillo/compiler_setup_post.hpp"

#endif
