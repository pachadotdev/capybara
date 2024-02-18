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

//! \addtogroup fn_polyfit
//! @{

template <typename T1, typename T2>
inline
    typename enable_if2<is_supported_blas_type<typename T1::elem_type>::value,
                        bool>::result
    polyfit(Mat<typename T1::elem_type> &out,
            const Base<typename T1::elem_type, T1> &X,
            const Base<typename T1::elem_type, T2> &Y, const uword N) {
  arma_extra_debug_sigprint();

  const bool status =
      glue_polyfit::apply_direct(out, X.get_ref(), Y.get_ref(), N);

  if (status == false) {
    out.soft_reset();
    arma_debug_warn_level(3, "polyfit(): failed");
  }

  return status;
}

template <typename T1, typename T2>
arma_warn_unused inline
    typename enable_if2<is_supported_blas_type<typename T1::elem_type>::value,
                        const Glue<T1, T2, glue_polyfit>>::result
    polyfit(const Base<typename T1::elem_type, T1> &X,
            const Base<typename T1::elem_type, T2> &Y, const uword N) {
  arma_extra_debug_sigprint();

  return Glue<T1, T2, glue_polyfit>(X.get_ref(), Y.get_ref(), N);
}

//! @}
