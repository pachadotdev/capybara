   specified C++20
   using C++ compiler: ‘clang version 22.1.5’
   using C++20
   /usr/bin/clang++ -std=gnu++20 -I"/opt/R-4.6.0/lib64/R/include" -DNDEBUG  -I'/home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include' -I'/home/pacha/R/x86_64-pc-linux-gnu-library/4.6/cpp4r/include' -I/usr/local/include   -fopenmp -O3 -ffast-math -march=native -mtune=native -mavx -mavx2 -mfma -funroll-loops -fprefetch-loop-arrays -ftree-vectorize -ftree-slp-vectorize -flto -DARMA_NO_DEBUG -DARMA_USE_BLAS -DARMA_USE_LAPACK -DARMA_USE_OPENMP -DARMA_OPENMP_THREADS=4 -DARMA_64BIT_WORD -DARMA_DONT_USE_WRAPPER -DARMADILLO4R_NO_SPARSE -DCAPYBARA_DEFAULT_OMP_THREADS=4 -fpic  -Wall -O3 -pedantic -fno-lto -Wno-ignored-optimization-argument -UNDEBUG -Wall -pedantic -g -O0 -fdiagnostics-color=always  -c capybara.cpp -o capybara.o
   In file included from capybara.cpp:8:
   In file included from /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r.hpp:52:
   In file included from /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo.hpp:166:
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/fill.hpp:102:89: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     102 |   template<>            inline bool isfinite_wrapper(float                 x)  { return std::isfinite(x);                                   }
         |                                                                                         ^~~~~~~~~~~~~~~~
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/fill.hpp:103:89: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     103 |   template<>            inline bool isfinite_wrapper(double                x)  { return std::isfinite(x);                                   }
         |                                                                                         ^~~~~~~~~~~~~~~~
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/fill.hpp:104:89: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     104 |   template<>            inline bool isfinite_wrapper(std::complex<float>&  x)  { return std::isfinite(x.real()) && std::isfinite(x.imag()); }
         |                                                                                         ^~~~~~~~~~~~~~~~~~~~~~~
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/fill.hpp:104:116: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     104 |   template<>            inline bool isfinite_wrapper(std::complex<float>&  x)  { return std::isfinite(x.real()) && std::isfinite(x.imag()); }
         |                                                                                                                    ^~~~~~~~~~~~~~~~~~~~~~~
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/fill.hpp:105:89: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     105 |   template<>            inline bool isfinite_wrapper(std::complex<double>& x)  { return std::isfinite(x.real()) && std::isfinite(x.imag()); }
         |                                                                                         ^~~~~~~~~~~~~~~~~~~~~~~
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/fill.hpp:105:116: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     105 |   template<>            inline bool isfinite_wrapper(std::complex<double>& x)  { return std::isfinite(x.real()) && std::isfinite(x.imag()); }
         |                                                                                                                    ^~~~~~~~~~~~~~~~~~~~~~~
   In file included from capybara.cpp:8:
   In file included from /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r.hpp:52:
   In file included from /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo.hpp:445:
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/arma_cmath.hpp:44:10: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
      44 |   return std::isfinite(x);
         |          ^~~~~~~~~~~~~~~~
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/arma_cmath.hpp:54:10: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
      54 |   return std::isfinite(x);
         |          ^~~~~~~~~~~~~~~~
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/arma_cmath.hpp:86:11: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
      86 |   return (std::isfinite(x) == false);
         |           ^~~~~~~~~~~~~~~~
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/arma_cmath.hpp:96:11: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
      96 |   return (std::isfinite(x) == false);
         |           ^~~~~~~~~~~~~~~~
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/arma_cmath.hpp:130:10: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     130 |   return std::isinf(x);
         |          ^~~~~~~~~~~~~
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/arma_cmath.hpp:140:10: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     140 |   return std::isinf(x);
         |          ^~~~~~~~~~~~~
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/arma_cmath.hpp:176:10: warning: use of NaN is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     176 |   return std::isnan(x);
         |          ^~~~~~~~~~~~~
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/arma_cmath.hpp:186:10: warning: use of NaN is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     186 |   return std::isnan(x);
         |          ^~~~~~~~~~~~~
   capybara.cpp:285:22: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     285 |       tobit_lower = -std::numeric_limits<double>::infinity();
         |                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   capybara.cpp:292:21: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     292 |       tobit_upper = std::numeric_limits<double>::infinity();
         |                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   In file included from capybara.cpp:337:
   ./01_04_center_berge.h:503:20: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     503 |               if (!std::isfinite(y[i])) {
         |                    ^~~~~~~~~~~~~~~~~~~
   In file included from capybara.cpp:357:
   ./08_glm.h:134:9: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     134 |         std::isfinite(lower) && std::fabs(yi - lower) < eps;
         |         ^~~~~~~~~~~~~~~~~~~~
   ./08_glm.h:136:9: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     136 |         std::isfinite(upper) && std::fabs(yi - upper) < eps;
         |         ^~~~~~~~~~~~~~~~~~~~
   ./08_glm.h:201:9: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     201 |         std::isfinite(lower) && std::fabs(yi - lower) < eps;
         |         ^~~~~~~~~~~~~~~~~~~~
   ./08_glm.h:203:9: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     203 |         std::isfinite(upper) && std::fabs(yi - upper) < eps;
         |         ^~~~~~~~~~~~~~~~~~~~
   ./08_glm.h:1165:14: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
    1165 |         if (!std::isfinite(ww_ptr[i]) || !std::isfinite(z_ptr[i])) {
         |              ^~~~~~~~~~~~~~~~~~~~~~~~
   ./08_glm.h:1165:43: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
    1165 |         if (!std::isfinite(ww_ptr[i]) || !std::isfinite(z_ptr[i])) {
         |                                           ^~~~~~~~~~~~~~~~~~~~~~~
   ./08_glm.h:1280:18: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
    1280 |       dev_crit = std::isfinite(dev);
         |                  ^~~~~~~~~~~~~~~~~~
   ./08_glm.h:1097:9: warning: variable 'convergence_count' set but not used [-Wunused-but-set-variable]
    1097 |   uword convergence_count = 0;
         |         ^
   ./08_glm.h:1751:18: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
    1751 |       dev_crit = std::isfinite(dev);
         |                  ^~~~~~~~~~~~~~~~~~
   In file included from capybara.cpp:359:
   ./09_negbin.h:138:30: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     138 |     if (theta_new <= 0.0 || !std::isfinite(theta_new)) {
         |                              ^~~~~~~~~~~~~~~~~~~~~~~~
   In file included from capybara.cpp:360:
   ./10_fepoisson_asymmetric.h:80:10: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
      80 |     if (!std::isfinite(eta_work(i))) {
         |          ^~~~~~~~~~~~~~~~~~~~~~~~~~
   ./10_fepoisson_asymmetric.h:83:10: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
      83 |     if (!std::isfinite(mu_work(i)) || mu_work(i) <= 0.0) {
         |          ^~~~~~~~~~~~~~~~~~~~~~~~~
   ./10_fepoisson_asymmetric.h:102:15: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     102 |   double cv = std::numeric_limits<double>::infinity();
         |               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ./10_fepoisson_asymmetric.h:326:11: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     326 |       if (std::isfinite(result.residuals(i)) && result.residuals(i) < 0.0) {
         |           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   capybara.cpp:1231:12: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
    1231 |       if (!std::isfinite(safe_eta))
         |            ^~~~~~~~~~~~~~~~~~~~~~~
   capybara.cpp:1302:39: warning: braces around scalar initializer [-Wbraced-scalar-init]
    1302 |        "conv"_nm = writable::logicals({result.conv}),
         |                                       ^~~~~~~~~~~~~
   capybara.cpp:1314:61: warning: braces around scalar initializer [-Wbraced-scalar-init]
    1314 |     out.push_back({"has_separation"_nm = writable::logicals({true})});
         |                                                             ^~~~~~
   capybara.cpp:1370:55: warning: braces around scalar initializer [-Wbraced-scalar-init]
    1370 |     out.push_back({"has_apes"_nm = writable::logicals({true})});
         |                                                       ^~~~~~
   capybara.cpp:1376:60: warning: braces around scalar initializer [-Wbraced-scalar-init]
    1376 |     out.push_back({"has_bias_corr"_nm = writable::logicals({true})});
         |                                                            ^~~~~~
   capybara.cpp:1641:12: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
    1641 |       if (!std::isfinite(safe_eta))
         |            ^~~~~~~~~~~~~~~~~~~~~~~
   capybara.cpp:1714:39: warning: braces around scalar initializer [-Wbraced-scalar-init]
    1714 |        "conv"_nm = writable::logicals({result.conv}),
         |                                       ^~~~~~~~~~~~~
   capybara.cpp:1726:61: warning: braces around scalar initializer [-Wbraced-scalar-init]
    1726 |     out.push_back({"has_separation"_nm = writable::logicals({true})});
         |                                                             ^~~~~~
   capybara.cpp:1781:55: warning: braces around scalar initializer [-Wbraced-scalar-init]
    1781 |     out.push_back({"has_apes"_nm = writable::logicals({true})});
         |                                                       ^~~~~~
   capybara.cpp:1787:60: warning: braces around scalar initializer [-Wbraced-scalar-init]
    1787 |     out.push_back({"has_bias_corr"_nm = writable::logicals({true})});
         |                                                            ^~~~~~
   capybara.cpp:1931:39: warning: braces around scalar initializer [-Wbraced-scalar-init]
    1931 |        "conv"_nm = writable::logicals({result.conv}),
         |                                       ^~~~~~~~~~~~~
   capybara.cpp:1936:45: warning: braces around scalar initializer [-Wbraced-scalar-init]
    1936 |        "conv_outer"_nm = writable::logicals({result.conv_outer})});
         |                                             ^~~~~~~~~~~~~~~~~~~
   capybara.cpp:2106:39: warning: braces around scalar initializer [-Wbraced-scalar-init]
    2106 |        "conv"_nm = writable::logicals({result.conv}),
         |                                       ^~~~~~~~~~~~~
   capybara.cpp:2111:45: warning: braces around scalar initializer [-Wbraced-scalar-init]
    2111 |        "conv_outer"_nm = writable::logicals({result.conv_outer}),
         |                                             ^~~~~~~~~~~~~~~~~~~
   capybara.cpp:2124:61: warning: braces around scalar initializer [-Wbraced-scalar-init]
    2124 |     out.push_back({"has_separation"_nm = writable::logicals({true})});
         |                                                             ^~~~~~
   In file included from capybara.cpp:8:
   In file included from /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r.hpp:52:
   In file included from /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo.hpp:161:
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/constants.hpp:76:59: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
      76 |       return (std::numeric_limits<eT>::has_infinity) ? eT(std::numeric_limits<eT>::infinity()) : eT(std::numeric_limits<eT>::max());
         |                                                           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/constants.hpp:226:77: note: in instantiation of function template specialization 'arma::priv::Datum_helper::pos_inf<double>' requested here
     226 | template<typename eT> const eT Datum<eT>::inf         = priv::Datum_helper::pos_inf<eT>();
         |                                                                             ^
   ./01_03_center_stammann.h:50:27: note: in instantiation of static data member 'arma::Datum<double>::inf' requested here
      50 |   double ssr_old = datum::inf;
         |                           ^
   In file included from capybara.cpp:8:
   In file included from /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r.hpp:52:
   In file included from /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo.hpp:161:
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/constants.hpp:117:60: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     117 |       return (std::numeric_limits<eT>::has_infinity) ? eT(-std::numeric_limits<eT>::infinity()) : eT(std::numeric_limits<eT>::lowest());
         |                                                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/Mat_meat.hpp:7935:93: note: in instantiation of function template specialization 'arma::priv::Datum_helper::neg_inf<double>' requested here
    7935 |   if(is_same_type<fill_type, fill::fill_neg_inf>::yes)  { (*this).fill( priv::Datum_helper::neg_inf<eT>() ); }
         |                                                                                             ^
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/Col_meat.hpp:171:11: note: in instantiation of function template specialization 'arma::Mat<double>::fill<arma::fill::fill_none>' requested here
     171 |   (*this).fill(f);
         |           ^
   ./03_beta.h:21:9: note: in instantiation of function template specialization 'arma::Col<double>::Col<arma::fill::fill_none>' requested here
      21 |       : coefficients(p, fill::none), fitted_values(n, fill::none),
         |         ^
   In file included from capybara.cpp:8:
   In file included from /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r.hpp:52:
   In file included from /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo.hpp:161:
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/constants.hpp:297:54: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     297 |     return (std::numeric_limits<eT>::has_infinity) ? std::numeric_limits<eT>::infinity() : std::numeric_limits<eT>::max();
         |                                                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/op_min_meat.hpp:382:24: note: in instantiation of function template specialization 'arma::priv::most_pos<double>' requested here
     382 |   eT min_val_i = priv::most_pos<eT>();
         |                        ^
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/op_min_meat.hpp:96:30: note: in instantiation of function template specialization 'arma::op_min::direct_min<double>' requested here
      96 |       out_mem[col] = op_min::direct_min( X.colptr(col), X_n_rows );
         |                              ^
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/op_min_meat.hpp:68:11: note: in instantiation of function template specialization 'arma::op_min::apply_noalias<double>' requested here
      68 |   op_min::apply_noalias(out, U.M, dim);
         |           ^
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/Mat_meat.hpp:5291:12: note: in instantiation of function template specialization 'arma::op_min::apply<arma::Mat<double>>' requested here
    5291 |   op_type::apply(static_cast< Mat_noalias<eT>& >(*this), X);
         |            ^
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/Row_meat.hpp:524:5: note: in instantiation of function template specialization 'arma::Mat<double>::Mat<arma::Mat<double>, arma::op_min>' requested here
     524 |   : Mat<eT>(X.get_ref(), arma_vec_indicator(), 2)
         |     ^
   ./05_03_separation_simplex.h:226:21: note: in instantiation of function template specialization 'arma::Row<double>::Row<arma::Op<arma::Mat<double>, arma::op_min>>' requested here
     226 |   rowvec col_mins = min(residuals, 0);
         |                     ^
   In file included from capybara.cpp:8:
   In file included from /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r.hpp:52:
   In file included from /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo.hpp:161:
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/constants.hpp:277:56: warning: use of infinity is undefined behavior due to the currently enabled floating-point options [-Wnan-infinity-disabled]
     277 |     return (std::numeric_limits<eT>::has_infinity) ? -(std::numeric_limits<eT>::infinity()) : std::numeric_limits<eT>::lowest();
         |                                                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/op_max_meat.hpp:382:24: note: in instantiation of function template specialization 'arma::priv::most_neg<double>' requested here
     382 |   eT max_val_i = priv::most_neg<eT>();
         |                        ^
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/op_max_meat.hpp:96:30: note: in instantiation of function template specialization 'arma::op_max::direct_max<double>' requested here
      96 |       out_mem[col] = op_max::direct_max( X.colptr(col), X_n_rows );
         |                              ^
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/op_max_meat.hpp:68:11: note: in instantiation of function template specialization 'arma::op_max::apply_noalias<double>' requested here
      68 |   op_max::apply_noalias(out, U.M, dim);
         |           ^
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/Mat_meat.hpp:5291:12: note: in instantiation of function template specialization 'arma::op_max::apply<arma::Mat<double>>' requested here
    5291 |   op_type::apply(static_cast< Mat_noalias<eT>& >(*this), X);
         |            ^
   /home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include/armadillo4r/armadillo/Row_meat.hpp:524:5: note: in instantiation of function template specialization 'arma::Mat<double>::Mat<arma::Mat<double>, arma::op_max>' requested here
     524 |   : Mat<eT>(X.get_ref(), arma_vec_indicator(), 2)
         |     ^
   ./05_03_separation_simplex.h:227:21: note: in instantiation of function template specialization 'arma::Row<double>::Row<arma::Op<arma::Mat<double>, arma::op_max>>' requested here
     227 |   rowvec col_maxs = max(residuals, 0);
         |                     ^
   50 warnings generated.
   /usr/bin/clang++ -std=gnu++20 -I"/opt/R-4.6.0/lib64/R/include" -DNDEBUG  -I'/home/pacha/R/x86_64-pc-linux-gnu-library/4.6/armadillo4r/include' -I'/home/pacha/R/x86_64-pc-linux-gnu-library/4.6/cpp4r/include' -I/usr/local/include   -fopenmp -O3 -ffast-math -march=native -mtune=native -mavx -mavx2 -mfma -funroll-loops -fprefetch-loop-arrays -ftree-vectorize -ftree-slp-vectorize -flto -DARMA_NO_DEBUG -DARMA_USE_BLAS -DARMA_USE_LAPACK -DARMA_USE_OPENMP -DARMA_OPENMP_THREADS=4 -DARMA_64BIT_WORD -DARMA_DONT_USE_WRAPPER -DARMADILLO4R_NO_SPARSE -DCAPYBARA_DEFAULT_OMP_THREADS=4 -fpic  -Wall -O3 -pedantic -fno-lto -Wno-ignored-optimization-argument -UNDEBUG -Wall -pedantic -g -O0 -fdiagnostics-color=always  -c cpp4r.cpp -o cpp4r.o
   /usr/bin/clang++ -std=gnu++20 -shared -L/opt/R-4.6.0/lib64/R/lib -L/usr/local/lib64 -o capybara.so capybara.o cpp4r.o -fopenmp -lopenblas -lgfortran -lm -latomic_asneeded -lquadmath -L/opt/R-4.6.0/lib64/R/lib -lR
   installing to /tmp/RtmpXhDR5v/devtools_install_10a9d686d1e8f/00LOCK-capybara/00new/capybara/libs
   ** checking absolute paths in shared objects and dynamic libraries
─  DONE (capybara)
