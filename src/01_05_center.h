#ifndef CAPYBARA_CENTER_H
#define CAPYBARA_CENTER_H

namespace capybara {
inline void center_variables(mat &V, const vec &w, FlatFEMap &map, double tol,
                             uword max_iter, uword grand_acc_period,
                             CenterWarmStart *warm = nullptr,
                             CenteringMethod method = STAMMANN) {
#ifdef _OPENMP
  set_omp_threads_from_config();
#endif
  if (V.is_empty() || map.K == 0)
    return;

  CenterWarmStart local_warm;
  CenterWarmStart &ws = warm ? *warm : local_warm;

  if (method == STAMMANN) {
    if (map.K == 2) {
      center_2fe_stammann(V, w, map, ws, tol, max_iter, grand_acc_period);
    } else {
      center_kfe_stammann(V, w, map, ws, tol, max_iter, grand_acc_period);
    }
  } else {
    if (map.K == 2) {
      center_2fe_berge(V, w, map, ws, tol, max_iter, grand_acc_period);
    } else {
      center_kfe_berge(V, w, map, ws, tol, max_iter, grand_acc_period);
    }
  }
}

} // namespace capybara

#endif // CAPYBARA_CENTER_H
