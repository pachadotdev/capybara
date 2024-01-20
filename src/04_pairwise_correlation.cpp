#include "00_main.h"

[[cpp11::register]] double pairwise_cor_(const doubles& y,
                                         const doubles& yhat) {
  // time complexity: O(n^2)
  // https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient

  // concordant:
  // Any pair of observations (xi, yi) and (xj, yj), where i<j,
  // are said to be concordant if the sort order of (xi, yi) and (yi, yj)
  // agrees: that is, if either both xi > xj and yi > yj holds or both xi < xj
  // and yi < yj

  // explicit expression
  // cor = 2/(n(n-1)) * sum_{i<j} sign(y_i - y_j) * sign(yhat_i - yhat_j)

  // Auxiliary variables

  int n = y.size();
  int sign_y;
  int sign_yhat;
  int sum_signs = 0;

  // Computation

  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (y[i] > y[j]) {
        sign_y = 1;
      } else if (y[i] < y[j]) {
        sign_y = -1;
      } else {
        sign_y = 0;
      }

      if (yhat[i] > yhat[j]) {
        sign_yhat = 1;
      } else if (yhat[i] < yhat[j]) {
        sign_yhat = -1;
      } else {
        sign_yhat = 0;
      }

      sum_signs += sign_y * sign_yhat;
    }
  }

  double cor = (2.0 / (n * (n - 1))) * sum_signs;

  return cor;
}
