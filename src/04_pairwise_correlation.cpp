#include "00_main.h"

// https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient

// concordant:
// Any pair of observations (xi, yi) and (xj, yj), where i<j,
// are said to be concordant if the sort order of (xi, yi) and (yi, yj)
// agrees: that is, if either both xi > xj and yi > yj holds or both xi < xj
// and yi < yj

// test:
// fit <- capybara::fepoisson(mpg ~ wt + disp | am, data = mtcars)
// x <- mtcars$mpg
// y <- predict(fit, type = "response")
// pairwise_cor_base_ = 0.7580645
// pairwise_cor_ = 0.7741935
// base R = 0.7642418

// cor = 2/(n(n-1)) * sum_{i<j} sign(y_i - y_j) * sign(yhat_i - yhat_j)
// time complexity: O(n^2)

// [[cpp11::register]]
double pairwise_cor_base_(const doubles& y, const doubles& yhat) {
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

// cor = 1 - 2 * "number of non-concordant pairs" / "factorial of n over 2"
// time complexity: O(n * log(n)) using trees to count inversions

class fenwick_tree_ {
  vector<int> BIT;
  int N;

 public:
  fenwick_tree_(int n) : BIT(n + 1, 0), N(n) {}

  void update(int idx, int val) {
    for (; idx <= N; idx += idx & -idx) {
      BIT[idx] += val;
    }
  }

  int query(int idx) {
    int sum = 0;
    for (; idx > 0; idx -= idx & -idx) {
      sum += BIT[idx];
    }
    return sum;
  }

  int query(int l, int r) { return query(r) - query(l - 1); }
};

[[cpp11::register]] double pairwise_cor_(const doubles& y,
                                         const doubles& yhat) {
  int n = y.size();
  vector<pair<double, double>> vec(n);
  for (int i = 0; i < n; ++i) {
    vec[i] = {y[i], yhat[i]};
  }

  sort(vec.begin(), vec.end());

  // Map the yhat values to the range [1, n]
  vector<double> yhat_values(n);
  for (int i = 0; i < n; ++i) {
    yhat_values[i] = vec[i].second;
  }
  sort(yhat_values.begin(), yhat_values.end());
  for (int i = 0; i < n; ++i) {
    vec[i].second =
        lower_bound(yhat_values.begin(), yhat_values.end(), vec[i].second) -
        yhat_values.begin() + 1;
  }

  // Count inversions
  fenwick_tree_ tree(n);
  double inversions = 0;
  for (int i = n - 1; i >= 0; --i) {
    inversions += tree.query(vec[i].second);
    tree.update(vec[i].second, 1);
  }

  return 1 - 4 * inversions / (n * (n - 1));
}
