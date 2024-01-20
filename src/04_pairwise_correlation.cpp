#include "00_main.h"

// Kendall's tau using inversion counting

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
  // https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
  // cor = 1 - 2 * "number of non-concordant pairs" / "factorial of n over 2"
  // time complexity: O(n * log(n)) using trees to count inversions

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
