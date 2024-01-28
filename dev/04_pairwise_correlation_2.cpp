// #include <omp.h>

#include <algorithm>
#include <cpp11.hpp>

using namespace cpp11;
using namespace std;

// Bubble sort distance between the input array and the sorted array

uint64_t insert_sort_(double* arr, size_t len) {
  size_t max_j, i;
  uint64_t swap_count = 0;

  if (len < 2) {
    return 0;
  }

  max_j = len - 1;
  for (i = len - 2; i < len; --i) {
    size_t j = i;
    double val = arr[i];

    for (; j < max_j && arr[j + 1] < val; ++j) {
      arr[j] = arr[j + 1];
    }

    arr[j] = val;
    swap_count += (j - i);
  }

  return swap_count;
}

static uint64_t merge_(double* from, double* to, size_t middle, size_t len) {
  size_t buf_index, left_len, right_len;
  uint64_t swaps;
  double* left;
  double* right;

  buf_index = 0;
  swaps = 0;

  left = from;
  right = from + middle;
  right_len = len - middle;
  left_len = middle;

  while (left_len && right_len) {
    if (right[0] < left[0]) {
      to[buf_index] = right[0];
      swaps += left_len;
      right_len--;
      right++;
    } else {
      to[buf_index] = left[0];
      left_len--;
      left++;
    }
    buf_index++;
  }

  if (left_len) {
    memcpy(to + buf_index, left, left_len * sizeof(double));
  } else if (right_len) {
    memcpy(to + buf_index, right, right_len * sizeof(double));
  }

  return swaps;
}

uint64_t merge_sort_(double* x, double* buf, size_t len) {
  uint64_t swaps;
  size_t half;

  if (len < 10) {
    return insert_sort_(x, len);
  }

  swaps = 0;

  if (len < 2) {
    return 0;
  }

  half = len / 2;
  swaps += merge_sort_(x, buf, half);
  swaps += merge_sort_(x + half, buf + half, len - half);
  swaps += merge_(x, buf, half, len);

  memcpy(x, buf, len * sizeof(double));
  return swaps;
}

// Count ties

static uint64_t count_ties_(double* data,
                            size_t len) { /* Assumes data is sorted.*/
  uint64_t sum_counts = 0, tie_count = 0;
  size_t i;

  for (i = 1; i < len; i++) {
    if (data[i] == data[i - 1]) {
      tie_count++;
    } else if (tie_count) {
      sum_counts += (tie_count * (tie_count + 1)) / 2;
      tie_count++;
      tie_count = 0;
    }
  }
  if (tie_count) {
    sum_counts += (tie_count * (tie_count + 1)) / 2;
    tie_count++;
  }
  return sum_counts;
}

// Kendall correlation

// Assumes arr1 has already been sorted and arr2 has already been reordered in
// lockstep.

// This can be done within R before calling this function with:
// perm <- order(arr1)
// arr1 <- arr1[perm]
// arr2 <- arr2[perm]

double kendall_n_log_n_(const double* arr1, const double* arr2, size_t len) {
  uint64_t m1 = 0, m2 = 0, tie_count, swap_count, n_pair;
  int64_t s;
  size_t i;

  n_pair = (uint64_t)len * ((uint64_t)len - 1) / 2;
  s = n_pair;

  tie_count = 0;
  for (i = 1; i < len; i++) {
    if (arr1[i - 1] == arr1[i]) {
      tie_count++;
    } else if (tie_count > 0) {
      insert_sort_(const_cast<double*>(arr2) + i - tie_count - 1,
                   tie_count + 1);
      m1 += tie_count * (tie_count + 1) / 2;
      s += count_ties_(const_cast<double*>(arr2) + i - tie_count - 1,
                       tie_count + 1);
      tie_count++;
      tie_count = 0;
    }
  }
  if (tie_count > 0) {
    insert_sort_(const_cast<double*>(arr2) + i - tie_count - 1, tie_count + 1);
    m1 += tie_count * (tie_count + 1) / 2;
    s += count_ties_(const_cast<double*>(arr2) + i - tie_count - 1,
                     tie_count + 1);
    tie_count++;
  }

  swap_count =
      merge_sort_(const_cast<double*>(arr2), const_cast<double*>(arr1), len);

  m2 = count_ties_(const_cast<double*>(arr2), len);
  s -= (m1 + m2) + 2 * swap_count;

  double denominator1 = n_pair - m1;
  double denominator2 = n_pair - m2;
  double cor = s / sqrt(denominator1) / sqrt(denominator2);
  return cor;
}

// Wrapper to R

[[cpp11::register]] double pairwise_cor_(const doubles& y,
                                         const doubles& yhat) {
  size_t len_y = y.size();
  size_t len_yhat = yhat.size();

  if (len_y != len_yhat) {
    stop("y and yhat must be the same length");
  }

  const double* y_data = REAL(y);
  const double* yhat_data = REAL(yhat);

  double cor = kendall_n_log_n_(y_data, yhat_data, len_y);

  return cor;
}
