// this part heavily borrows from
// Knight 1966 (A Computer Method for Calculating Kendall's Tau with Ungrouped
// Data)
// Filzmoser, Fritz, and Kalcher 2023 (pcaPP package)

// note: the len < 2 conditions are commented out because the R function checks
// for this condition before calling the C++ functions

#include "00_main.h"

uint64_t insertion_sort_(double *arr, size_t len) {
  // if (len < 2) {
  //   return 0;
  // }

  size_t maxJ = len - 1, i;
  uint64_t swapCount = 0;

  for (i = len - 2; i < len; --i) {
    size_t j = i;
    double val = arr[i];

    while (j < maxJ && arr[j + 1] < val) {
      arr[j] = arr[j + 1];
      ++j;
    }

    arr[j] = val;
    swapCount += (j - i);
  }

  return swapCount;
}

static uint64_t merge_(double *from, double *to, size_t middle, size_t len) {
  size_t bufIndex = 0, leftLen, rightLen;
  uint64_t swaps = 0;
  double *left;
  double *right;

  left = from;
  right = from + middle;
  rightLen = len - middle;
  leftLen = middle;

  while (leftLen && rightLen) {
    if (right[0] < left[0]) {
      to[bufIndex] = right[0];
      swaps += leftLen;
      rightLen--;
      right++;
    } else {
      to[bufIndex] = left[0];
      leftLen--;
      left++;
    }
    bufIndex++;
  }

  if (leftLen) {
    memcpy(to + bufIndex, left, leftLen * sizeof(double));
  } else if (rightLen) {
    memcpy(to + bufIndex, right, rightLen * sizeof(double));
  }

  return swaps;
}

uint64_t merge_sort_(double *x, double *buf, size_t len) {
  // if (len < 2) {
  //   return 0;
  // }

  if (len < 10) {
    return insertion_sort_(x, len);
  }

  uint64_t swaps = 0;
  size_t half = len / 2;

  swaps += merge_sort_(x, buf, half);
  swaps += merge_sort_(x + half, buf + half, len - half);
  swaps += merge_(x, buf, half, len);

  memcpy(x, buf, len * sizeof(double));
  return swaps;
}

[[cpp11::register]] double kendall_cor_(const doubles_matrix<> &m) {
  size_t len = m.nrow();
  std::vector<double> arr1(len), arr2(len);
  std::vector<double> buf(len);
  uint64_t m1 = 0, m2 = 0, tieCount, swapCount, nPair;
  int64_t s;

  // Sort the 1st vector and rearrange the 2nd vector accordingly
  std::vector<size_t> perm(len);
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(),
            [&](size_t i, size_t j) { return m(i, 0) < m(j, 0); });
  for (size_t i = 0; i < len; i++) {
    arr1[i] = m(perm[i], 0);
    arr2[i] = m(perm[i], 1);
  }

  // Compute nPair and initialize s
  nPair = static_cast<uint64_t>(len) * (static_cast<uint64_t>(len) - 1) / 2;
  s = nPair;

  // Compute m1
  tieCount = 0;
  for (size_t i = 1; i < len; i++) {
    if (arr1[i] == arr1[i - 1]) {
      tieCount++;
    } else if (tieCount > 0) {
      m1 += tieCount * (tieCount + 1) / 2;
      tieCount = 0;
    }
  }
  if (tieCount > 0) {
    m1 += tieCount * (tieCount + 1) / 2;
  }

  swapCount = merge_sort_(arr2.data(), buf.data(), len);

  // Compute m2
  m2 = 0;
  tieCount = 0;
  for (size_t i = 1; i < len; i++) {
    if (arr2[i] == arr2[i - 1]) {
      tieCount++;
    } else if (tieCount) {
      m2 += (tieCount * (tieCount + 1)) / 2;
      tieCount = 0;
    }
  }
  if (tieCount) {
    m2 += (tieCount * (tieCount + 1)) / 2;
  }

  // Adjust for ties
  s -= (m1 + m2) + 2 * swapCount;

  return (s / std::sqrt(nPair - m1) / std::sqrt(nPair - m2));
}

double ckendall_(int k, int n, std::vector<std::vector<double>> &w) {
  int u = n * (n - 1) / 2;
  if (k < 0 || k > u) {
    return 0;
  }
  if (w[n][k] < 0) {
    if (n == 1)
      w[n][k] = (k == 0) ? 1 : 0;
    else {
      double s = 0;
      for (int i = 0; i <= u; i++) {
        s += ckendall_(k - i, n - 1, w);
      }
      w[n][k] = s;
    }
  }
  return w[n][k];
}

[[cpp11::register]] doubles pkendall_(doubles Q, int n) {
  int len = Q.size();
  writable::doubles P(len);
  std::vector<std::vector<double>> w(
      n + 1, std::vector<double>((n * (n - 1) / 2) + 1, -1));

  for (int i = 0; i < len; i++) {
    double q = std::floor(Q[i] + 1e-7);
    if (q < 0)
      P[i] = 0;
    else if (q > n * (n - 1) / 2)
      P[i] = 1;
    else {
      double p = 0;
      for (int j = 0; j <= q; j++) {
        p += ckendall_(j, n, w);
      }
      P[i] = p / gammafn(n + 1);
    }
  }
  return P;
}
