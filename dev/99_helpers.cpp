// maybe add these directly to cpp11armadillo

#include "00_main.h"

uvec as_uvec(const cpp11::integers &x) {
  uvec res(x.size());
  std::copy(x.begin(), x.end(), res.begin());
  return res;
}
