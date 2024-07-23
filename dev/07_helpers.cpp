#include "00_main.h"

// Generate auxiliary list of indexes for different sub panels

[[cpp11::register]] list get_index_list_(const strings &k_vars,
                                         const data_frame &data) {
  writable::integers indexes(data.nrow());
  std::iota(indexes.begin(), indexes.end(), 0);

  writable::list out;

  auto split = cpp11::package("base")["split"];

  for (const auto &k_var : k_vars) {
    out.push_back(split(indexes, data[k_var]));
  }

  return out;
}
