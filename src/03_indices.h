#ifndef CAPYBARA_INDICES_H
#define CAPYBARA_INDICES_H

inline indices_info list_to_indices_info(const list &k_list) {
  const uword K = static_cast<uword>(k_list.size());
  indices_info info;

  // First pass: compute totals (pre-allocate vectors once)
  size_t total_indices = 0;
  size_t total_groups = 0;

  // Reserve vectors to avoid repeated reallocation in the loop
  // guess for average number of groups
  // TODO: get the actual number
  std::vector<uword> group_sizes_temp;
  group_sizes_temp.reserve(K * 10);

  for (uword k = 0; k < K; ++k) {
    const list &jlist = k_list[static_cast<std::size_t>(k)];
    const uword J = static_cast<uword>(jlist.size());
    total_groups += J;

    // Pre-record group sizes for second pass
    for (uword j = 0; j < J; ++j) {
      const integers idx = jlist[static_cast<std::size_t>(j)];
      const uword sz = static_cast<uword>(idx.size());
      group_sizes_temp.push_back(sz);
      total_indices += sz;
    }
  }

  // Allocate memory only once with exact sizes
  info.all_indices.set_size(total_indices);
  info.group_offsets.set_size(total_groups);
  info.group_sizes.set_size(total_groups);
  info.fe_offsets.set_size(K);
  info.fe_sizes.set_size(K);

  // Second pass: fill data efficiently
  uword current_index = 0;
  uword current_group = 0;
  size_t group_idx = 0;

  for (uword k = 0; k < K; ++k) {
    const list &jlist = k_list[static_cast<std::size_t>(k)];
    const uword J = static_cast<uword>(jlist.size());

    info.fe_offsets(k) = current_group;
    info.fe_sizes(k) = J;

    for (uword j = 0; j < J; ++j) {
      const integers idx = jlist[static_cast<std::size_t>(j)];
      const uword sz = group_sizes_temp[group_idx++];

      info.group_offsets(current_group) = current_index;
      info.group_sizes(current_group) = sz;

      // Copy contiguous data
      if (sz > 0) {
        for (size_t i = 0; i < sz; ++i) {
          info.all_indices(current_index + i) = static_cast<uword>(idx[i]);
        }
      }

      current_index += sz;
      ++current_group;
    }
  }

  // Compute non-empty groups
  info.compute_nonempty_groups();
  info.precompute_all_groups();
  return info;
}

inline single_fe_indices list_to_single_fe_indices(const list &jlist) {
  const uword J = static_cast<uword>(jlist.size());
  single_fe_indices info;

  // First pass: gather sizes more efficiently
  uword total_indices = 0;
  std::vector<uword> sizes(J);

  for (uword j = 0; j < J; ++j) {
    const integers idx = jlist[static_cast<std::size_t>(j)];
    sizes[j] = static_cast<uword>(idx.size());
    total_indices += sizes[j];
  }

  // Allocate all memory at once
  info.all_indices.set_size(total_indices);
  info.group_offsets.set_size(J);
  info.group_sizes.set_size(J);

  // Second pass: fill data with minimal intermediate copies
  uword offset = 0;
  for (uword j = 0; j < J; ++j) {
    const integers idx = jlist[static_cast<std::size_t>(j)];
    const uword sz = sizes[j];

    info.group_offsets(j) = offset;
    info.group_sizes(j) = sz;

    // Direct copying without creating intermediates
    if (sz > 0) {
      for (size_t i = 0; i < sz; ++i) {
        info.all_indices(offset + i) = static_cast<uword>(idx[i]);
      }
    }

    offset += sz;
  }

  return info;
}

#endif // CAPYBARA_INDICES_H
