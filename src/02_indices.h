#ifndef CAPYBARA_INDICES_H
#define CAPYBARA_INDICES_H

// Convert a nested R list of group indices to an indices_info structure
inline indices_info list_to_indices_info(const list &k_list) {
  const uword K = static_cast<uword>(k_list.size());
  indices_info info;

  size_t total_indices = 0;
  size_t total_groups = 0;

  std::vector<uword> group_sizes_temp;
  group_sizes_temp.reserve(K * 10);

  for (uword k = 0; k < K; ++k) {
    const list &jlist = k_list[static_cast<std::size_t>(k)];
    const uword J = static_cast<uword>(jlist.size());
    total_groups += J;

    for (uword j = 0; j < J; ++j) {
      const integers idx = jlist[static_cast<std::size_t>(j)];
      const uword sz = static_cast<uword>(idx.size());
      group_sizes_temp.push_back(sz);
      total_indices += sz;
    }
  }

  info.all_indices.set_size(total_indices);
  info.group_offsets.set_size(total_groups);
  info.group_sizes.set_size(total_groups);
  info.fe_offsets.set_size(K);
  info.fe_sizes.set_size(K);

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

      if (sz > 0) {
        for (size_t i = 0; i < sz; ++i) {
          info.all_indices(current_index + i) = static_cast<uword>(idx[i]);
        }
      }

      current_index += sz;
      ++current_group;
    }
  }

  info.compute_nonempty_groups();
  info.precompute_all_groups();
  return info;
}

// Convert a single-level R list of group indices to a single_fe_indices
// structure
inline single_fe_indices list_to_single_fe_indices(const list &jlist) {
  const uword J = static_cast<uword>(jlist.size());
  single_fe_indices info;

  uword total_indices = 0;
  std::vector<uword> sizes(J);

  for (uword j = 0; j < J; ++j) {
    const integers idx = jlist[static_cast<std::size_t>(j)];
    sizes[j] = static_cast<uword>(idx.size());
    total_indices += sizes[j];
  }

  info.all_indices.set_size(total_indices);
  info.group_offsets.set_size(J);
  info.group_sizes.set_size(J);

  uword offset = 0;
  for (uword j = 0; j < J; ++j) {
    const integers idx = jlist[static_cast<std::size_t>(j)];
    const uword sz = sizes[j];

    info.group_offsets(j) = offset;
    info.group_sizes(j) = sz;

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
