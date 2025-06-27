#ifndef CAPYBARA_CENTER_WORKSPACE_H
#define CAPYBARA_CENTER_WORKSPACE_H

struct center_workspace {
  // Basic projection workspace vectors
  vec x;      // Current iterate
  vec x0;     // Previous iterate
  vec Gx;     // Single projection step G(x)
  vec G2x;    // Double projection step G(G(x))
  vec deltaG; // Difference for acceleration
  vec delta2; // Second difference for acceleration

  // Group structure storage
  field<field<uvec>> group_indices; // Indices for each group in each FE
  field<vec> group_inv_w;           // Inverse weights for each group
  vec group_means;                  // Temporary storage for group means
  double ratio0;                    // Convergence ratio tracking
  double ssr0;                      // Sum of squares tracking
  size_t max_groups;                // Maximum number of groups across FEs

  // Irons-Tuck acceleration workspace for K>=2 systems
  mat acceleration_history;       // Store last few iterations for enhanced
                                  // acceleration
  vec acceleration_weights;       // Adaptive weights for combining history
  size_t history_size;            // Number of previous iterates to store
  size_t history_pos;             // Current position in circular buffer
  bool use_enhanced_acceleration; // Whether to use enhanced vs memory-efficient
                                  // mode
  double acceleration_damping;    // Damping factor for stability (0.7-0.8)
  double min_acceleration_norm;   // Minimum norm threshold for acceleration

  center_workspace()
      : ratio0(datum::inf), ssr0(datum::inf), max_groups(0), history_size(3),
        history_pos(0), use_enhanced_acceleration(false),
        acceleration_damping(0.8), min_acceleration_norm(1e-12) {}
};

#endif // CAPYBARA_CENTER_WORKSPACE_H
