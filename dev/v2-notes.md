Acceleration / Centering notes for capybara 2.0

01_center.h changes:
1. Hestenes-Stiefel formula: Instead of Fletcher-Reeves for computing beta, which is more stable for MAP methods according to reghdfe's comments.
2. Gradient computation: The gradient g is computed as x_old - x_new after projection, which represents the change made by the projection.
3. Line search: The line search computes alpha = <g,g> / <p,Ap> where Ap = (I-P)*p.
4. Restart conditions: CG is restarted when beta is negative or too large (>10), or when the denominator is too small.
5. Acceleration start: CG acceleration starts after a warmup period (default 6 iterations, matching reghdfe).
6. Optional CG: Added a use_cg parameter to allow switching between CG and Irons-Tuck acceleration.
7. Memory efficiency: The vectors are allocated once outside the loop and reused.
8. The implementation follows reghdfe's approach closely while adapting it to the C++/Armadillo environment. The default behavior still uses Irons-Tuck (when use_cg=false), but CG can be enabled for potentially better convergence.
9. Important: v1.x used Halperin projections with Irons-Tuck acceleration, while v2.x uses Symmetric Kaczmarz (i.e., the same with a back pass) projections with CG acceleration.
