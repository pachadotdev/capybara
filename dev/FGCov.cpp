// Draft code from Julian Costas-Fernandez
// Fully added to src/ using a different formulation

// dyad_cov_from_R.cpp
// [[Rcpp::depends(RcppArmadillo, RcppProgress)]]
#include <RcppArmadillo.h>
#include <omp.h>
#include <progress.hpp>

struct FGCov {
  const arma::mat &A_inv;
  const arma::uvec &from_id, to_id;
  const arma::uword N, K, G;
  arma::mat M;
  arma::uvec sec_nodes;
  const bool verbose;

public:
  /////////////////////////////////////////////////////////////////////////
  // Constructor: Main usefulness is to share objectes with struct methods
  // and reorder conditional moments so that neighbours can be cheaply found
  // through indexing
  FGCov(const arma::mat &A_inv_, const arma::mat &X_, const arma::vec &u_hat_,
        const arma::uvec &from_id_, const arma::uvec &to_id_,
        const bool verbose_ = false, const int ncores = 1)
      : A_inv(A_inv_), from_id(from_id_), to_id(to_id_), N(X_.n_rows),
        K(X_.n_cols), G(std::max(arma::max(from_id_), arma::max(to_id_))),
        verbose(verbose_) {

    omp_set_num_threads(ncores);

    if (u_hat_.n_elem != (unsigned)N || from_id.n_elem != (unsigned)N ||
        to_id.n_elem != (unsigned)N) {
      Rcpp::stop("Dimensions mismatch: nrow(X) must match lengths of u_hat, "
                 "from_id, to_id.");
    }

    // Store conditional moments into M
    // Reorder M so that we can use indexing to
    // find neighbours. NOTE: I assume directed graph with self-nodes
    // thus there are G ^ 2 dyads. If some dyads are missing, e.g. there
    // are not self-nodes then those entries are always zero, and all should
    // be fine.
    M = arma::mat(std::pow(G, 2), K, arma::fill::zeros);
    M.rows((from_id - 1) * G + (to_id - 1)) = X_.each_col() % u_hat_;

    // Set of nodes
    sec_nodes = arma::regspace<arma::uvec>(0, G - 1);
  }

  /////////////////////////////////////////////////////////////////////////
  // Compute the meat of the sandwich using indexes in parallelized chunks
  arma::mat meatIndexed() const {
    if (verbose)
      std::cout
          << "Computing 'meat' of covariance sandwich (chunked parallel)\n";
    arma::mat B(K, K, arma::fill::zeros);

    const int nthreads = omp_get_max_threads();
    int nchunks = nthreads * 10; // Every thread gets about 10 jobs
    if (nchunks > static_cast<int>(N))
      nchunks = static_cast<int>(N);
    ;
    int chunk_size = std::ceil(static_cast<int>(N) / nchunks);

    Progress prog(nchunks, verbose);

#pragma omp parallel for schedule(static)
    for (int c = 0; c < nchunks; ++c) {
      // This is the "outer loop across chunks" that is parallelised.

      // Per-chunk local accumulator (ONE per chunk/thread)
      arma::mat B_local(K, K, arma::fill::zeros);

      // Determine the range [start, end) of d for this chunk
      const int start = c * chunk_size;
      const int stop = start + chunk_size;
      const int end = std::min(static_cast<int>(N), stop);

      // Inner loop: iterate through the elements of the chunk
      for (int d = start; d < end; ++d) {
        const int e = from_id(d) - 1;
        const int r = to_id(d) - 1;
        arma::rowvec m_d = M.row(e * G + r);

        if (e != r) {
          arma::uvec idx(4 * G);
          idx.subvec(0, G - 1) = e * G + sec_nodes;
          idx.subvec(G, 2 * G - 1) = sec_nodes * G + e;
          idx.subvec(2 * G, 3 * G - 1) = r * G + sec_nodes;
          idx.subvec(3 * G, 4 * G - 1) = sec_nodes * G + r;

          // Sum all rows (with duplicates)
          arma::rowvec s_d = arma::sum(M.rows(idx), 0);

          // Correct for duplicates:
          // e*G + e is counted twice -> subtract one copy
          s_d -= M.row(e * G + e);

          // r*G + r is counted twice -> subtract one copy
          s_d -= M.row(r * G + r);

          // Other common elements
          s_d -= M.row(e * G + r);
          s_d -= M.row(r * G + e);

          // Single outer product
          B_local += m_d.t() * s_d;
        } else if (e == r) {
          // Only A and B are needed when e == r
          arma::uvec idx(2 * G);

          // A: e*G + sec_nodes
          idx.subvec(0, G - 1) = e * G + sec_nodes;

          // B: sec_nodes*G + e
          idx.subvec(G, 2 * G - 1) = sec_nodes * G + e;

          // Sum all rows (with duplicates)
          arma::rowvec s_d = arma::sum(M.rows(idx), 0);

          // (e,e) corresponds to index e*G + e
          // It appears twice (once in A, once in B), but we want it once,
          // so subtract one copy.
          s_d -= m_d;

          // Outer product
          B_local += m_d.t() * s_d;
        }
      }
      // Merge chunk result into global B
#pragma omp critical
      {
        B += B_local;
      }

      if (verbose) {
        prog.increment(); // one tick per chunk
      }
    }

    return B;
  }

  /////////////////////////////////////////////////////////////////////////
  // Wrapper for computing sandwich covariance
  arma::mat get_cov() const {
    const arma::mat B = meatIndexed();
    return A_inv * B * A_inv;
  }
};

// [[Rcpp::export]]
arma::mat vcovFG_Rcpp(const arma::mat &A_inv, const arma::mat &X,
                      const arma::vec &u_hat, const arma::uvec &from_id,
                      const arma::uvec &to_id, const int ncores = 1,
                      const bool verbose = false) {

  FGCov covObjec = FGCov(A_inv, X, u_hat, from_id, to_id, verbose, ncores);

  return covObjec.get_cov();
}
