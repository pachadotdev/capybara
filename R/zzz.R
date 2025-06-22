.onAttach <- function(libname, pkgname) {
  cores <- as.integer(parallel::detectCores() / 2)
  if (is.na(cores) || cores < 1) cores <- 1

  # Detect BLAS and set to same thread count
  blas_info <- sessionInfo()$BLAS
  blas_type <- "unknown"

  if (grepl("openblas", blas_info, ignore.case = TRUE)) {
    blas_type <- "OpenBLAS"
  } else if (grepl("mkl", blas_info, ignore.case = TRUE)) {
    blas_type <- "Intel MKL"
  } else if (grepl("blis", blas_info, ignore.case = TRUE)) {
    blas_type <- "BLIS"
  } else if (grepl("accelerate|veclib", blas_info, ignore.case = TRUE)) {
    blas_type <- "Accelerate"
  }

  if (interactive()) {
    # Package info message
    pkg_info <- sprintf(
      "capybara %s is using OpenMP and %s. Unless you installed capybara with\nCAPYBARA_NCORES=<number>, it will use %s cores for parallel processing.",
      utils::packageVersion("capybara"), blas_type, cores
    )

    # Display the message
    packageStartupMessage(pkg_info)
  }
}