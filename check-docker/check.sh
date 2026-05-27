#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:?Usage: $0 <rhub-image>}"

if [ -n "${FULL_IMAGE_OVERRIDE:-}" ]; then
  FULL_IMAGE="$FULL_IMAGE_OVERRIDE"
else
  FULL_IMAGE="ghcr.io/r-hub/containers/${IMAGE}:latest"
fi
LOG_DIR="./check-docker"
LOG="${LOG_DIR}/${IMAGE}.log"
CHECK_DIR=$(mktemp -d)

CACHE_DIR="$(pwd)/check-docker/cache/${IMAGE}"

mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_DIR"
trap 'rm -rf "$CHECK_DIR"' EXIT

echo "==============================="
echo "Docker check: $IMAGE"
echo "==============================="

if ! docker image inspect "$FULL_IMAGE" >/dev/null 2>&1; then
  echo "Pulling $FULL_IMAGE..."
  if ! docker pull "$FULL_IMAGE" >/dev/null 2>&1; then
    echo "Initial pull failed for $FULL_IMAGE"
    # If the image is the r-hub GHCR image and pull was denied, try common fallbacks
    FALLBACK=""
    if [[ "${FULL_IMAGE}" == ghcr.io/r-hub/containers/rocky8:* || "${IMAGE}" == "rocky8" ]]; then
      FALLBACK="docker.io/rockylinux/rockylinux:8"
    elif [[ "${IMAGE}" == "debian10" ]]; then
      FALLBACK="docker.io/library/debian:10-slim"
    else
      # Try a docker.io mirror of the same path if present
      FALLBACK="docker.io/${IMAGE}:latest"
    fi

    if [ -n "${FALLBACK}" ]; then
      echo "Attempting fallback image: ${FALLBACK}"
      if docker pull "${FALLBACK}" >/dev/null 2>&1; then
        echo "Using fallback image ${FALLBACK}"
        FULL_IMAGE="${FALLBACK}"
      else
        echo "Fallback pull failed for ${FALLBACK}; aborting."
        exit 1
      fi
    else
      echo "No fallback available; aborting."
      exit 1
    fi
  fi
else
  echo "Using cached image $FULL_IMAGE"
fi

echo "Building package tarballs..."
Rscript -e 'cpp4r::register(".")'
Rscript -e 'devtools::document(".")'

REDATAM_TARBALL=$(Rscript -e 'cat(devtools::build(".", quiet = TRUE))')

REDATAM_FILE=$(basename "$REDATAM_TARBALL")

cp "$REDATAM_TARBALL" "$CHECK_DIR/"

# Create a minimal R script that installs an explicit list of packages when run inside the container
cat > "$CHECK_DIR/install_required.R" <<'R_EOF'
user_lib <- strsplit(Sys.getenv('R_LIBS_USER'), ':')[[1]][1]
.libPaths(c(user_lib, .libPaths()))
repos_snapshot_env <- Sys.getenv('RSPM_SNAPSHOT', '')
if (nzchar(repos_snapshot_env)) {
  if (grepl('^https?://', repos_snapshot_env)) {
    options(repos = c(CRAN = repos_snapshot_env))
  } else {
    options(repos = c(CRAN = paste0('https://packagemanager.rstudio.com/cran/', repos_snapshot_env)))
  }
} else {
  options(repos = c(CRAN = 'https://cloud.r-project.org'))
}

if (!requireNamespace('remotes', quietly = TRUE)) {
  install.packages('remotes', lib = user_lib)
}

has_cpp23 <- tryCatch({
  cxx23 <- tryCatch(
    system("R CMD config CXX23", intern = TRUE, ignore.stderr = TRUE),
    error = function(e) system("R CMD config CXX", intern = TRUE, ignore.stderr = TRUE)
  )
  system(paste(cxx23, "-std=gnu++23 -x c++ /dev/null -fsyntax-only"),
         ignore.stdout = TRUE, ignore.stderr = TRUE) == 0
}, error = function(e) FALSE)

if (has_cpp23) {
  remotes::install_github("pachadotdev/testthat", lib = user_lib, upgrade = 'never')
} else if (!requireNamespace('testthat', quietly = TRUE)) {
  install.packages('testthat', lib = user_lib)
}

if (!requireNamespace('xml2', quietly = TRUE)) {
  install.packages('xml2', lib = user_lib)
}

if (!requireNamespace('knitr', quietly = TRUE)) {
  install.packages('knitr', lib = user_lib)
}

if (!requireNamespace('rmarkdown', quietly = TRUE)) {
  install.packages('rmarkdown', lib = user_lib)
}

# Install Depends/Imports/LinkingTo for each tarball (no Suggests)
pkgs <- list.files('/check', pattern = '\\.(tar\\.gz|tar|tgz)$', full.names = TRUE)
for (p in pkgs) {
  message('remotes::install_local(', p, ')')
  tryCatch(
    remotes::install_local(p, dependencies = c('Depends','Imports','LinkingTo'), upgrade = 'never', lib = user_lib),
    error = function(e) message('install_local failed: ', conditionMessage(e))
  )
}
R_EOF
# Create a helper script to configure yum and install system build deps inside the container
# (minimal mode) no pre-install scripts; install tarballs directly inside container

clear

DOCKER_RC=0
docker run --rm \
  -v "${CHECK_DIR}:/check" \
  -v "${CACHE_DIR}:/cache" \
  "$FULL_IMAGE" \
  bash -c "
    set -euo pipefail
    show_logs_and_fix_perms() {
      echo '=== 00install.out ==='
      cat /check/redatam.Rcheck/00install.out || true
      echo '=== 00check.log ==='
      cat /check/redatam.Rcheck/00check.log || true
      chmod -R a+rwX /check
    }
    trap show_logs_and_fix_perms EXIT
    export R_LIBS_USER=/cache/R_libs
    mkdir -p /cache/R_libs
    # Install minimal system build deps needed by R packages (libuv for 'fs')
    if command -v apt-get >/dev/null 2>&1; then
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq || true
      apt-get install -y --no-install-recommends libuv1-dev libxml2-dev pkg-config pandoc || true
    elif command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then
      PKG_MGR=dnf
      if command -v yum >/dev/null 2>&1; then PKG_MGR=yum; fi
      \$PKG_MGR -y install libuv-devel libxml2-devel pkgconfig || true
    elif command -v zypper >/dev/null 2>&1; then
      zypper --non-interactive install libuv libxml2-devel pkg-config || true
    fi

    # Run minimal missing-package installer if present
    if [ -f /check/install_required.R ]; then Rscript /check/install_required.R || true; fi
    # Remove stale locks and old redatam before reinstalling
    rm -rf /cache/R_libs/00LOCK-* /cache/R_libs/redatam
    R CMD INSTALL --library=/cache/R_libs /check/${REDATAM_FILE}
    # Fall back to gnu++2b if the compiler does not support gnu++23
    CXX=\$(R CMD config CXX23 2>/dev/null || R CMD config CXX)
    if ! \$CXX -std=gnu++23 -x c++ /dev/null -fsyntax-only 2>/dev/null; then
      mkdir -p ~/.R
      echo 'CXX23STD = -std=gnu++2b' >> ~/.R/Makevars
    fi
    cd /check
    export _R_CHECK_FORCE_SUGGESTS_=false
    R CMD check --as-cran --no-manual ${REDATAM_FILE}
  " 2>&1 | grep -v 'readelf: Warning:' | tee "${CHECK_DIR}/docker.log" || DOCKER_RC="${PIPESTATUS[0]}"

cp "${CHECK_DIR}/docker.log" "$LOG"

if [ -d "${CHECK_DIR}/redatam.Rcheck" ]; then
  RCHECK_DEST="${LOG_DIR}/${IMAGE}-redatam.Rcheck"
  rm -rf "$RCHECK_DEST"
  cp -r "${CHECK_DIR}/redatam.Rcheck" "$RCHECK_DEST"
  echo "Rcheck directory saved to: ${RCHECK_DEST}"
fi

echo "==============================="
echo "Check complete. Log: $LOG"
echo "==============================="

exit $DOCKER_RC
