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
trap 'rm -rf "$CHECK_DIR"; rm -f "${CAPYBARA_TARBALL:-}" "${ARMADILLO4R_TARBALL:-}" "${CPP4R_TARBALL:-}"' EXIT

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
Rscript -e 'cpp4r::register(".")' 2>/dev/null
Rscript -e 'tinydev::pkg_document(".")' 2>/dev/null

CPP4R_TARBALL=$(Rscript -e 'out <- tinydev::pkg_build("../cpp4r"); cat(out)' 2>/dev/null | grep -o '[^ ]*\.tar\.gz$' | tail -1)
ARMADILLO4R_TARBALL=$(Rscript -e 'out <- tinydev::pkg_build("../armadillo4r"); cat(out)' 2>/dev/null | grep -o '[^ ]*\.tar\.gz$' | tail -1)
CAPYBARA_TARBALL=$(Rscript -e 'out <- tinydev::pkg_build("."); cat(out)' 2>/dev/null | grep -o '[^ ]*\.tar\.gz$' | tail -1)

CPP4R_FILE=$(basename "$CPP4R_TARBALL")
ARMADILLO4R_FILE=$(basename "$ARMADILLO4R_TARBALL")
CAPYBARA_FILE=$(basename "$CAPYBARA_TARBALL")

cp "$CPP4R_TARBALL" "$CHECK_DIR/"
cp "$ARMADILLO4R_TARBALL" "$CHECK_DIR/"
cp "$CAPYBARA_TARBALL" "$CHECK_DIR/"

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

if (!requireNamespace('tinytest', quietly = TRUE)) {
  install.packages('tinytest', lib = user_lib)
}

if (!requireNamespace('curl', quietly = TRUE)) {
  install.packages('curl', lib = user_lib)
}

if (!requireNamespace('Formula', quietly = TRUE)) {
  install.packages('Formula', lib = user_lib)
}

if (!requireNamespace('generics', quietly = TRUE)) {
  install.packages('generics', lib = user_lib)
}

if (!requireNamespace('ggplot2', quietly = TRUE)) {
  install.packages('ggplot2', lib = user_lib)
}

if (!requireNamespace('broom', quietly = TRUE)) {
  install.packages('broom', lib = user_lib)
}

if (!requireNamespace('knitr', quietly = TRUE)) {
  install.packages('knitr', lib = user_lib)
}

if (!requireNamespace('rmarkdown', quietly = TRUE)) {
  install.packages('rmarkdown', lib = user_lib)
}

if (!requireNamespace('units', quietly = TRUE)) {
  install.packages('units', lib = user_lib)
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
      cat /check/capybara.Rcheck/00install.out || true
      echo '=== 00check.log ==='
      cat /check/capybara.Rcheck/00check.log || true
      chmod -R a+rwX /check
    }
    trap show_logs_and_fix_perms EXIT
    export R_LIBS_USER=/cache/R_libs
    export CAPYBARA_FULL_TESTING="yes"
    mkdir -p /cache/R_libs
    # Install minimal system build deps needed by R packages (libuv for 'fs')
    if command -v apt-get >/dev/null 2>&1; then
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq || true
      apt-get install -y --no-install-recommends libuv1-dev libxml2-dev pkg-config libudunits2-dev wget libcurl4-openssl-dev || true
      # Ubuntu 22.04 ships pandoc 2.9 which lacks built-in citeproc; install a modern version from GitHub
      wget -q -O /tmp/pandoc.deb https://github.com/jgm/pandoc/releases/download/3.1.1/pandoc-3.1.1-1-amd64.deb && dpkg -i /tmp/pandoc.deb || true
    elif command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then
      PKG_MGR=dnf
      if command -v yum >/dev/null 2>&1; then PKG_MGR=yum; fi
      \$PKG_MGR -y install libuv-devel libxml2-devel pkgconfig || true
    elif command -v zypper >/dev/null 2>&1; then
      zypper --non-interactive install libuv libxml2-devel pkg-config || true
    fi

    # Run minimal missing-package installer if present
    if [ -f /check/install_required.R ]; then Rscript /check/install_required.R || true; fi
    # Remove stale locks and old capybara before reinstalling
    rm -rf /cache/R_libs/00LOCK-* /cache/R_libs/capybara
    R CMD INSTALL --library=/cache/R_libs /check/${CPP4R_FILE}
    R CMD INSTALL --library=/cache/R_libs /check/${ARMADILLO4R_FILE}
    R CMD INSTALL --library=/cache/R_libs /check/${CAPYBARA_FILE}
    cd /check
    export _R_CHECK_FORCE_SUGGESTS_=false
    export _R_CHECK_CRAN_INCOMING_REMOTE_=false
    R CMD check --as-cran --no-manual ${CAPYBARA_FILE}
  " 2>&1 | grep -v 'readelf: Warning:' | tee "${CHECK_DIR}/docker.log" || DOCKER_RC="${PIPESTATUS[0]}"

cp "${CHECK_DIR}/docker.log" "$LOG"

if [ -d "${CHECK_DIR}/capybara.Rcheck" ]; then
  RCHECK_DEST="${LOG_DIR}/${IMAGE}-capybara.Rcheck"
  rm -rf "$RCHECK_DEST"
  cp -r "${CHECK_DIR}/capybara.Rcheck" "$RCHECK_DEST"
  echo "Rcheck directory saved to: ${RCHECK_DEST}"
fi

echo "==============================="
echo "Check complete. Log: $LOG"
echo "==============================="

exit $DOCKER_RC
