#!/usr/bin/env bash
# Docker-based R CMD check for capybara.
#
# Two modes, selected by whether a C++ standard is given as the 2nd arg:
#
#   check.sh <image>              Full CRAN-style check using the image's
#                                  default toolchain, e.g. "gcc16" for the
#                                  latest GCC, or "rocky8" / "debian10" for
#                                  other platforms.
#
#   check.sh <image> <std> [cc]   Also pins CXX to a single C++ standard
#                                  (cxx11|cxx14|cxx17|cxx20|cxx23) with
#                                  compiler "gcc" (default) or "clang", to
#                                  confirm clean compilation, e.g.
#                                  "check.sh ubuntu-release cxx17 clang".
#
# Usage: check.sh <rhub-image> [cxx-std] [gcc|clang]
set -euo pipefail

IMAGE="${1:?Usage: $0 <rhub-image> [cxx-std] [gcc|clang]}"
STD="${2:-}"
COMPILER="${3:-gcc}"

case "$STD" in
  "" ) STD_FLAG="" ;;
  cxx11) STD_FLAG="-std=c++11 -pedantic-errors -Wall -Wextra" ;;
  cxx14) STD_FLAG="-std=c++14 -pedantic-errors -Wall -Wextra" ;;
  cxx17) STD_FLAG="-std=c++17 -pedantic-errors -Wall -Wextra" ;;
  cxx20) STD_FLAG="-std=c++20 -pedantic-errors -Wall -Wextra" ;;
  cxx23) STD_FLAG="-std=c++23 -pedantic-errors -Wall -Wextra" ;;
  *) echo "Unknown C++ standard: $STD (expected cxx11|cxx14|cxx17|cxx20|cxx23)"; exit 1 ;;
esac

case "$COMPILER" in
  gcc|clang) ;;
  *) echo "Unknown compiler: $COMPILER (expected gcc|clang)"; exit 1 ;;
esac

if [ -n "${FULL_IMAGE_OVERRIDE:-}" ]; then
  FULL_IMAGE="$FULL_IMAGE_OVERRIDE"
else
  FULL_IMAGE="ghcr.io/r-hub/containers/${IMAGE}:latest"
fi

SUFFIX="${IMAGE}${STD:+-$STD-$COMPILER}"
LOG_DIR="./check-docker"
LOG="${LOG_DIR}/${SUFFIX}.log"
# R library cache: kept per-project under ./check-docker/cache, deliberately
# NOT shared, so installed package versions/binaries for this repo never
# bleed into (or get clobbered by) another package's check.
CACHE_DIR="$(pwd)/check-docker/cache/${SUFFIX}"
CHECK_DIR=$(mktemp -d)

# Image cache: shared across every repo/package that runs this script, so
# the (large) r-hub images are only ever pulled once instead of once per
# package. Override with R_HUB_DOCKER_CACHE if you want it somewhere else.
IMAGE_CACHE_DIR="${R_HUB_DOCKER_CACHE:-$HOME/.cache/r-hub-docker}"

mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$IMAGE_CACHE_DIR"
trap 'rm -rf "$CHECK_DIR"' EXIT

echo "==============================="
if [ -n "$STD" ]; then
  echo "Docker check: $IMAGE (forcing ${STD}, ${COMPILER})"
else
  echo "Docker check: $IMAGE (default toolchain)"
fi
echo "==============================="

# Filesystem-safe tarball name for an image ref, e.g.
# ghcr.io/r-hub/containers/ubuntu-release:latest -> ghcr.io_r-hub_containers_ubuntu-release_latest.tar
image_tar_path() {
  echo "${IMAGE_CACHE_DIR}/$(echo "$1" | tr '/:' '__').tar"
}

# 1. Already in Docker's local image store (from a previous run, on this
#    machine): nothing to pull/load.
# 2. Not in the local store, but a tarball for it exists in the shared
#    cache: load it (fast, no network).
# 3. Neither: pull from the registry (with fallbacks), then fall through to
#    the save step below.
if docker image inspect "$FULL_IMAGE" >/dev/null 2>&1; then
  echo "Using locally cached image $FULL_IMAGE"
else
  IMAGE_TAR="$(image_tar_path "$FULL_IMAGE")"
  if [ -f "$IMAGE_TAR" ]; then
    echo "Loading $FULL_IMAGE from shared image cache ($IMAGE_TAR)..."
    docker load -i "$IMAGE_TAR" >/dev/null
  fi
fi

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
fi

# Regardless of how we got the image (already local, loaded from the shared
# cache, or just pulled), make sure a tarball for it exists in the shared
# cache so the next check of this or any other package can `docker load` it
# instead of hitting the registry. Note this pins whatever ":latest"
# resolved to right now; delete the tar file under $IMAGE_CACHE_DIR to force
# a re-pull of a newer ":latest" later.
IMAGE_TAR="$(image_tar_path "$FULL_IMAGE")"
if [ ! -f "$IMAGE_TAR" ]; then
  echo "Saving $FULL_IMAGE to shared image cache ($IMAGE_TAR)..."
  docker save "$FULL_IMAGE" -o "$IMAGE_TAR"
fi

echo "Building package tarballs..."

# Have each Rscript call write its result to a temp file, then read the path
# back.
BUILD_LOG="${LOG_DIR}/${SUFFIX}-build.log"
: > "$BUILD_LOG"
CPP4R_TARBALL_FILE=$(mktemp)
ARMADILLO4R_TARBALL_FILE=$(mktemp)
CAPYBARA_TARBALL_FILE=$(mktemp)
trap 'rm -rf "$CHECK_DIR" "$CPP4R_TARBALL_FILE" "$ARMADILLO4R_TARBALL_FILE" "$CAPYBARA_TARBALL_FILE"' EXIT

Rscript -e 'tinydev::pkg_register(".")' >>"$BUILD_LOG" 2>&1
Rscript -e 'tinydev::pkg_document(".")' >>"$BUILD_LOG" 2>&1

Rscript -e "writeLines(tinydev::pkg_build('../cpp4r'), '${CPP4R_TARBALL_FILE}')" >>"$BUILD_LOG" 2>&1

Rscript -e "writeLines(tinydev::pkg_build('../armadillo4r'), '${ARMADILLO4R_TARBALL_FILE}')" >>"$BUILD_LOG" 2>&1
Rscript -e "writeLines(tinydev::pkg_build('.'), '${CAPYBARA_TARBALL_FILE}')" >>"$BUILD_LOG" 2>&1

CPP4R_TARBALL=$(cat "$CPP4R_TARBALL_FILE")
ARMADILLO4R_TARBALL=$(cat "$ARMADILLO4R_TARBALL_FILE")
CAPYBARA_TARBALL=$(cat "$CAPYBARA_TARBALL_FILE")

CPP4R_FILE=$(basename "$CPP4R_TARBALL")
ARMADILLO4R_FILE=$(basename "$ARMADILLO4R_TARBALL")
CAPYBARA_FILE=$(basename "$CAPYBARA_TARBALL")

cp "$CPP4R_TARBALL" "$CHECK_DIR/"
cp "$ARMADILLO4R_TARBALL" "$CHECK_DIR/"
cp "$CAPYBARA_TARBALL" "$CHECK_DIR/"

# Create a minimal R script that installs the packages' own declared
# Imports/Suggests/LinkingTo (for a full test run, not just the CRAN-check
# metadata) when run inside the container. Dependencies are read straight
# out of each tarball's DESCRIPTION via base R's read.dcf (so this doesn't
# itself depend on any package being installed yet), rather than a
# hardcoded package list that silently goes stale whenever a test starts
# depending on a new Suggested package (e.g. 'desc').
cat > "$CHECK_DIR/install_required.R" <<'R_EOF'
user_lib <- strsplit(Sys.getenv('R_LIBS_USER'), ':')[[1]][1]
.libPaths(c(user_lib, .libPaths()))
options(repos = c('https://yihui.r-universe.dev', 'https://cloud.r-project.org'))

deps_from_tarball <- function(tarfile, own_names) {
  td <- tempfile()
  dir.create(td)
  utils::untar(tarfile, exdir = td)
  pkgdir <- list.dirs(td, recursive = FALSE)[1]
  dcf <- read.dcf(file.path(pkgdir, 'DESCRIPTION'))
  fields <- intersect(c('Depends', 'Imports', 'Suggests', 'LinkingTo'), colnames(dcf))
  if (length(fields) == 0) return(character())
  raw <- unlist(strsplit(dcf[1, fields], ','))
  raw <- trimws(sub('\\(.*\\)', '', raw))
  raw <- raw[nzchar(raw) & raw != 'R']
  setdiff(raw, own_names)
}

tarballs <- c(__TARBALLS__)
own_names <- c(__OWN_NAMES__)
pkgs <- unique(unlist(lapply(tarballs, deps_from_tarball, own_names = own_names)))
pkgs <- setdiff(pkgs, rownames(installed.packages()))

if (length(pkgs) > 0) {
  message('Installing declared dependencies: ', paste(pkgs, collapse = ', '))
  install.packages(pkgs, lib = user_lib)
}

R_EOF
sed -i \
  -e "s|__TARBALLS__|'/check/${CPP4R_FILE}', '/check/${CAPYBARA_FILE}'|" \
  -e "s|__OWN_NAMES__|'cpp4r', 'capybara'|" \
  "$CHECK_DIR/install_required.R"

# When a C++ standard was requested, build the shell snippet that pins CXX to
# it inside the container. Left empty to use the image's default toolchain.
#
# Built from a fully-quoted heredoc (so nothing is expanded/escaped here) plus
# plain text substitutions, to keep the "evaluate later, inside the
# container" bits (bare $GXX/$GCC) free of backslash-escaping gymnastics.
MAKEVARS_STEP=""
if [ -n "$STD_FLAG" ]; then
  if [ "$COMPILER" = "clang" ]; then
    GXX_LINE='GXX="clang++"'
    GCC_LINE='GCC="clang"'
  else
    GXX_LINE='GXX=$(R CMD config CXX | awk '"'"'{print $1}'"'"')'
    GCC_LINE='GCC=$(R CMD config CC  | awk '"'"'{print $1}'"'"')'
  fi

  MAKEVARS_STEP=$(cat <<'HEREDOC'

      # R installations often bake the default standard into CXX itself
      # (e.g. CXX = g++ -std=gnu++20); `awk '{print $1}'` strips any such
      # flags, keeping just the bare compiler binary. The desired standard is
      # then applied exactly once via the CXX*STD variables, which R appends
      # to CXX* when invoking the compiler (baking it into CXX* too would
      # duplicate the flags on the command line).
      __GXX_LINE__
      __GCC_LINE__
      mkdir -p ~/.R
      {
        echo "CC=$GCC"
        echo "CXX=$GXX"
        echo "CXXSTD=__STD_FLAG__"
        echo "CXX11=$GXX"
        echo "CXX11STD=__STD_FLAG__"
        echo "CXX14=$GXX"
        echo "CXX14STD=__STD_FLAG__"
        echo "CXX17=$GXX"
        echo "CXX17STD=__STD_FLAG__"
        echo "CXX20=$GXX"
        echo "CXX20STD=__STD_FLAG__"
        echo "CXX23=$GXX"
        echo "CXX23STD=__STD_FLAG__"
      } > ~/.R/Makevars
HEREDOC
)
  MAKEVARS_STEP="${MAKEVARS_STEP//__GXX_LINE__/$GXX_LINE}"
  MAKEVARS_STEP="${MAKEVARS_STEP//__GCC_LINE__/$GCC_LINE}"
  MAKEVARS_STEP="${MAKEVARS_STEP//__STD_FLAG__/$STD_FLAG}"
fi

# Extra system packages to install so the requested compiler is available.
EXTRA_APT_PKGS=""
EXTRA_DNF_PKGS=""
EXTRA_ZYPPER_PKGS=""
if [ "$COMPILER" = "clang" ]; then
  # clang needs LLVM's OpenMP runtime (libomp) explicitly: unlike gcc, whose
  # -fopenmp support (libgomp) ships with the gcc package itself, clang's
  # -fopenmp links against -lomp, which isn't installed by the "clang"
  # package alone.
  EXTRA_APT_PKGS="clang libomp-dev"
  EXTRA_DNF_PKGS="clang libomp-devel"
  EXTRA_ZYPPER_PKGS="clang libomp-devel"
fi

# capybara/configure regenerates src/Makevars from Makevars.in, honoring
# this env var for CXX_STD (defaulting to CXX23 when unset/empty).
CAPYBARA_CXX_STD=""
if [ -n "$STD" ]; then
  CAPYBARA_CXX_STD=$(echo "$STD" | tr '[:lower:]' '[:upper:]')
fi

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
    export R_LIBS=/cache/R_libs
    # Debian/Ubuntu's r-base Makeconf honors DEB_BUILD_OPTIONS=noopt by
    # rebuilding CFLAGS/CXXFLAGS as '-UNDEBUG -Wall -pedantic -g -O0',
    # overriding the normal -O2. That -O0 then conflicts with the image's
    # hardened -D_FORTIFY_SOURCE=3 default (glibc requires -O1+ for it to
    # take effect), producing a '#warning' on every compile. Clearing it
    # restores normal -O2 optimized builds inside the container.
    unset DEB_BUILD_OPTIONS
    RHOME=\$(R RHOME)
    mkdir -p \"\$RHOME/etc\"
    cat >> \"\$RHOME/etc/Rprofile.site\" <<'RPROFILE_EOF'
options(repos = c('https://yihui.r-universe.dev', 'https://cloud.r-project.org'))
RPROFILE_EOF
    mkdir -p /cache/R_libs
    # Install minimal system build deps needed by R packages (libuv for 'fs')
    if command -v apt-get >/dev/null 2>&1; then
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq || true
      apt-get install -y --no-install-recommends \
        devscripts pkg-config gfortran libcurl4-openssl-dev ${EXTRA_APT_PKGS} || true
    elif command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then
      PKG_MGR=\$(command -v dnf 2>/dev/null || echo yum)
      \$PKG_MGR -y install pkgconfig gcc-gfortran libcurl-devel ${EXTRA_DNF_PKGS} || true
    elif command -v zypper >/dev/null 2>&1; then
      zypper --non-interactive install pkg-config gcc-fortran libcurl-devel ${EXTRA_ZYPPER_PKGS} || true
    fi

    # Install system deps (xml2, etc) and R package deps before writing
    # ~/.R/Makevars so packages with C++ code (diffobj, etc) compile with the
    # image's default standard instead of the one under test.
    if [ -f /check/install_required.R ]; then Rscript /check/install_required.R || true; fi

    # --as-cran's 'checking CRAN incoming feasibility' step uses the 'curl'
    # R package to verify URLs/DOIs in the docs. It isn't a dependency of
    # any of our packages; without it, URL/DOI verification errors out
    # (rather than just flagging a bad link), which escalates that check from an
    # informational NOTE to a WARNING.
    Rscript -e \"if (!requireNamespace('curl', quietly = TRUE)) install.packages('curl', lib = Sys.getenv('R_LIBS_USER'))\" || true
${MAKEVARS_STEP}
    # Remove stale locks and old cpp4r/capybara before reinstalling
    rm -rf /cache/R_libs/00LOCK-* /cache/R_libs/cpp4r /cache/R_libs/capybara
    export CAPYBARA_CXX_STD='${CAPYBARA_CXX_STD}'
    R CMD INSTALL --library=/cache/R_libs /check/${CPP4R_FILE}
    R CMD INSTALL --library=/cache/R_libs /check/${ARMADILLO4R_FILE}
    R CMD INSTALL --library=/cache/R_libs /check/${CAPYBARA_FILE}
    cd /check
    export _R_CHECK_FORCE_SUGGESTS_=false
    R CMD check --as-cran --no-manual ${CAPYBARA_FILE}
  " 2>&1 | grep -v 'readelf: Warning:' | tee "${CHECK_DIR}/docker.log" || DOCKER_RC="${PIPESTATUS[0]}"

if awk 'found { print; next } /^\*\* this is package .*capybara.* version/ { found=1; print }' "${CHECK_DIR}/docker.log" > "${CHECK_DIR}/docker.trimmed.log" && [ -s "${CHECK_DIR}/docker.trimmed.log" ]; then
  cp "${CHECK_DIR}/docker.trimmed.log" "$LOG"
else
  cp "${CHECK_DIR}/docker.log" "$LOG"
fi

if [ -d "${CHECK_DIR}/capybara.Rcheck" ]; then
  RCHECK_DEST="${LOG_DIR}/${SUFFIX}-capybara.Rcheck"
  rm -rf "$RCHECK_DEST"
  cp -r "${CHECK_DIR}/capybara.Rcheck" "$RCHECK_DEST"
  rm "$BUILD_LOG"
  echo "Rcheck directory saved to: ${RCHECK_DEST}"
fi

echo "==============================="
echo "Check complete. Log: $LOG"
echo "==============================="

exit $DOCKER_RC
