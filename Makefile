.PHONY: check-cran check-cran-extra check-% clean install docs

# CRAN-like containers (pair: CRAN name : r-hub image)
CRAN_PAIRS := \
	r-devel-linux-x86_64-debian-clang:ubuntu-clang \
 	r-devel-linux-x86_64-debian-gcc:ubuntu-gcc15 \
 	r-patched-linux-x86_64:ubuntu-next \
 	r-release-linux-x86_64:ubuntu-release

# Extra CRAN check images
CRAN_EXTRA := atlas clang-asan clang-ubsan clang21 clang22 donttest \
	gcc16 gcc-asan lto mkl nold nosuggests rchk valgrind

check-cran:
	@chmod +x ./scripts/check.sh
	@for pair in $(CRAN_PAIRS); do \
		cran=$${pair%%:*}; rhub=$${pair##*:}; \
		echo "=== checking $$cran (r-hub: $$rhub) ==="; \
		./scripts/check.sh $$rhub; \
	done

check-cran-extra:
	@chmod +x ./scripts/check.sh
	@for rhub in $(CRAN_EXTRA); do \
		echo "=== checking $$rhub ==="; \
		./scripts/check.sh $$rhub; \
	done

# Individual check target, e.g. `make check-clang22`
check-%:
	@chmod +x ./scripts/check.sh
	@./scripts/check.sh $*

clean:
	@Rscript -e 'tinydev::pkg_clean(".");'

install:
	@Rscript -e 'tinydev::pkg_install(".")'

build:
	clear
	@Rscript -e 'tinydev::pkg_build(".")'

nonascii:
	@find R/ src/ -type f -exec grep -P -H -n "[^\x00-\x7F]" {} + || true

clang_format=`which clang-format-21`

format: $(shell find . -not -path './check-docker/*' -name '*.h') $(shell find . -not -path './check-docker/*' -name '*.hpp') $(shell find . -not -path './check-docker/*' -name '*.cpp')
	@${clang_format} -i $?
	@Rscript -e 'styler::style_pkg(exclude_dirs = "check-docker", exclude_files = "R/cpp4r\\.R")'
	@Rscript -e 'styler::style_dir("inst/tinytest")'
