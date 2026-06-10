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
	@chmod +x ./check-docker/check.sh
	@for pair in $(CRAN_PAIRS); do \
		cran=$${pair%%:*}; rhub=$${pair##*:}; \
		echo "=== checking $$cran (r-hub: $$rhub) ==="; \
		./check-docker/check.sh $$rhub; \
	done

check-cran-extra:
	@chmod +x ./check-docker/check.sh
	@for rhub in $(CRAN_EXTRA); do \
		echo "=== checking $$rhub ==="; \
		./check-docker/check.sh $$rhub; \
	done

# Individual check target, e.g. `make check-clang22`
check-%:
	@chmod +x ./check-docker/check.sh
	@./check-docker/check.sh $*

clean:
	@Rscript --vanilla -e 'devtools::clean_dll(".");'

install:
	@Rscript --vanilla -e 'devtools::install(".", upgrade = FALSE)'

build:
	clear
	@Rscript -e 'tinydev::pkg_build(".")'

nonascii:
	@find R/ src/ -type f -exec grep -P -H -n "[^\x00-\x7F]" {} + || true

clang_format=`which clang-format-21`

format: $(shell find . -not -path './check-docker/*' -name '*.h') $(shell find . -not -path './check-docker/*' -name '*.hpp') $(shell find . -not -path './check-docker/*' -name '*.cpp')
	@${clang_format} -i $?
