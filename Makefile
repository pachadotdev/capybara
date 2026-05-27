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
	@Rscript -e 'devtools::clean_dll(".");'

install:
	@Rscript -e 'devtools::install(".", upgrade = FALSE)'

clang_format=`which clang-format-21`

format: $(shell find . -not -path './check-docker/*' -not -path './src/vendor/*' -name '*.h') $(shell find . -not -path './check-docker/*' -not -path './src/vendor/*' -name '*.hpp') $(shell find . -not -path './check-docker/*' -not -path './src/vendor/*' -name '*.cpp')
	@${clang_format} -i $?

cran:
	clear
	@cp DESCRIPTION DESCRIPTION.bak
	@awk '/^Remotes:/ {skip=1} /^Roxygen:/ {skip=1} skip && NF==0 {skip=0; next} !skip' DESCRIPTION.bak > DESCRIPTION
	@Rscript -e 'devtools::build()'
	@mv DESCRIPTION.bak DESCRIPTION

nonascii:
	@find R/ src/ -type f -exec grep -P -H -n "[^\x00-\x7F]" {} + || true
