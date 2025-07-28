clean:
	clear
	@Rscript -e 'devtools::clean_dll()'

test:
	clear
	@echo "Testing R code"
	@Rscript -e 'devtools::document()'
	@Rscript -e 'devtools::load_all(); devtools::test()'

test-long:
	clear
	@echo "Testing R code with models from AGTPA"
	Rscript dev/article/benchmark-ppml.r

bench-ppml:
	clear
	@echo "Benchmark - PPML"
	Rscript dev/article/benchmark-ppml.r; \

check:
	clear
	@echo "Local"
	@Rscript -e 'devtools::check()'
	@echo "RHub"
	@Rscript -e 'devtools::check_rhub()'
	@echo "Win Builder"
	@Rscript -e 'devtools::check_win_release()'
	@Rscript -e 'devtools::check_win_devel()'

site:
	clear
	@Rscript -e 'pkgdown::build_site()'

install:
	clear
	@export CAPYBARA_ADVANCED_BUILD="yes"
	@Rscript -e 'devtools::install(upgrade = "never")'

clang_format=`which clang-format`

format: $(shell find . -name '*.h') $(shell find . -name '*.hpp') $(shell find . -name '*.cpp')
	@${clang_format} -i $?

cran:
	clear
	@cp DESCRIPTION DESCRIPTION.bak
	@awk '/^Remotes:/ {skip=1} /^Roxygen:/ {skip=1} skip && NF==0 {skip=0; next} !skip' DESCRIPTION.bak > DESCRIPTION
	@echo "Running devtools::check()"
	@Rscript -e 'devtools::check()'
	@echo "Submitting to CRAN"
	@Rscript -e 'devtools::submit_cran()'
	@mv DESCRIPTION.bak DESCRIPTION
