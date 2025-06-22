clean:
	@Rscript -e 'devtools::clean_dll()'

test:
	@echo "Testing R code"
	@Rscript -e 'devtools::document()'
	@Rscript -e 'devtools::load_all(); devtools::test()'

test2:
	@echo "Testing R code with extended tests"
	@export CAPYBARA_EXTENDED_TESTS="yes"; \
  Rscript -e 'devtools::document()'; \
  Rscript -e 'devtools::load_all(); devtools::test()'; \
  unset CAPYBARA_EXTENDED_TESTS

check:
	@echo "Local"
	@Rscript -e 'devtools::check()'
	@echo "RHub"
	@Rscript -e 'devtools::check_rhub()'
	@echo "Win Builder"
	@Rscript -e 'devtools::check_win_release()'
	@Rscript -e 'devtools::check_win_devel()'

site:
	@Rscript -e 'pkgdown::build_site()'

install:
	@Rscript -e 'devtools::install()'

clang_format=`which clang-format-14`

format: $(shell find . -name '*.h') $(shell find . -name '*.hpp') $(shell find . -name '*.cpp')
	@${clang_format} -i $?

cran:
	@cp DESCRIPTION DESCRIPTION.bak
	@awk '/^Remotes:/ {skip=1} /^Roxygen:/ {skip=1} skip && NF==0 {skip=0; next} !skip' DESCRIPTION.bak > DESCRIPTION
	@echo "Running devtools::check()"
	@Rscript -e 'devtools::check()'
	@echo "Submitting to CRAN"
	@Rscript -e 'devtools::submit_cran()'
	@mv DESCRIPTION.bak DESCRIPTION
