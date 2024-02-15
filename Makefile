test:
	@Rscript -e 'devtools::clean_dll(); cpp11::cpp_register(); devtools::test(".")'

clean:
	@Rscript -e 'devtools::clean_dll()'

document:
  @Rscript -e 'devtools::document()'

site:
	@Rscript -e 'pkgdown::build_site()'

install:
	@Rscript -e 'devtools::install()'

clang_format=`which clang-format-14`

format: $(shell find . -name '*.h') $(shell find . -name '*.hpp') $(shell find . -name '*.cpp')
	@${clang_format} -i $?
