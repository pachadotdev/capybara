test <- F

# cpp11::cpp_vendor() # run only when updating C++ headers

devtools::clean_dll()
cpp11::cpp_register()
devtools::document()

if (test) {
  devtools::load_all()
} else {
  devtools::install()
  pkgdown::build_site()
}
