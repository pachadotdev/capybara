foo <- readRDS("dev/foo.rds")

source("R/center_variables.R")

x <- center_variables_(foo$MX, foo$w, foo$k.list, foo$center.tol, 100L)

cpp11::cpp_source("dev/01_center_variables.cpp")

y <- center_variables2_(foo$MX, foo$w, foo$k.list, foo$center.tol, 100L)

all.equal(x, y)
