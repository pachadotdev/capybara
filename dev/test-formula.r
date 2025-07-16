f <- mpg ~ wt | cyl + am

f <- Formula::as.Formula(f)

f[[1]] # ~
f[[2]] # mpg
f[[3]] # wt | cyl + am

rhs <- strsplit(as.character(f[[3]]), "\\|")

rhs[[1]] # ""
rhs[[2]] # wt
rhs[[3]] # cyl + am

fe <- strsplit(rhs[[3]], " \\+ ")

for (i in seq_along(fe)) {
  tmp <- fe[[i]]
  tmp <- trimws(tmp) # Remove leading/trailing whitespace
  fe[[i]] <- tmp
}
