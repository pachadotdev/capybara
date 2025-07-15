devtools::load_all()

data(mtcars)
formula <- mpg ~ wt | cyl | am

parsed <- .parse_formula(formula, colnames(mtcars), "poisson()")
cat("Parsed formula structure:\n")
str(parsed)