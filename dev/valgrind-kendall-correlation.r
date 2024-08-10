# this is also tested in /tests

set.seed(123)
x <- rnorm(100)
y <- rpois(100, 2)

for (i in 1:100) {
  kendall_cor(x, y)
}
