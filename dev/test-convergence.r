load_all()
# Create test data
set.seed(123)
n <- 1000
x1 <- rnorm(n)
x2 <- x1  # Perfect collinearity
f <- sample(letters[1:10], n, replace = TRUE)
lambda <- exp(0.02553 * x1 + 0.02553 * x2)  # True coefficient
y <- rpois(n, lambda)
d <- data.frame(y = y, x1 = x1, x2 = x2, f = f)

# Test with default tolerance  
result1 <- feglm(y ~ x1 + x2 | f, data = d, family = poisson())
cat('Default dev_tol result:', result1$coefficients, '\n')

# Test with tighter tolerance
result2 <- feglm(y ~ x1 + x2 | f, data = d, family = poisson(), 
                control = list(dev_tol = 1e-12))
cat('Tighter dev_tol result:', result2$coefficients, '\n')

# Also try with more iterations
result3 <- feglm(y ~ x1 + x2 | f, data = d, family = poisson(), 
                control = list(dev_tol = 1e-12, iter_max = 100L))
cat('Tighter dev_tol + more iters:', result3$coefficients, '\n')
