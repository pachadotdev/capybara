library(capybara)

# Refitting models

fepoisson(
    mpg ~ wt | cyl,
    mtcars,
    control = fit_control(vcov_type = "hetero")
)

fepoisson(
    mpg ~ wt | cyl | am,
    mtcars,
    control = fit_control(vcov_type = "m-estimator")
)

# Reusing models

# Store required components
mod <- fepoisson(
    mpg ~ wt | cyl,
    mtcars,
    control = fit_control(keep_tx = TRUE, return_hessian = TRUE)
)

# Heteroskedastic-robust HC0 sandwich (no cluster variable needed)
v1 <- sandwich_vcov(mod, type = "hetero")

# One-way M-estimator sandwich (cluster variable required)
v2 <- sandwich_vcov(mod, cluster1 = mtcars$am, type = "m-estimator")

round(sqrt(v1), 4)

round(sqrt(v2), 4)
