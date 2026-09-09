dMOD <- readxl::read_excel("~/Downloads/Replication_JCF.xlsx")

library(capybara)

mod <- fepoisson(nij_elite ~ z_1_1 | xi + yj | xi + yj, data = dMOD,
    control = fit_control(keep_tx = TRUE, return_hessian = TRUE), vcov = "dyadic")

vcov(mod)

mod <- fepoisson(nij_elite ~ z_1_1 | xi + yj | xi + yj, data = dMOD,
    control = fit_control(keep_tx = TRUE, return_hessian = TRUE), vcov = "cluster")

vcov(mod)

# Separation found in 9025 observation(s)

sandwich_vcov(mod, cluster1 = dMOD$xi, cluster2 = dMOD$yj, type = "dyadic")

# Error: Centered design matrix (tx) not found. Re-fit the model with keep_tx = TRUE in control parameters.
