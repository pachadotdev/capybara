# Debug fixed effects issue
devtools::load_all()

# Base R model
base_mod <- glm(mpg ~ wt + as.factor(cyl), mtcars, family = quasipoisson(link = "log"))

# Our model
our_mod <- fepoisson(mpg ~ wt | cyl | am, mtcars)

cat("=== COEFFICIENT COMPARISON ===\n")
cat('Base R wt coefficient:', coef(base_mod)['wt'], '\n')
cat('Our wt coefficient:', coef(our_mod)['wt'], '\n')
cat('Difference:', coef(base_mod)['wt'] - coef(our_mod)['wt'], '\n\n')

cat("=== LINEAR PREDICTOR COMPARISON ===\n")
base_eta <- predict(base_mod)
our_eta <- our_mod[['linear.predictors']]

cat('Base R eta (first 10):\n')
print(head(base_eta, 10))
cat('Our eta (first 10):\n')
print(head(our_eta, 10))
cat('Difference (first 10):\n')
print(head(base_eta - our_eta, 10))

# Check if the differences are systematic by group
cat('\n=== SYSTEMATIC DIFFERENCES BY GROUP ===\n')
diff_eta <- base_eta - our_eta
base_cyl <- as.factor(mtcars[['cyl']])
cat('Mean difference by cylinder:\n')
cat('cyl=4:', mean(diff_eta[base_cyl == '4']), '\n')
cat('cyl=6:', mean(diff_eta[base_cyl == '6']), '\n')
cat('cyl=8:', mean(diff_eta[base_cyl == '8']), '\n')

cat('\n=== MANUAL FIXED EFFECTS CALCULATION ===\n')
# Base R method
base_wt <- mtcars[['wt']]
base_beta_wt <- coef(base_mod)['wt']
base_pie <- base_eta - base_wt * base_beta_wt

cat('Base R fixed effects:\n')
cat('cyl=4:', mean(base_pie[base_cyl == '4']), '\n')
cat('cyl=6:', mean(base_pie[base_cyl == '6']), '\n')
cat('cyl=8:', mean(base_pie[base_cyl == '8']), '\n')

# Our method
our_wt <- our_mod[['data']][['wt']]
our_beta_wt <- coef(our_mod)['wt']
our_pie <- our_eta - our_wt * our_beta_wt
our_cyl <- our_mod[['data']][['cyl']]

cat('\nOur fixed effects:\n')
cat('cyl=4:', mean(our_pie[our_cyl == '4']), '\n')
cat('cyl=6:', mean(our_pie[our_cyl == '6']), '\n')
cat('cyl=8:', mean(our_pie[our_cyl == '8']), '\n')

cat('\nFixed effects from capybara function:\n')
fe <- fixed_effects(our_mod)
print(fe$cyl)

cat('\n=== ANALYZING THE OFFSET ===\n')
# If we add the systematic differences to our linear predictors, do we get base R values?
our_eta_corrected <- our_eta + mean(diff_eta[base_cyl == '4']) * (base_cyl == '4') +
                              mean(diff_eta[base_cyl == '6']) * (base_cyl == '6') +
                              mean(diff_eta[base_cyl == '8']) * (base_cyl == '8')

cat('Corrected eta matches base R eta:\n')
cat('Max difference after correction:', max(abs(base_eta - our_eta_corrected)), '\n')

# What would the corrected fixed effects be?
our_pie_corrected <- our_eta_corrected - our_wt * our_beta_wt
cat('\nCorrected fixed effects:\n')
cat('cyl=4:', mean(our_pie_corrected[base_cyl == '4']), '\n')
cat('cyl=6:', mean(our_pie_corrected[base_cyl == '6']), '\n')
cat('cyl=8:', mean(our_pie_corrected[base_cyl == '8']), '\n')
