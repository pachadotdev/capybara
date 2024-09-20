devtools::load_all()

# install.packages("bife")
data(psid, package = "bife")

lmod = feglm(
  LFP ~ KID1 + KID2 + KID3 + log(INCH) | ID + TIME,
  data   = psid,
  family = binomial()
  )

summary(lmod)

bias_corr(lmod)
