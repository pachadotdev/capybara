library(dplyr)
library(tidyr)
library(haven)

cudata <- read_dta("dev/1-s2.0-S0014292116300630-mmc1/EER-D-15-00582R3_replication_files/EER-D-15-00582R3_data/EER-D-15-00582R3_data_dataaxj1.dta")

cudata <- cudata %>%
  # Subset relevant variables
  select(
    exp1to2, custrict11, ldist, comlang, border, regional,
    comcol, curcol, colony, comctry, cuwoemu, emu, cuc,
    cty1, cty2, year, pairid
  ) %>%
  # Generate identifiers required for structural gravity
  mutate(
    pairid = factor(pairid),
    exp.time = interaction(cty1, year),
    imp.time = interaction(cty2, year)
  ) %>%
  # Generate dummies for disaggregated currency unions
  mutate(
    cuau = as.numeric(cuc == "au"),
    cube = as.numeric(cuc == "be"),
    cuca = as.numeric(cuc == "ca"),
    cucf = as.numeric(cuc == "cf"),
    cucp = as.numeric(cuc == "cp"),
    cudk = as.numeric(cuc == "dk"),
    cuea = as.numeric(cuc == "ea"),
    cuec = as.numeric(cuc == "ec"),
    cuem = as.numeric(cuc == "em"),
    cufr = as.numeric(cuc == "fr"),
    cugb = as.numeric(cuc == "gb"),
    cuin = as.numeric(cuc == "in"),
    cuma = as.numeric(cuc == "ma"),
    cuml = as.numeric(cuc == "ml"),
    cunc = as.numeric(cuc == "nc"),
    cunz = as.numeric(cuc == "nz"),
    cupk = as.numeric(cuc == "pk"),
    cupt = as.numeric(cuc == "pt"),
    cusa = as.numeric(cuc == "sa"),
    cusp = as.numeric(cuc == "sp"),
    cuua = as.numeric(cuc == "ua"),
    cuus = as.numeric(cuc == "us"),
    cuwa = as.numeric(cuc == "wa"),
    cuwoo = custrict11
  ) %>%
  mutate(
    cuwoo = if_else(cuc %in% c("em", "au", "cf", "ec", "fr", "gb", "in", "us"), 0, cuwoo)
  ) %>%
  # Set missing trade flows to zero
  replace_na(list(exp1to2 = 0)) %>%
  mutate(
    # Re-scale trade flows
    exp1to2 = exp1to2 / 1000,
    # Construct binary and lagged dependent variable for the extensive margin
    trade = as.numeric(exp1to2 > 0)
  ) %>%
  group_by(pairid) %>%
  mutate(ltrade = lag(trade)) %>%
  select(year, cty1, cty2, exp.time, imp.time, pairid, trade, ltrade, cuc, everything())

# now make it lighter with proper data types

colnames(cudata)

cudata %>%
  ungroup() %>%
  select(comlang:cuwoo) %>%
  pivot_longer(everything()) %>%
  distinct() %>%
  mutate(foo = 1L) %>%
  pivot_wider(names_from = value, values_from = foo)

# => convert comlang:cuwoo to INT

cudata <- cudata %>%
  ungroup() %>%
  select(year:cuc) %>%
  bind_cols(
    cudata %>%
      ungroup() %>%
      select(comlang:cuwoo) %>%
      mutate_if(is.numeric, as.integer)
  ) %>%
  group_by(pairid)

unique(cudata$cuc)

cudata <- cudata %>%
  mutate(cuc = if_else(cuc == "", NA_character_, cuc)) %>%
  mutate(cuc = as_factor(cuc))

unique(cudata$trade)

cudata <- cudata %>%
  mutate(
    year = as.integer(year),
    cty1 = as_factor(cty1),
    cty2 = as_factor(cty2),
    trade = as.integer(trade),
    ltrade = as.integer(ltrade)
  )

saveRDS(cudata, "dev/cudata.rds", compress = "xz")
