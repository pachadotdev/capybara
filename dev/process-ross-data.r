library(haven)
library(dplyr)
library(countrycode)

ross2004 <- read_dta("dev/data4web.dta")

# FROM ROSS 2004:
#
# The exact specification of the gravity model used below is:
#
# ln(Xijt) = β0 + β1lnDij + β2ln(YiYj)t + β3ln(YiYj/PopiPopj)t + β4Langij + β5Contij  
# + β6Landlij + β7Islandij +β8ln(AreaiAreaj) + β9ComColij  + β10CurColijt   
# + β11Colonyij  + β12ComNatij + β13CUijt + β14FTAijt, + ΣtφtTt
# + γ1Bothinijt + γ2Oneinijt + γ3GSPijt  + εijt
#
# where i and j denotes trading partners, t denotes time, and the variables are defined as: 
#
# •  Xijt denotes the average value of real bilateral trade between i and j at time t, 
# •  Y is real GDP,
# •  Pop is population,
# •  D is the distance between i and j,
# •  Lang is a binary “dummy” variable which is unity if i and j have a common language and 
# zero otherwise,
# •  Cont is a binary variable which is unity if i and j share a land border,
# Landl is the number of landlocked countries in the country-pair (0, 1, or 2). 
# •  Island is the number of island nations in the pair (0, 1, or 2),
# •  Area is the area of the country (in square kilometers),
# •  ComCol is a binary variable which is unity if i and j were ever colonies after 1945 with the 
# same colonizer,
# •  CurCol is a binary variable which is unity if i and j are colonies at time t, 
# •  Colony is a binary variable which is unity if i ever colonized j or vice versa, 
# •  ComNat is a binary variable which is unity if i and j remained part of the same nation during 
# the sample (e.g., France and Guadeloupe), 
# •  CU is a binary variable which is unity if i and j use the same currency at time t, 
# •  FTA is a binary variable which is unity if i and j both belong to the same regional trade 
# agreement,
# •  {Tt} is a comprehensive set of time “fixed effects”,
# •  β and φ are vectors of nuisance coefficients,
# •  Bothinijt is a binary variable which is unity if both i and j are GATT/WTO members at t,  
# •  Oneinijt is a binary variable which is unity if either i or j is a GATT/WTO member at t, 
# •  GSPijt is a binary variable which is unity if i was a GSP beneficiary of j or vice versa at t, and 
# •  εij represents the omitted other influences on bilateral trade, assumed to be well behaved.
#
# I estimate the gravity model using ordinary least squares, computing standard errors that 
# are robust to clustering by country-pairs.  I also include a comprehensive set of year-specific 
# “fixed” effects to account for such factors as the value of the dollar, the global business cycle, 
#  6
# the extent of globalization, oil shocks, and so forth.  Since the data set is a (country-pair x time) 
# panel I also use “random effects” (GLS) and “fixed effects” (“within”) estimators as robustness 
# checks (unless otherwise noted, fixed- and random-effects are always country-pair specific). 

colnames(ross2004)

# > colnames(ross2004)
#  [1] "cty1"     "cty2"     "year"     "pairid"   "landl"    "island"  
#  [7] "border"   "comlang"  "comcol"   "comctry"  "colony"   "curcol"  
# [13] "custrict" "ltrade"   "regional" "lareap"   "ldist"    "lrgdp"   
# [19] "lrgdppc"  "rta"      "sasia1"   "ssafr1"   "easia1"   "highi1"  
# [25] "latca1"   "least1"   "lowin1"   "menaf1"   "midin1"   "sasia2"  
# [31] "ssafr2"   "easia2"   "highi2"   "latca2"   "least2"   "lowin2"  
# [37] "menaf2"   "midin2"   "carib1"   "carib2"   "join1"    "join2"   
# [43] "onein"    "bothin"   "nonein"   "found1"   "found2"   "years1"  
# [49] "years2"   "minyrs"   "maxyrs"   "gsp"      "ecd"      "usi"     
# [55] "naf"      "car"      "pat"      "anz"      "cac"      "mer"     
# [61] "ase"      "spr"      "cty1name" "cty2name"

# subset columns

nrow(ross2004)

ross2004 <- ross2004 %>%
    select(ltrade, bothin, onein, gsp, ldist, lrgdp, lrgdppc, regional,
    custrict, comlang, border, landl, island, lareap, comcol,
    curcol, colony, comctry, cty1 = cty1name, cty2 = cty2name, year) %>%
    mutate(
        year = as.integer(year),
        ctry1 = countrycode(cty1, "country.name", "iso3c"),
        ctry2 = countrycode(cty2, "country.name", "iso3c")
    )

# ! Some values were not matched unambiguously: KYRQYZ REPUBLIC, MOLDVA, PAPUA N.GUINEA, YUGOSLAVIA, SOCIALIST FED. REP. OF

# fix those manually
ross2004 <- ross2004 %>%
    mutate(
        ctry1 = case_when(
            cty1 == "KYRQYZ REPUBLIC" ~ "KGZ",
            cty1 == "MOLDVA" ~ "MDA",
            cty1 == "PAPUA N.GUINEA" ~ "PNG",
            cty1 == "YUGOSLAVIA, SOCIALIST FED. REP. OF" ~ "YUG",
            TRUE ~ ctry1
        ),
        ctry2 = case_when(
            cty2 == "KYRQYZ REPUBLIC" ~ "KGZ",
            cty2 == "MOLDVA" ~ "MDA",
            cty2 == "PAPUA N.GUINEA" ~ "PNG",
            cty2 == "YUGOSLAVIA, SOCIALIST FED. REP. OF" ~ "YUG",
            TRUE ~ ctry2
        )
    )

ross2004 %>%
    filter(is.na(ctry1) | is.na(ctry2))

# create pair variable
ross2004 <- ross2004 %>%
    mutate(pair = paste(pmin(ctry1, ctry2), "-", pmax(ctry1, ctry2))) %>%
    mutate(
        ctry1 = as.factor(ctry1),
        ctry2 = as.factor(ctry2),
        pair = as.factor(pair)
    ) %>%
    select(-cty1, -cty2) %>%
    select(-year, everything())

glimpse(ross2004)

fit <- lm(ltrade ~ bothin + onein + gsp + ldist + lrgdp + lrgdppc + regional +
    custrict + comlang + border + landl + island + lareap + comcol +
    curcol + colony + comctry + year, data = ross2004)

fit_coef <- round(coef(fit), 2)

fit_coef[!grepl("year", names(fit_coef))]

use_data(ross2004, overwrite = TRUE, compress = "xz")
