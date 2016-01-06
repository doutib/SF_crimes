source('Load_data.R')

# Training data set
df = crimes
source("Features.R")

# Include Features ---------------------------------------------------------

data = as.data.frame(cbind(Meta_Category = crimes$Meta_Category,features)) # CATEGORY
head(data)


# xgboost -----------------------------------------------------------------
#source("xgboost.R")


# adaboost ----------------------------------------------------------------
source("adaboost.R")


# randomForest ------------------------------------------------------------

source("randomforest.R")


# Save workspace ----------------------------------------------------------

save.image("workspace.RData")

