source("Load_data.R")
source("Features.R")

# Include Features ---------------------------------------------------------

data = data.frame(Category = as.numeric(crimes$Category)-1,
                  day = day, hour = hour, year = year, month = month,
                  grid = grid, #2.71
                  first_street = first_street, second_street = second_street, #2.73
                  PdDistrict = as.numeric(crimes$PdDistrict)) #2.82


# Model Training -----------------------------------------------------------

require(xgboost)
y=as.matrix(data[,1])
trainMatrix=as.matrix(data[,-1])
numberOfClasses=max(y)+1
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = numberOfClasses)
cv.nround = 5
cv.nfold = 3

bst.cv = xgb.cv(param=param, data = trainMatrix, label = y, 
                nfold = cv.nfold, nrounds = cv.nround)

nround = 20
bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround)


# Model interpretation ----------------------------------------------------

# Importance matrix
names <- dimnames(trainMatrix)[[2]]
importance_matrix <- xgb.importance(names, model = bst)

# Nice graph
require(Ckmeans.1d.dp)
xgb.plot.importance(importance_matrix)

# Plot tree
require(DiagrammeR)
xgb.plot.tree(feature_names = names, model = bst, n_first_tree = 2)


# Save workspace ----------------------------------------------------------

save.image("workspace.RData")

