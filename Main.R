source('Load_data.R')

# Training data set
df = crimes
source("Features.R")

# Include Features ---------------------------------------------------------

data = as.data.frame(cbind(crimes$Meta_Category,features)) # CATEGORY
head(data)


# Model Training -----------------------------------------------------------

require(xgboost)
y=as.integer(data[,1])-1 # CATEGORY 

trainMatrix=as.matrix(data[,-1])
numberOfClasses=max(y)+1
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = numberOfClasses,
              "max.depth" = 3)

# Cross validation
cv.nround = 15
cv.nfold = 3
bst.cv = xgb.cv(param=param, data = trainMatrix, label = y, 
                nfold = cv.nfold, nrounds = cv.nround) #1.58

# Plot
plot(bst.cv$test.mlogloss.mean,type="l",col="red",main = "Error")
lines(bst.cv$train.mlogloss.mean,lty=1)
legend("topright",c("train","test"),col=c("black","red"),lty = 1)

# Train model
nround = 15
bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround)


# Model interpretation ----------------------------------------------------

# Importance matrix
names <- dimnames(trainMatrix)[[2]]
importance_matrix <- xgb.importance(names, model = bst)

# Nice graph
require(Ckmeans.1d.dp)
xgb.plot.importance(importance_matrix)


# Save workspace ----------------------------------------------------------

save.image("workspace.RData")

