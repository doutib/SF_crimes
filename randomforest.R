require(randomForest)



# Cross validation --------------------------------------------------------

train = sample(1:nrow(data),nrow(data/2))
randomForest(y= as.factor(data$Meta_Category), x=data[,-1],,subset = train, importance = TRUE)
