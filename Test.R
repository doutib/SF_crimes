source("Load_data.R")

# Train model -------------------------------------------------------------

source("Main.R")
# Test data set
df = crimes_test
source("Features.R")

# Include Features ---------------------------------------------------------

data_test = features

# Model Testing  -----------------------------------------------------------

head(data_test)
xgb.pred = predict(bst,as.matrix(data_test))
pred = matrix(xgb.pred,ncol = 39,byrow = T) # CATEGORY
head(pred)

dim(pred)

# Scale --------------------------------------------------------------------

pred_final = pred

pred_final = as.data.frame(cbind(as.integer(crimes_test$Id),pred_final))
names(pred_final) = c("Id",rapply(names_cat, function(x) x))

pred_final = ceiling(pred_final*100)/100
pred_final$Id = as.integer(pred_final$Id)

pred_final = pred_final[,c(1,1+order(names(pred_final)[2:40]))]
head(pred_final)


# Write csv ---------------------------------------------------------------

write.csv(pred_final,file = "submission.csv",quote = F,row.names = F)

