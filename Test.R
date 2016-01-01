
# Train model -------------------------------------------------------------

source("Main.R")
# Test data set
df = crimes_test
source("Features.R")

# Include Features ---------------------------------------------------------

data_test = data.frame(#day = day,
                       hour = hour, year = year, month = month,
                       grid = grid,
                       first_street = first_street, second_street = second_street, 
                       PdDistrict = as.numeric(crimes_test$PdDistrict)) 

# Model Testing  -----------------------------------------------------------

require(xgboost)

xgb.pred = predict(bst,as.matrix(data_test))
pred = matrix(xgb.pred,ncol = n.cat,byrow = T)
head(pred)

# Scale --------------------------------------------------------------------

pred_final = matrix(ncol=39,nrow=nrow(pred))

cursor=1
for (i in 1:n.cat){
  pred_final[,cursor:(cursor+length(prop_cat[[i]])-1)] = tcrossprod(pred[,i],prop_cat[[i]])
  cursor = cursor+length(prop_cat[[i]])
}

pred_final = as.data.frame(cbind(crimes_test$Id,pred_final))
names(pred_final) = c("Id",rapply(names_cat, function(x) x))
head(pred_final)

# Write csv ---------------------------------------------------------------

write.csv(pred_final,file = "submission.csv",quote = F,row.names = F)
