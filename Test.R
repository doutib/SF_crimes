
# Train model -------------------------------------------------------------

source("Main.R")
# Test data set
df = crimes_test
source("Features.R")

# Include Features ---------------------------------------------------------

data_test = features

# Save workspace ----------------------------------------------------------

save.image("workspace.RData")


# Model Testing  -----------------------------------------------------------

require(xgboost)
xgb.pred = predict(bst,as.matrix(data_test))
pred = matrix(xgb.pred,ncol = n.cat,byrow = T) # CATEGORY

head(pred)

# Scale --------------------------------------------------------------------

options(scipen=10)

# CATEGORY : original classes
#pred_final = pred
#pred_final = as.data.frame(cbind(as.integer(crimes_test$Id),pred_final))
#names(pred_final) = c("Id",sort(rapply(names_cat, function(x) x)))
#pred_final[,2:(ncat+1)] = floor(pred_final[,2:(ncat+1)]*1000000)/1000000

# CATEGORY : meta classes
pred_final = matrix(nrow=nrow(pred),ncol=39)
counter = 1
for (i in 1:length(prop_cat)){
  pred_final[,counter:(counter+length(prop_cat[[i]])-1)] = tcrossprod(pred[,i],prop_cat[[i]])
  counter = counter+length(prop_cat[[i]])
}
pred_final = floor(pred_final*100000)/100000
pred_final = as.data.frame(pred_final)
names(pred_final) = rapply(names_cat, function(x) x)
pred_final = pred_final[,order(names(pred_final))]
pred_final = cbind(Id = as.integer(crimes_test$Id), pred_final)

head(pred_final)
head(papa_sub)
ensemble = pred_final
ensemble[,2:40] = sqrt(pred_final[,2:40]*papa_sub[,2:40])
ensemble[,2:40] = floor(ensemble[,2:40]*100000)/100000
head(ensemble)
sum(ensemble[2,])
ensemble[200001,]

# Write csv ---------------------------------------------------------------

write.csv(ensemble,file = "submission_ensemblePAPA_geom.csv",quote = F,row.names = F)

