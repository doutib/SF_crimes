library(neuralnet)

data.nnet = data
train = sample(1:nrow(data.nnet),nrow(data.nnet)/2)
test = -train



network = neuralnet(Meta_Category ~ season+Intersection+GridL_Year+gridL+day_week+hour+year+grid+first_street+second_street+PdDistrict
                    ,data = data.nnet[train,], hidden = 25, lifesign ="minimal",  threshold = 0.01)


#plot(network, rep = "best")

temp_test <- subset(data.nnet[test,], select = c("season","Intersection","GridL_Year","gridL","day_week","hour","year","grid","first_street","second_street","PdDistrict"))
network.results = compute(network,temp_test)


results <- round(data.frame(actual = data.nnet[test,]$Meta_Category, prediction = network.results$net.result))

table(results)
