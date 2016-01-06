require(adabag)

head(data)

# Bagging -----------------------------------------------------------------

# CV
bag = bagging.cv(Meta_Category ~.,data=data,v=3,mfinal=1, control=rpart.control(maxdepth=5,cp = 0.001))


# Boosting ----------------------------------------------------------------

# CV
boost = boosting.cv(Meta_Category ~.,data=data,v=3,mfinal=2, control=rpart.control(maxdepth=5,cp = 0.001))


# Save data ---------------------------------------------------------------

save(bag,boost,file = "adaboost.Rdata")
