# Load csv ----------------------------------------------------------------

crimes0 = read.csv("train.csv")

# Remove space outliers ---------------------------------------------------

crimes = crimes0[-which(  crimes0$Y > 37.83 | 
                          crimes0$Y < 37.70 | 
                          crimes0$X > -122.37 | 
                          crimes0$X < -122.52),]



