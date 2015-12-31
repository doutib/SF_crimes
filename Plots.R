
# Load SF Map -------------------------------------------------------------
rm(list=ls()) 
library(ggplot2)
library(ggmap)

sf_center = as.numeric(geocode("san francisco"))
sf_map = ggmap(get_googlemap(center = c(-122.43,37.75), zoom = 12), extent = "normal")  
sf_map

# Plot --------------------------------------------------------------------
"LARCENY/THEFT"
"OTHER OFFENSES"
"NON-CRIMINAL"
"ASSAULT"
"DRUG/NARCOTIC"
crimes_sub1 = crimes[which(crimes$Category=="LARCENY/THEFT"),]
crimes_sub2 = crimes[which(crimes$Category=="ASSAULT"),]
p = sf_map + 
  geom_point(data = crimes_sub1, aes(x = X, y = Y), 
             size = 1, colour = "blue", alpha = 0.02) + 
  geom_point(data = crimes_sub2, aes(x = X, y = Y), 
             size = 1, colour = "orange", alpha = 0.01)
p

