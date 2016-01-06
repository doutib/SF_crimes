# Address processing ------------------------------------------------------
require(stringr)
split_adress = 
  str_split(
    str_replace_all(
      df$Address,"([0-9]+[:space:]|Block[:space:]of[:space:])",""),
    "[:space:]/[:space:]")

# Each crime is affected to two streets.
first_street = as.numeric(as.factor(
  rapply(split_adress, function(x) head(x, 1))))-1
second_street = as.numeric(as.factor(
  rapply(split_adress, function(x) tail(x, 1))))-1


# Day of week -------------------------------------------------------------

split_day = as.character(df$DayOfWeek)
day_week = as.numeric(as.factor(split_day)) - 1


# Grid --------------------------------------------------------------------

Y_scaled = round((df$Y-min(df$Y))/(max((df$Y-min(df$Y))))*100)
X_scaled = round((df$X-min(df$X))/(max((df$X-min(df$X))))*100)
grid = Y_scaled*99+X_scaled-min(Y_scaled*99+X_scaled)

# Grid Large ---------------------------------------------------------------

Y_scaledL = round((df$Y-min(df$Y))/(max((df$Y-min(df$Y))))*20)
X_scaledL = round((df$X-min(df$X))/(max((df$X-min(df$X))))*20)
gridL = Y_scaled*19+X_scaled-min(Y_scaled*19+X_scaled)


# Year --------------------------------------------------------------------

split_date = str_split(df$Dates,"(-|[:space:])")

extract_year = as.numeric(rapply(split_date, function(x) head(x, 1)))
year = extract_year - min(extract_year)


# Month -------------------------------------------------------------------

extract_month = as.numeric(rapply(split_date, function(x) x[2]))
month = extract_month - min(extract_month)


# Season ------------------------------------------------------------------

season = month
season[which(season == 1 | season == 11 | season == 0 )] = 0
season[which(season == 2 | season == 3  | season == 4 )] = 1
season[which(season == 5 | season == 6  | season == 7 )] = 2
season[which(season == 8 | season == 9  | season == 10)] = 3


# Hour --------------------------------------------------------------------

hour = as.numeric(str_extract_all(
  rapply(split_date, function(x) tail(x, 1)),
  "^([:digit:]+)"))

# Day of Month-------------------------------------------------------------

day_month = as.numeric(rapply(split_date, function(x) x[3])) - 1


# GridL * Year ------------------------------------------------------------

GridL_Year = (gridL+1)*(year+1) - 1


# Intersection ------------------------------------------------------------

Intersection = (first_street==second_street)*first_street


# District ----------------------------------------------------------------

#closest = function(x,y){
#  which.min(apply(y,1,function(r) sum((r-x)^2)))
#}
#library(tcltk)
#district = rep(NA,length(df$X))
#coord = cbind(df$X,df$Y)
#distr = cbind(District$X,District$Y)
#total = length(df$X)
#pb = txtProgressBar(min = 0, max = total, style = 3)
#print('Computing closest district')
#for (i in 1:total){
#  district[i] = closest(coord[i,],distr)
#  setTxtProgressBar(pb, i)
#}
#district = district - 1



# Load features -----------------------------------------------------------

features = data.frame(
  #district = district,
  season = season,
  Intersection = Intersection,
  GridL_Year = GridL_Year,
  day_month = day_month,
  gridL = gridL,
  day_week = day_week,
  hour = hour, 
  year = year, 
  #month = month,
  grid = grid,
  first_street = first_street, second_street = second_street, 
  PdDistrict = as.numeric(df$PdDistrict)) 

