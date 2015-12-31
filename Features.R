
# Address processing ------------------------------------------------------

library(stringr)
split_adress = 
  str_split(
    str_replace_all(
      crimes$Address,"([0-9]+[:space:]|Block[:space:]of[:space:])",""),
    "[:space:]/[:space:]")

# Each crime is affected to two streets.
first_street = as.numeric(as.factor(
  rapply(split_adress, function(x) head(x, 1))))-1
second_street = as.numeric(as.factor(
  rapply(split_adress, function(x) tail(x, 1))))-1


# Day of week -------------------------------------------------------------

split_day = as.character(crimes$DayOfWeek)
day = as.numeric(as.factor(split_day)) - 1


# Grid --------------------------------------------------------------------

Y_scaled = round((crimes$Y-min(crimes$Y))/(max((crimes$Y-min(crimes$Y))))*100)
X_scaled = round((crimes$X-min(crimes$X))/(max((crimes$X-min(crimes$X))))*100)
grid = Y_scaled*99+X_scaled-min(Y_scaled*99+X_scaled)


# Year --------------------------------------------------------------------

split_date = str_split(crimes$Dates,"(-|[:space:])")

extract_year = as.numeric(rapply(split_date, function(x) head(x, 1)))
year = extract_year - min(extract_year)


# Month -------------------------------------------------------------------

extract_month = as.numeric(rapply(split_date, function(x) x[2]))
month = extract_month - min(extract_month)


# Hour --------------------------------------------------------------------

hour = as.numeric(str_extract_all(
  rapply(split_date, function(x) tail(x, 1)),
  "^([:digit:]+)"))


