# Load csv ----------------------------------------------------------------

crimes0 = read.csv("train.csv")

# Remove space outliers ---------------------------------------------------

crimes = crimes0[-which(  crimes0$Y > 37.83 | 
                            crimes0$Y < 37.70 | 
                            crimes0$X > -122.37 | 
                            crimes0$X < -122.52),]

# Meta-classes ------------------------------------------------------------

library(stringr)
Meta_Category = rep("",length(crimes$Category))

# CATEGORY 1
Meta_Category[which(str_detect(crimes$Category,
"(LARCENY/THEFT|VEHICLE THEFT|BURGLARY|ROBBERY|FORGERY/COUNTERFEITING|STOLEN PROPERTY|RECOVERED VEHICLE|EXTORTION|TRESPASS)")
)] = "THEFTS"

# CATEGORY 2
Meta_Category[str_detect(crimes$Category,
"(NON-CRIMINAL|DRIVING UNDER THE INFLUENCE|)")
] = "SOFT"

# CATEGORY 3
Meta_Category[str_detect(crimes$Category,
"(OTHER OFFENSES|SUICIDE|FAMILY OFFENSES|BAD CHECKS|BRIBERY|SEX OFFENSES NON FORCIBL|GAMBLING|PORNOGRAPHY/OBSCENE MAT|TREA)")
] = "OTHERS"

# CATEGORY 4
Meta_Category[str_detect(crimes$Category,
"(WARRANTS|SUSPICIOUS OCC|KIDNAPPING|WEAPON LAWS|ASSAULT)")
] = "CRIMINAL ORGANIZATIONS"

# CATEGORY 5
Meta_Category[str_detect(crimes$Category,
"(DRUG/NARCOTIC|DRUNKENNESS|LIQUOR LAWS)")
] = "ADDICTS"

# CATEGORY 6
Meta_Category[str_detect(crimes$Category,
"(VANDALISM|SEX OFFENSES FORCIBLE|DISORDERLY CONDUCT|ARSON|LOITERING|EMBEZZLEMENT|PROSTITUTION)")
] = "THUGS"

# Add meta-Category
crimes$Meta_Category=as.factor(Meta_Category)
