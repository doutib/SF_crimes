# Load csv ----------------------------------------------------------------

crimes0 = read.csv("train.csv")
crimes_test = read.csv("test.csv")
#District = read.csv("district.csv")
#papa_sub = read.csv("submissionPAPA.csv")


# Remove space outliers ---------------------------------------------------

crimes = crimes0[-which(  crimes0$Y > 37.83 | 
                            crimes0$Y < 37.70 | 
                            crimes0$X > -122.37 | 
                            crimes0$X < -122.52),]

# Category ----------------------------------------------------------------

Cat1 = c("LARCENY/THEFT",
         "VEHICLE THEFT",
         "BURGLARY",
         "ROBBERY",
         "FORGERY/COUNTERFEITING",
         "STOLEN PROPERTY",
         "RECOVERED VEHICLE",
         "EXTORTION",
         "TRESPASS")

Cat2 = c("NON-CRIMINAL",
         "DRIVING UNDER THE INFLUENCE")

Cat3 = c("OTHER OFFENSES",
         "SUICIDE",
         "FAMILY OFFENSES",
         "BAD CHECKS",
         "BRIBERY",
         "SEX OFFENSES NON FORCIBLE",
         "GAMBLING",
         "PORNOGRAPHY/OBSCENE MAT",
         "TREA",
         "MISSING PERSON")

Cat4 = c("WARRANTS",
         "SUSPICIOUS OCC",
         "KIDNAPPING",
         "WEAPON LAWS",
         "ASSAULT",
         "SECONDARY CODES")

Cat5 = c("DRUG/NARCOTIC",
         "DRUNKENNESS",
         "LIQUOR LAWS")

Cat6 = c("VANDALISM",
         "SEX OFFENSES FORCIBLE",
         "DISORDERLY CONDUCT",
         "ARSON",
         "LOITERING",
         "EMBEZZLEMENT",
         "PROSTITUTION",
         "FRAUD",
         "RUNAWAY")

names_cat = list(Cat1,Cat2,Cat3,Cat4,Cat5,Cat6)
n.cat = length(names_cat)


# Meta-classes ------------------------------------------------------------

require(stringr)
  
Meta_Category = rep("",length(crimes$Category))
concatenate_cat = function(cat){
  paste("(",paste(cat,sep = "",collapse = "|"),")",sep="",collapse="")
}

# CATEGORY 1
Meta_Category[str_detect(crimes$Category,concatenate_cat(Cat1)
)] = "1_THEFTS"

# CATEGORY 2
Meta_Category[str_detect(crimes$Category,concatenate_cat(Cat2)
)] = "2_SOFT"

# CATEGORY 3
Meta_Category[str_detect(crimes$Category,concatenate_cat(Cat3)
)] = "3_OTHERS"

# CATEGORY 4
Meta_Category[str_detect(crimes$Category,concatenate_cat(Cat4)
)] = "4_CRIMINAL ORGANIZATIONS"

# CATEGORY 5
Meta_Category[str_detect(crimes$Category,concatenate_cat(Cat5)
)] = "5_ADDICTS"

# CATEGORY 6
Meta_Category[str_detect(crimes$Category,concatenate_cat(Cat6)
)] = "6_THUGS"

# Add meta-Category
crimes$Meta_Category=as.factor(Meta_Category)

# Proportion --------------------------------------------------------------

prop_cat = list()
for (i in 1:n.cat){
  cat = names_cat[[i]]
  tot_sum = sum(str_detect(crimes$Category,paste(cat,sep = "",collapse = "|")))
  prop_cat[[i]] = sapply(cat,function(x) sum(str_detect(crimes$Category,x)))/tot_sum
}

