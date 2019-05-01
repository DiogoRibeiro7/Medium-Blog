##### Fisher's exact test for independency between two categorical variables ------
cont_table <- table(df$Yelp.Stars, df$Inspection.Grade)
fisher.test(cont_table, workspace=2e+08, hybrid = TRUE)
# p-value = 0.7293 -> do not reject H0 : independency b/w row and columns
# Thus, Yelp rating and Inspection grade do not have relationship. 
# There is no dependency between them.

##### Visualization -----
# ref : https://rstudio.github.io/leaflet/
install.packages("leaflet")
library(leaflet)

# import the data 
df <- read.csv("Yelp_Inspection_comparison_output.csv", header=TRUE)

# color pallette
factpal <- colorFactor(c("red","blue","green"), df$Inspection.Grade)

# w/ color
m <- leaflet() %>%
  addTiles() %>%  # Add default OpenStreetMap map tiles
  addCircleMarkers(lng=df$Longitude, lat=df$Latitude, radius = 3, opacity = 1,
                   popup=df$Text, color = factpal(df$Inspection.Grade))
m
