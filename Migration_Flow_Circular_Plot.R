#######################

# PART 1: Package Loading 

library(tidyverse)
#library(tidyr)
library(circlize)
library(readxl)
library(forestmangr)
library(dplyr)
library(magrittr)



# PART 2: Data Preparation

# set working directory
setwd("C:/Users/caltr/PycharmProjects/GA_Intl_Migration/data")

# excel sheets with data
readxl::excel_sheets('abel-database-s2.xlsx')

# choose sheet name and read in its data
data <- read_excel(path = 'abel-database-s2.xlsx', sheet = '2005-10')

# round estimated number of migrated people (package forestmangr)
data_r <- round_df(data, 0, rf = "round")

# read in file with country codes
## this file contains various country codes, allowing us to aggregate countries and corresponding flows at a higher regional level 
codes <- read_excel(path = 'abel-database-s1.xlsx', sheet = 'look up')

# change name of world regions (shorter version)
codes["world_region"][codes["world_region"] == "Central America (incl. Mexico and Carribean)"] <- "Central America"
codes["world_region"][codes["world_region"] == "Central Asia (incl. Russia)"] <- "Central Asia"
codes["world_region"][codes["world_region"] == "Northern Africa (excl. Sudan)"] <- "Northern Africa"


# select variables "world region" and "country iso 3 code" from .csv file
## "world region" contains aggregated regional information for each country
## "country" helps us to assign the regional information to the right country in our migration data
codes_s <- codes %>% select("iso 3 code", world_region)

# join variables "world region" and "country iso3 code" ("codes-s" data set) with our migration data (data-r)
data_codes <-  left_join(data_r, codes_s, by = "iso 3 code")

# group origin countries by "cregion" and summarise the flows within each group
regflow1 <- data_codes %>% 
  group_by(world_region) %>%
  summarise_if(is.numeric, funs(sum)) %>%
  ungroup()

# outflow to long format (preparing data for calculating inflows)
regflow_long <- regflow1 %>%
  pivot_longer(ABW:ZWE, names_to = "destination", values_to = "flows")

# select variables "cregion" and "country" in "codes" and rename them
## we rename these variables because we now use them to summarise inflows of countries within the same region
## summarising inflows requires us to identify to which countries/regions migrants are moving
## hence, we name the variables "dest_region" and "destination"
codes_rn <- codes %>% dplyr::rename(dest_region = world_region, destination = "iso 3 code") %>%
  select(dest_region, destination)

# join "dest_region" variable ("codes-rn" data set) with outflow-long
## since we already prepared "outflow-long" by turning it to long format and listing the countries in a variable named "destination",
## the "destination" variable in "codes-rn" helps us assigning the information of regional destination ("dest_region") to our main data set
regflow_codes <- left_join(regflow_long, codes_rn, by = "destination")

# group countries by "world region" and by "dest_region" and summarise the flows within each group
regflow2 <- regflow_codes %>% 
  group_by(world_region, dest_region) %>%
  dplyr::summarise(flows = sum(flows)) %>% # if you do not use "dplyr::" summarise command is not recognized
  ungroup()

# transform migration data into adjacency matrix by turning it to wide format
matrix <- regflow2 %>% 
  pivot_wider(names_from = dest_region, values_from = flows)

# remove na values
matrix_no_na <- matrix %>% drop_na()
matrix_no_na <- matrix_no_na[,1:15]

# use values of first column "world_region" as row names
mat <- column_to_rownames(matrix_no_na, var = "world_region")

# transform migration data set into real matrix
rmat <-  mat %>%
  as.matrix()


##### CIRCULAR  ####

# preliminaries

# SVG graphics device
svg(file="plot_circular_flow_countries.svg")

# in case the code is run a couple of times, the following command disables layout function
par(mfrow=c(1,1))

# create matrix for link visibility
visible <- matrix(TRUE, nrow = nrow(rmat), ncol = ncol(rmat))
visible[rmat < 400000] = FALSE

# initialize circular plot

# (re)set parameters
circos.clear()
circos.par(
  
  # where to start "drawing" the circle
  start.degree = 145, 
  
  # gaps between segments
  gap.degree = c(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2), 
  
  # gaps between segments and links
  track.margin = c(0.001, 0.001), 
  
  # turn off warnings for plotting outside cell
  points.overflow.warning = FALSE)


# circular plot


chordDiagram(
  
  # matrix
  rmat, 
  
  # change order of segments
  order = c("South America", "Central America", "North America", 
            "Sub-Saharan Africa", "Northern Africa", 
            "Northern Europe", "Western Europe", "Southern Europe", "Eastern Europe",
            "Central Asia", "Western Asia", "South Asia", "South-East Asia", 
            "Oceania"), 
  
  # segment and link colours
  grid.col = c("#eb7005", "#f23b2e", "#ba0000", 
               "#8f6b01", "#e3d802", 
               "#079605", "#618f06", "#27fc0f", "#014a12",
               "#7007e8", "#111df5", "#00a3b5", "#5aede6", 
               "#bd0dd1"),
  
  
  # transparency of links
  transparency = 0.2,
  link.visible = visible,
  
  # determine direction of links (i.e. direction of relationship between cells)
  directional = 1,
  diffHeight = -0.03,
  
  # how directionality is represented
  direction.type = "diffHeight", 
  
  # style and order of links
  link.sort = TRUE, 
  link.largest.ontop = TRUE,
  
  # adjust height of all links
  h.ratio = 0.85,
  
  # defining outer segments
  annotationTrack = "grid", 
  annotationTrackHeight = c(0.03, 0.01)
  
)


# Add text and axis
circos.trackPlotRegion( # or short: circos.track()
  track.index = 1, 
  bg.border = NA, 
  panel.fun = function(x, y) { # applies plotting to a cell. it thus requires x and y values
    
    xlim = get.cell.meta.data("xlim") # get.cell.meta.data = obtain info of current cell
    sector.index = get.cell.meta.data("sector.index") 
    
    # parameters for text 
    circos.text(
      
      # text location
      x = mean(xlim),
      y = 3.2,
      
      # names of segments
      labels = sector.index,
      
      # facing of names
      facing = "bending",
      niceFacing = TRUE,
      
      # font size
      cex = 0.8,
      
      # font color
      col = "black"
    )
  }
)

# close .svg device
dev.off()
