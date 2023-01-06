#######################

# PART 1: Package Loading 

library(tidyverse)
library(circlize)
library(readxl)
library(forestmangr)
library(dplyr)
library(magrittr)



# PART 2: Data Preparation

# set working directory
setwd("C:/Users/caltr/OneDrive/Documents/UniPisa/Y2S1/Geospatial_Analytics/Project")

# excel sheets with data
readxl::excel_sheets('abel-database-s1-mod.xlsx')

# choose sheet name and read in its data
data <- read_excel(path = 'abel-database-s1-mod.xlsx', sheet = 'flow estimates by region 2005', range = cell_cols("B:Q"))

data <- data[-1,]

# round estimated number of migrated people (package forestmangr)
# data_r <- round_df(data, 0, rf = "round")


colnames(data) <- c( "world_region", 
                     "North America",
                     "Central America",
                     "South America",
                     "North Africa",
                     "Sub-Saharan Africa",
                     "Northern Europe",
                     "Western Europe",
                     "Southern Europe",
                     "Eastern Europe",
                     "Central Asia",
                     "Western Asia",
                     "South Asia",
                     "East Asia",
                     "South-East Asia",
                     "Oceania")

# use values of first column "world_region" as row names
mat <- column_to_rownames(data, var = "world_region")

# transform migration data set into real matrix
rmat <-  mat %>%
  as.matrix()

class(rmat) <- "numeric"


##### CIRCULAR  ####

# preliminaries

# SVG graphics device
svg(file="Circular_mig_flowregions_.svg")

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
  gap.degree = c(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2), 
  
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
            "Sub-Saharan Africa", "North Africa", 
            "Northern Europe", "Western Europe", "Southern Europe", "Eastern Europe",
            "Central Asia", "Western Asia", "East Asia", "South Asia", "South-East Asia", 
            "Oceania"), 
  
  # segment and link colours
  grid.col = c("#eb7005", "#f23b2e", "#ba0000", 
               "#8f6b01", "#e3d802", 
               "#079605", "#618f06", "#27fc0f", "#014a12",
               "#7007e8", "#111df5", "#061f4d", "#00a3b5", "#5aede6", 
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
