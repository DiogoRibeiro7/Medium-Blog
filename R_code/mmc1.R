# Libraries
# Pre-processing and visualization
library(tidyverse)
library(haven)
library(car)
library(misty)
library(psych)
library(viridis)
library(plyr)
library(likert)
# Missing data
library(VIM)
library(missForest)
# Clustering
library(cluster)
library(factoextra)
library(reshape2)
# LCCA
library(VarSelLCM)
# Read the data from your working directory
# Retrieved from https://www.iea.nl/data-tools/repository/icils
dataAll.Lab <- read_sav("BTGDEUI2.sav")
# This format has labels we need to remove
dataAll <- as.data.frame(haven::zap_labels(dataAll.Lab))

# Select variables
dataVarMis <- dplyr::select(dataAll, starts_with("IT2G18"))
colnames(dataVarMis) <- c(LETTERS[1:13])

# Check ICC with school ID
dataSch <- cbind(dataVarMis, dataAll$IDSCHOOL)
misty::multilevel.icc(dataSch, group = dataSch$`dataAll$IDSCHOOL`)
length(unique(dataAll$IDSCHOOL))

# Deal with missing data
# Remove rows with 100% missing
NArows <- rowSums(is.na(dataVarMis)) / ncol(dataVarMis)
sum(NArows == 1)
sum(NArows == 1) / nrow(dataVarMis)
dataVar <- dataVarMis[NArows < 1,]
# Missing data
mean(is.na(dataVar))
# Aggregation plot
VIM::aggr(dataVar, cex.axis = 0.8)
# Impute with random forest
set.seed(2)
predict <- stats::predict # was overwritten by VarSelLCM
dataVarImp <- missForest(as.data.frame(lapply(dataVar, as.factor)))
dataVarNew <- as.data.frame(dataVarImp$ximp)

# Recode positive items: the higher score the more positive ICT view
keys <- c(1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1)
dataVarNew2 <- as.data.frame(lapply(dataVarNew, as.numeric))
dataVarNew3 <- as.data.frame(reverse.code(keys, dataVarNew2, mini = 1, maxi = 4))
dataBoth <- as.data.frame(lapply(dataVarNew3, as.factor))
colnames(dataBoth) <- c(LETTERS[1:13])

# Visualize response frequencies
likert::likert.bar.plot(likert(dataBoth), ordered = F, plot.percent.low = F, plot.percent.high = F, plot.percents = T)

# Select positive and negative separately
dataNeg <- dplyr::select(dataBoth, c("A", "D", "F", "G", "H", "I")) # left for the reader
dataPos <- dplyr::select(dataBoth, c("B", "C", "E", "J", "K", "L", "M"))

## Select the number of clusters
# Define the LCCAselection function
LCCAselection <- function(data) {
  for (j in 1:ncol(data)) {
    if (identical(class(data[, j]), "factor") == FALSE)
      stop('non-factor column(s) in the data')
  }
  mydaisy <- daisy(data)
  result <- data.frame(matrix(nrow = 10, ncol = 4))
  colnames(result) <- c("Model", "BIC", "ICL", "ASW")
  clustfun <- function(i) {
    clust <- VarSelCluster(data, gvals = i, vbleSelec = FALSE)
    result[i, 1] <<- i
    result[i, 2] <<- -2 * clust@criteria@BIC
    result[i, 3] <<- clust@criteria@ICL
    result[i, 4] <<- summary(silhouette(clust@partitions@zMAP, dist = mydaisy))[4]
  }
  lapply(1:10, clustfun)
  result$Model <- as.factor(result$Model)
  result$BIC <- round(result$BIC, 2)
  result$ICL <- round(result$ICL, 2)
  result$ASW <- round(as.numeric(result$ASW), 2)
  mm <- melt(subset(result, select = c(Model, BIC, ASW)), id.var = "Model")
  plt <- ggplot(mm, aes(x = Model, y = value, group = 1)) +
    facet_grid(variable ~ ., scales = "free_y") +
    geom_line(aes(color = variable), lwd = 1) +
    theme(legend.position = "bottom") +
    geom_vline(aes(xintercept = which.max(result$ICL), linetype = "min_ICL"), size = 0.7) +
    geom_vline(aes(xintercept = which.min(result$BIC), linetype = "min_BIC"), size = 0.7) +
    guides(colour = F) +
    scale_linetype_manual(
      name = "Index",
      values = c("min_ICL" = "dotdash", "min_BIC" = "dotted"))
  print(plt)
  return(result)
}
# LCCAselection for positive dataset
set.seed(2)
SelectP <- LCCAselection(dataPos)
# for negative
# SelectN <- LCCAselection(dataNeg)

# Get models for 4 and 6 clusters
set.seed(2)
modP4 <- VarSelCluster(dataPos, gvals = 4, vbleSelec = FALSE)
modP6 <- VarSelCluster(dataPos, gvals = 6, vbleSelec = FALSE)

# Validation
# Define the function
valfunc <- function(data, k, n){
  resval <- data.frame(matrix(nrow = n, ncol = 2))
  colnames(resval) <- c("ARI", "Jaccard")
  mod <- VarSelCluster(data, gvals = k, vbleSelec = FALSE)
  fun <- function(i){
    bsampN <- sample(nrow(data),replace=TRUE)
    dataN <- data[bsampN, ]
    modN <- VarSelCluster(dataN, gvals = k, vbleSelec = FALSE)
    partN <- VarSelLCM::predict(mod, dataN, type = "partition")
    arival <- ARI(partN, modN@partitions@zMAP)
    jcval <- clusteval::cluster_similarity(
      partN,
      modN@partitions@zMAP,
      similarity = "jaccard")
    resval[i, 1] <<- arival
    resval[i, 2] <<- jcval
  }
  lapply(1:n, fun)
  return(colMeans(resval))
}

# Assess the stability for 4 clusters
set.seed(2)
valModP4 <- valfunc(dataPos, 4, 20) # run it with 100
# and for 6 clusters
set.seed(2)
valModP6 <- valfunc(dataPos, 6, 20) # run it with 100

# Check population shares
modP4@param@pi
modP6@param@pi

# Visualize the final 4-cluster solution
# Simpler barplot
barplot(
  modP4@criteria@discrim,
  horiz = T,
  col = viridis(7),
  xlab="Discriminative power",
  ylab="Items")

# Visualize clusters
objectP4 <- list(data = data.frame(lapply(dataPos, as.numeric)),
                 cluster = modP4@partitions@zMAP)
fviz_cluster(
  objectP4,
  geom = "point",
  ellipse.type = "norm",
  legend = "none",
  main = "",
  font.x = 16,
  font.y = 16,
  font.tickslab = 14
)
# Cluster silhouettes
sil <- silhouette(modP4@partitions@zMAP, dist = daisy(dataPos))
fviz_silhouette(
  sil,
  palette = "Set2",
  xlab = "Clusters",
  legend = "none",
  main = "",
  font.x = 16,
  font.y = 16,
  font.tickslab = 14
)

# Item probability plot
lcmodel <- reshape2::melt(modP4@param@paramCategorical@alpha)
zp1 <- ggplot(lcmodel, aes(x = L1, y = value, fill = Var2))
zp1 <- zp1 + geom_bar(stat = "identity", position = "stack")
zp1 <- zp1 + facet_wrap(~ Var1)
zp1 <- zp1 + labs(x = "Items", y = "Response Probability", fill = "Response")
print(zp1)

# Reorder the classes and run lines 195-199 again
lcmodel$Var1 <- car::recode(
  lcmodel$Var1,
  "'class-1' = 'class-2';'class-2' = 'class-3';
  'class-3' = 'class-4';'class-4' = 'class-1'")
# THE END
