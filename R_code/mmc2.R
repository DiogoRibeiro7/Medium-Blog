# Libraries
library(tidyverse)
library(cluster)
library(factoextra)
library(VarSelLCM)
library(reshape2)
library(clusterSim)

# Define the function for silhouette widths
Sil <- function(data) {
  summary(silhouette(clusters$clusters, dist = daisy(data)))[2]
}
# Define the function for cluster visualization
VisClust <- function(data) {
  object <- list(data = data.frame(lapply(data, as.numeric)),
                 cluster = clusters$clusters)
  fviz_cluster(
    object,
    geom = "point",
    ellipse.type = "norm",
    legend = "none",
    main = ""
  )
}
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
# Define the function for ARI and Jaccard in LCCA
VAL.LCCA <- function(data) {
  val.result <- data.frame(matrix(nrow = 10, ncol = 3))
  colnames(val.result) <- c("Model", "ARI", "JC")
  valfun <- function(i) {
    mod <<- VarSelCluster(data, gvals = i, vbleSelec = FALSE)
    val.result[i, 1] <<- i
    val.result[i, 2] <<- ARI(clusters$clusters, mod@partitions@zMAP)
    val.result[i, 3] <<- clusteval::cluster_similarity(clusters$clusters,
      mod@partitions@zMAP,
      similarity = "jaccard")
  }
  lapply(1:10, valfun)
  return(val.result)
}

# Simulate data and apply the functions
# Dataset A
# Write means and covariances into corresponding files
df <- matrix(c(
  1,0,0,0,0,0,
  0,1,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0
), nrow = 6)
write.csv(t(df), "means_23.csv", row.names = FALSE)

cluster1_main_axes = matrix(c(
  3,4,0,0,0,0,
  -4,3,0,0,0,0,
  0,0,1/2,0,0,0,
  0,0,0,5,0,0,
  0,0,0,0,2,0,
  0,0,0,0,0,3
), nrow = 6)
cluster1_cov = t(cluster1_main_axes) %*% cluster1_main_axes
write.csv(cluster1_cov, "cov_23_1.csv", row.names = FALSE)

cluster2_main_axes = matrix(c(
  3/2,2,0,0,0,0,
  -4/2,3/2,0,0,0,0,
  0,0,6,0,0,0,
  0,0,0,7,0,0,
  0,0,0,0,3,0,
  0,0,0,0,0,4
), nrow = 6)
cluster2_cov = t(cluster2_main_axes) %*% cluster2_main_axes
write.csv(cluster2_cov, "cov_23_2.csv", row.names = FALSE)

cluster3_main_axes = matrix(c(
  3/2,2,0,0,0,0,
  -2,3/2,0,0,0,0,
  0,0,2,0,0,0,
  0,0,0,2,0,0,
  0,0,0,0,4,0,
  0,0,0,0,0,5
), nrow = 6)
cluster3_cov = t(cluster3_main_axes) %*% cluster3_main_axes
write.csv(cluster3_cov, "cov_23_3.csv", row.names = FALSE)

cluster4_main_axes = matrix(c(
  -3,-4,0,0,0,0,
  4,-3/2,0,0,0,0,
  0,0,5,0,0,0,
  0,0,0,2,0,0,
  0,0,0,0,7,0,
  0,0,0,0,0,1
), nrow = 6)
cluster4_cov = t(cluster4_main_axes) %*% cluster4_main_axes
write.csv(cluster4_cov, "cov_23_4.csv", row.names = FALSE)

# Create data for clusters
cluster_sizes <- c(200, 500, 600, 250)
categories_num <- 4

set.seed(123)
clusters <- clusterSim::cluster.Gen(
  cluster_sizes,
  fixedCov=FALSE,
  model = 23,
  dataType = "o",
  numCategories = categories_num,
  inputType = 'csv',
  inputRowNames = FALSE
)
dataOrd <- data.frame(lapply(clusters$data, as.factor))
# Results for dataset A
Sil(dataOrd)
VisClust(dataOrd)
LCCAselection(dataOrd)
VAL.LCCA(dataOrd)

# Dataset B
# Write means and covariances into corresponding files
df <- matrix(c(
  -8,-8,-8,-8,-8,-8,
  7,7,7,7,7,7,
  -1,-1,-1,-1,-1,-1,
  -2,-2,-1,-3,-2,6
), nrow = 6)
write.csv(t(df), "means_23.csv", row.names = FALSE)

cluster1_main_axes = matrix(c(
  3,4,0,0,0,0,
  -4,3,0,0,0,0,
  0,0,1/2,0,0,0,
  0,0,0,5,0,0,
  0,0,0,0,2,0,
  0,0,0,0,0,3
), nrow = 6)
cluster1_cov = t(cluster1_main_axes) %*% cluster1_main_axes
write.csv(cluster1_cov, "cov_23_1.csv", row.names = FALSE)

cluster2_main_axes = matrix(c(
  3/2,2,0,0,0,0,
  -4/2,3/2,0,0,0,0,
  0,0,6,0,0,0,
  0,0,0,7,0,0,
  0,0,0,0,3,0,
  0,0,0,0,0,4
), nrow = 6)
cluster2_cov = t(cluster2_main_axes) %*% cluster2_main_axes
write.csv(cluster2_cov, "cov_23_2.csv", row.names = FALSE)

cluster3_main_axes = matrix(c(
  3/2,2,0,0,0,0,
  -2,3/2,0,0,0,0,
  0,0,2,0,0,0,
  0,0,0,2,0,0,
  0,0,0,0,4,0,
  0,0,0,0,0,5
), nrow = 6)
cluster3_cov = t(cluster3_main_axes) %*% cluster3_main_axes
write.csv(cluster3_cov, "cov_23_3.csv", row.names = FALSE)

cluster4_main_axes = matrix(c(
  -3,-4,0,0,0,0,
  4,-3/2,0,0,0,0,
  0,0,5,0,0,0,
  0,0,0,2,0,0,
  0,0,0,0,7,0,
  0,0,0,0,0,1
), nrow = 6)
cluster4_cov = t(cluster4_main_axes) %*% cluster4_main_axes
write.csv(cluster4_cov, "cov_23_4.csv", row.names = FALSE)

# Create data for clusters
cluster_sizes <- c(200, 500, 600, 250)
categories_num <- 4

set.seed(123)
clusters <- clusterSim::cluster.Gen(
  cluster_sizes,
  fixedCov=FALSE,
  model = 23,
  dataType = "o",
  numCategories = categories_num,
  inputType = 'csv',
  inputRowNames = FALSE
)
dataOrd <- data.frame(lapply(clusters$data, as.factor))
# Results for dataset B
Sil(dataOrd)
VisClust(dataOrd)
LCCAselection(dataOrd)
VAL.LCCA(dataOrd)


# Dataset C
# Write means and covariances into corresponding files
df <- matrix(c(
  9,9,9,9,9,9,
  -8,-8,-8,-8,-8,-8,
  3,3,3,3,3,3,
  -2,-2,-2,-2,-2,-2
), nrow = 6)
write.csv(t(df), "means_23.csv", row.names = FALSE)

cluster1_main_axes = matrix(c(
  3/2,2,0,0,0,0,
  -2,3/2,0,0,0,0,
  0,0,1/2,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster1_cov = t(cluster1_main_axes) %*% cluster1_main_axes
write.csv(cluster1_cov, "cov_23_1.csv", row.names = FALSE)

cluster2_main_axes = matrix(c(
  3/2,2,0,0,0,0,
  -2,3/2,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster2_cov = t(cluster2_main_axes) %*% cluster2_main_axes
write.csv(cluster2_cov, "cov_23_2.csv", row.names = FALSE)

cluster3_main_axes = matrix(c(
  3/2,2,0,0,0,0,
  -2,3/2,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster3_cov = t(cluster3_main_axes) %*% cluster3_main_axes
write.csv(cluster3_cov, "cov_23_3.csv", row.names = FALSE)

cluster4_main_axes = matrix(c(
  -3/4,-1,0,0,0,0,
  1,-3/4,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster4_cov = t(cluster4_main_axes) %*% cluster4_main_axes
write.csv(cluster4_cov, "cov_23_4.csv", row.names = FALSE)

# Create data for clusters
cluster_sizes <- c(200, 500, 600, 250)
categories_num <- 4

set.seed(123)
clusters <- clusterSim::cluster.Gen(
  cluster_sizes,
  fixedCov=FALSE,
  model = 23,
  dataType = "o",
  numCategories = categories_num,
  inputType = 'csv',
  inputRowNames = FALSE
)
dataOrd <- data.frame(lapply(clusters$data, as.factor))
# Results for dataset C
Sil(dataOrd)
VisClust(dataOrd)
LCCAselection(dataOrd)
VAL.LCCA(dataOrd)

# Dataset D
# Write means and covariances into corresponding files
df <- matrix(c(
  9,9,9,9,9,9,
  -8,-8,-8,-8,-8,-8,
  3,3,3,3,3,3,
  -2,-2,-2,-2,-2,-2,
  1,1,1,1,1,1,
  2,2,2,2,2,2
), nrow = 6)
write.csv(t(df), "means_23.csv", row.names = FALSE)

cluster1_main_axes = matrix(c(
  3/2,2,0,0,0,0,
  -2,3/2,0,0,0,0,
  0,0,1/2,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster1_cov = t(cluster1_main_axes) %*% cluster1_main_axes
write.csv(cluster1_cov, "cov_23_1.csv", row.names = FALSE)

cluster2_main_axes = matrix(c(
  3/2,2,0,0,0,0,
  -2,3/2,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster2_cov = t(cluster2_main_axes) %*% cluster2_main_axes
write.csv(cluster2_cov, "cov_23_2.csv", row.names = FALSE)

cluster3_main_axes = matrix(c(
  3/2,2,0,0,0,0,
  -2,3/2,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster3_cov = t(cluster3_main_axes) %*% cluster3_main_axes
write.csv(cluster3_cov, "cov_23_3.csv", row.names = FALSE)

cluster4_main_axes = matrix(c(
  -3/4,-1,0,0,0,0,
  1,-3/4,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster4_cov = t(cluster4_main_axes) %*% cluster4_main_axes
write.csv(cluster4_cov, "cov_23_4.csv", row.names = FALSE)

cluster5_main_axes = matrix(c(
  -3,-4,0,0,0,0,
  4,-3,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster5_cov = t(cluster5_main_axes) %*% cluster5_main_axes
write.csv(cluster5_cov, "cov_23_5.csv", row.names = FALSE)

cluster6_main_axes = matrix(c(
  -3/2,-2,0,0,0,0,
  2,-3/2,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster6_cov = t(cluster6_main_axes) %*% cluster6_main_axes
write.csv(cluster6_cov, "cov_23_6.csv", row.names = FALSE)

# Create data for clusters
cluster_sizes <- c(200, 500, 600, 250, 400, 300)
categories_num <- 4

set.seed(123)
clusters <- clusterSim::cluster.Gen(
  cluster_sizes,
  fixedCov=FALSE,
  model = 23,
  dataType = "o",
  numCategories = categories_num,
  inputType = 'csv',
  inputRowNames = FALSE
)
dataOrd <- data.frame(lapply(clusters$data, as.factor))
# Results for dataset D
Sil(dataOrd)
LCCAselection(dataOrd)
VAL.LCCA(dataOrd)


# Dataset E
# Write means and covariances
df <- matrix(c(
  9,9,9,9,9,9,
  -8,-8,-8,-8,-8,-8,
  3,3,3,3,3,3,
  -2,-2,-2,-2,-2,-2,
  6,6,6,6,6,6,
  -5,-5,-5,-5,-5,-5
), nrow = 6)
write.csv(t(df), "means_23.csv", row.names = FALSE)

cluster1_main_axes = matrix(c(
  3/2,2,0,0,0,0,
  -2,3/2,0,0,0,0,
  0,0,1/2,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster1_cov = t(cluster1_main_axes) %*% cluster1_main_axes
write.csv(cluster1_cov, "cov_23_1.csv", row.names = FALSE)

cluster2_main_axes = matrix(c(
  3/2,2,0,0,0,0,
  -2,3/2,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster2_cov = t(cluster2_main_axes) %*% cluster2_main_axes
write.csv(cluster2_cov, "cov_23_2.csv", row.names = FALSE)

cluster3_main_axes = matrix(c(
  3/2,2,0,0,0,0,
  -2,3/2,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster3_cov = t(cluster3_main_axes) %*% cluster3_main_axes
write.csv(cluster3_cov, "cov_23_3.csv", row.names = FALSE)

cluster4_main_axes = matrix(c(
  -3/4,-1,0,0,0,0,
  1,-3/4,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster4_cov = t(cluster4_main_axes) %*% cluster4_main_axes
write.csv(cluster4_cov, "cov_23_4.csv", row.names = FALSE)

cluster5_main_axes = matrix(c(
  -3/8,-1/2,0,0,0,0,
  1/2,-3/8,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster5_cov = t(cluster5_main_axes) %*% cluster5_main_axes
write.csv(cluster5_cov, "cov_23_5.csv", row.names = FALSE)

cluster6_main_axes = matrix(c(
  -3/2,-2,0,0,0,0,
  2,-3/2,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster6_cov = t(cluster6_main_axes) %*% cluster6_main_axes
write.csv(cluster6_cov, "cov_23_6.csv", row.names = FALSE)

# Create data for clusters
cluster_sizes <- c(200, 500, 600, 250, 400, 300)
categories_num <- 4

set.seed(123)
clusters <- clusterSim::cluster.Gen(
  cluster_sizes,
  fixedCov=FALSE,
  model = 23,
  dataType = "o",
  numCategories = categories_num,
  inputType = 'csv',
  inputRowNames = FALSE
)
dataOrd <- data.frame(lapply(clusters$data, as.factor))
# Results for dataset E
Sil(dataOrd)
LCCAselection(dataOrd)
VAL.LCCA(dataOrd)


# Dataset F
# Write means and covariances into corresponding files
df <- matrix(c(
  1,0,0,0,0,0,
  0,1,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)*20
write.csv(t(df), "means_23.csv", row.names = FALSE)

cluster1_main_axes = matrix(c(
  3/4,1,0,0,0,0,
  -1,3/4,0,0,0,0,
  0,0,1/2,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster1_cov = t(cluster1_main_axes) %*% cluster1_main_axes
write.csv(cluster1_cov, "cov_23_1.csv", row.names = FALSE)

cluster2_main_axes = matrix(c(
  3/8,1/2,0,0,0,0,
  -1/2,3/8,0,0,0,0,
  0,0,2,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster2_cov = t(cluster2_main_axes) %*% cluster2_main_axes
write.csv(cluster2_cov, "cov_23_2.csv", row.names = FALSE)

cluster3_main_axes = matrix(c(
  3/2,2,0,0,0,0,
  -2,3/2,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster3_cov = t(cluster3_main_axes) %*% cluster3_main_axes
write.csv(cluster3_cov, "cov_23_3.csv", row.names = FALSE)

cluster4_main_axes = matrix(c(
  -3/4,-1,0,0,0,0,
  1,-3/4,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster4_cov = t(cluster4_main_axes) %*% cluster4_main_axes
write.csv(cluster4_cov, "cov_23_4.csv", row.names = FALSE)

cluster5_main_axes = matrix(c(
  -3/4,-1,0,0,0,0,
  1,-3/4,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster5_cov = t(cluster5_main_axes) %*% cluster5_main_axes
write.csv(cluster5_cov, "cov_23_5.csv", row.names = FALSE)

cluster6_main_axes = matrix(c(
  -3/2,-2,0,0,0,0,
  2,-3/2,0,0,0,0,
  0,0,1,0,0,0,
  0,0,0,1,0,0,
  0,0,0,0,1,0,
  0,0,0,0,0,1
), nrow = 6)
cluster6_cov = t(cluster6_main_axes) %*% cluster6_main_axes
write.csv(cluster6_cov, "cov_23_6.csv", row.names = FALSE)

# Create data for clusters
cluster_sizes <- c(200, 500, 600, 250, 200, 300)
categories_num <- 4

set.seed(123)
clusters <- clusterSim::cluster.Gen(
  cluster_sizes,
  fixedCov=FALSE,
  model = 23,
  dataType = "o",
  numCategories = categories_num,
  inputType = 'csv',
  inputRowNames = FALSE
)
dataOrd <- data.frame(lapply(clusters$data, as.factor))
# Results for dataset F
Sil(dataOrd)
LCCAselection(dataOrd)
VAL.LCCA(dataOrd)

# THE END

