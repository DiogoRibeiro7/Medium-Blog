library(mclust)

data(iris)
X <- iris[, -5]  # We exclude the species column to perform unsupervised clustering

model <- Mclust(X)
summary(model)

plot(model, what = "classification")

clusters <- model$classification  # Cluster assignments
probabilities <- model$z          # Probabilities of cluster memberships
