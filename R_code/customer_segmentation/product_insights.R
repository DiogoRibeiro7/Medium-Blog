# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tm)
library(SnowballC)
library(cluster)
library(irlba)

# Read the data
data <- read.csv('data.csv', stringsAsFactors = FALSE)

# Convert Description to UTF-8, replace non-convertible characters with ""
data$Description <- iconv(data$Description, to = "UTF-8", sub = "")

# Proceed with converting to lower case
data$Description <- tolower(data$Description)
data$Description <- gsub("[^a-zA-Z ]", "", data$Description) # Remove non-alphabetical characters
data <- na.omit(data) # Remove rows with NA descriptions


# Create a corpus
corpus <- Corpus(VectorSource(data$Description))

# Text cleaning
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)

# Create a document-term matrix (DTM)
dtm <- DocumentTermMatrix(corpus)

# Convert DTM to a matrix for clustering
dtm_matrix <- as.matrix(dtm)


# Using k-means clustering on the DTM matrix; choose an appropriate value for k
set.seed(123) # For reproducibility
k <- 5 # Example value, adjust based on your data and needs

# Assuming dtm_matrix is your document-term matrix
dtm_sparse <- as(dtm_matrix, "sparseMatrix")

# Apply truncated SVD
svd_out <- irlba(dtm_sparse, nv = 50) # nv is the number of dimensions you want to keep

# Use the output for clustering
dtm_reduced <- svd_out$u

# Now dtm_reduced is much smaller and should be easier to cluster
clusters <- kmeans(dtm_reduced, centers = k, nstart = 25)


# Add cluster assignments to the original data
data$cluster <- clusters$cluster[match(data$Description, rownames(dtm_matrix))]


# Examining top terms in each cluster
findTopTerms <- function(dtm, clusters, k) {
  terms <- colnames(dtm)
  for (i in 1:k) {
    cat("Cluster", i, ":\n")
    cluster_terms <- terms[which.max(colSums(dtm[clusters$cluster == i, ]))]
    print(head(sort(cluster_terms, decreasing = TRUE), 10))
    cat("\n")
  }
}

findTopTerms(dtm_matrix, clusters, k)