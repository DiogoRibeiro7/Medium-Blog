library(dplyr)
library(lubridate)

# Assuming 'data' has been read previously and is still available
data <- read.csv('data.csv', stringsAsFactors = FALSE)


# Ensure Recency is calculated correctly and there are no all-NA scenarios
data$InvoiceDate <- as.Date(data$InvoiceDate)
current_date <- max(data$InvoiceDate, na.rm = TRUE) + 1
data$Recency <- as.numeric(current_date - data$InvoiceDate)

# Proceed only with rows having non-NA Recency
valid_data <- data[!is.na(data$Recency), ]

# Calculate RFM metrics again on the valid dataset
rfm_data <- valid_data %>%
  group_by(CustomerID) %>%
  summarise(Recency = as.numeric(current_date - max(InvoiceDate, na.rm = TRUE)),
            Frequency = n_distinct(InvoiceNo),
            Monetary = sum(Quantity * UnitPrice, na.rm = TRUE)) %>%
  filter(!is.na(CustomerID) & Monetary > 0)

# Adjusted score function to directly address potential issues
score <- function(x) {
  qs <- quantile(x, probs = seq(0, 1, by = 0.2), na.rm = TRUE)
  # Ensure at least one break between min and max if not enough unique quantiles
  if(length(unique(qs)) < 5) {
    qs <- c(min(x, na.rm = TRUE), max(x, na.rm = TRUE))
  }
  return(qs)
}

# Try to recalculate R_score with the adjusted approach
rfm_data$R_score <- cut(rfm_data$Recency, breaks = score(rfm_data$Recency), labels = FALSE, include.lowest = TRUE)
rfm_data$F_score <- cut(rfm_data$Frequency, breaks = score(rfm_data$Frequency), labels = FALSE, include.lowest = TRUE)
rfm_data$M_score <- cut(rfm_data$Monetary, breaks = score(rfm_data$Monetary), labels = FALSE, include.lowest = TRUE)

# Convert numeric scores to desired range (1-5)
# This approach ensures that the labels are matched correctly, even if the number of breaks is not as expected
# Adjust label assignment based on the actual range of breaks
rfm_data$R_score <- as.integer(scale(rfm_data$R_score, center = FALSE, scale = max(rfm_data$R_score)/5))
rfm_data$F_score <- as.integer(scale(rfm_data$F_score, center = FALSE, scale = max(rfm_data$F_score)/5))
rfm_data$M_score <- as.integer(scale(rfm_data$M_score, center = FALSE, scale = max(rfm_data$M_score)/5))

# Ensure scores are within the 1-5 range after scaling
rfm_data$R_score <- pmin(pmax(rfm_data$R_score, 1), 5)
rfm_data$F_score <- pmin(pmax(rfm_data$F_score, 1), 5)
rfm_data$M_score <- pmin(pmax(rfm_data$M_score, 1), 5)

# Combine RFM scores into a single score string
rfm_data$RFM_Score <- paste0(rfm_data$R_score, rfm_data$F_score, rfm_data$M_score)
# Analyzing RFM segments
rfm_segments <- rfm_data %>%
  group_by(RFM_Score) %>%
  summarise(Count = n(),
            Average_Recency = mean(Recency),
            Average_Frequency = mean(Frequency),
            Average_Monetary = mean(Monetary))

# View the top segments
head(rfm_segments[order(-rfm_segments$Count), ])

# This is a basic approach. For more detailed segmentation, consider clustering techniques like K-means on RFM scores.
