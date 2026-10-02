# Load necessary libraries
library(ggplot2)
library(dplyr)

# Load the exact pinned canonical Online Retail II source.
source("load_online_retail_ii.R")
data <- load_online_retail_ii()

# Convert InvoiceDate to Date type and create a TotalSales column
data$InvoiceDate <- as.POSIXct(data$InvoiceDate, format="%m/%d/%Y %H:%M")
data$Date <- as.Date(data$InvoiceDate)
data$TotalSales <- data$Quantity * data$UnitPrice

# Aggregate data by Date
daily_sales <- data %>%
  group_by(Date) %>%
  summarise(TotalSales = sum(TotalSales))

# Plot sales trends over time
ggplot(daily_sales, aes(x=Date, y=TotalSales)) +
  geom_line() +
  labs(title="Sales Trends Over Time",
       x="Date",
       y="Total Sales") +
  theme_minimal()
