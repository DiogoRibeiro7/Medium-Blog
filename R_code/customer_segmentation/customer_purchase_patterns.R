# Load necessary libraries
library(ggplot2)
library(dplyr)

# Read the data
data <- read.csv('data.csv', stringsAsFactors = FALSE)

# Convert InvoiceDate to Date type
data$InvoiceDate <- as.POSIXct(data$InvoiceDate, format="%m/%d/%Y %H:%M")
data$Date <- as.Date(data$InvoiceDate)

# Create a TotalSales column
data$TotalSales <- data$Quantity * data$UnitPrice

# Aggregate data by CustomerID to analyze purchase frequency and average spending
customer_patterns <- data %>%
  group_by(CustomerID) %>%
  summarise(TotalSpending = sum(TotalSales),
            AverageSpending = mean(TotalSales),
            PurchaseFrequency = n_distinct(InvoiceNo),
            TotalItemsPurchased = sum(Quantity)) %>%
  arrange(desc(TotalSpending))

# Visualizing Total Spending per Customer
ggplot(customer_patterns[1:20, ], aes(x=reorder(as.factor(CustomerID), TotalSpending), y=TotalSpending)) +
  geom_bar(stat="identity", fill="lightblue") +
  coord_flip() +
  labs(title="Top 20 Customers by Total Spending",
       x="CustomerID",
       y="Total Spending") +
  theme_minimal()

# Visualizing Purchase Frequency of Top 20 Customers
ggplot(customer_patterns[1:20, ], aes(x=reorder(as.factor(CustomerID), PurchaseFrequency), y=PurchaseFrequency)) +
  geom_bar(stat="identity", fill="coral") +
  coord_flip() +
  labs(title="Top 20 Customers by Purchase Frequency",
       x="CustomerID",
       y="Purchase Frequency") +
  theme_minimal()

# For product preferences, we focus on the most purchased items
product_preferences <- data %>%
  group_by(Description) %>%
  summarise(TotalQuantitySold = sum(Quantity)) %>%
  arrange(desc(TotalQuantitySold))

# Visualizing Top 20 Most Popular Products
ggplot(product_preferences[1:20, ], aes(x=reorder(Description, TotalQuantitySold), y=TotalQuantitySold)) +
  geom_bar(stat="identity", fill="purple") +
  coord_flip() +
  labs(title="Top 20 Most Popular Products",
       x="Product",
       y="Total Quantity Sold") +
  theme_minimal()
