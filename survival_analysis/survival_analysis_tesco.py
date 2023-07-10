import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import qth_survival_times

# First, download the dataset from the link below:
# https://www.kaggle.com/blastchar/telco-customer-churn
# Make sure you have the necessary permissions and save the CSV file in the same directory as this script.

# Read the data
df = pd.read_csv('Telco-Customer-Churn.csv')

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop any rows with missing data
df.dropna(inplace=True)

# Convert the 'Churn' column to numeric
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Convert categorical variables to numeric
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['Contract'] = df['Contract'].apply(lambda x: 1 if x == 'One year' else 0 if x == 'Month-to-month' else 2)

# Fit the Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'Contract', 'Churn']], 'tenure', event_col='Churn')

# Create a survival function for a new customer
new_customer = pd.DataFrame.from_dict({
    'tenure': [0],
    'MonthlyCharges': [70],
    'TotalCharges': [0],
    'gender': [1],
    'Contract': [1],
    'Churn': [0]
})

# Predict the survival function of the new customer
new_customer_survival_function = cph.predict_survival_function(new_customer)

# Calculate CLV for different percentile survival times (e.g., 25th, 50th, 75th percentiles)
percentiles = [0.25, 0.5, 0.75]
clv = {}
for p in percentiles:
    survival_time = qth_survival_times(p, new_customer_survival_function)
    clv[f'{int(p*100)}_percentile'] = survival_time * new_customer['MonthlyCharges'].values[0]
    
print(f'The CLV of the new customer at different percentiles is {clv}')