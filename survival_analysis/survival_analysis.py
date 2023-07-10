import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter

# Read the data
df = pd.read_csv('Telco-Customer-Churn.csv')

# Convert TotalCharges to numeric (there are some blank strings which are not valid float)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop any rows with missing data
df.dropna(inplace=True)

# Convert the 'Churn' column to numeric
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Assume 'tenure' is the duration
# and 'Churn' is the event_observed
kmf = KaplanMeierFitter()
kmf.fit(df['tenure'], event_observed=df['Churn'])

# Create a survival function for a new customer with a monthly bill of $70
new_customer = pd.DataFrame.from_dict({
    'tenure': [0],
    'MonthlyCharges': [70],
    'TotalCharges': [0],
    'Churn': [0]
})

# Assume the average lifetime of a customer is the median survival time
average_lifetime = kmf.median_survival_time_
average_monthly_profit = new_customer['MonthlyCharges'][0]

# Calculate the CLV
clv = average_lifetime * average_monthly_profit
print(f'The CLV of the new customer is ${clv}')

# A more advanced version would take into account the survival function
# We can do this using the Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']], 'tenure', event_col='Churn')

# Calculate the survival function of the new customer
new_customer_survival_function = cph.predict_survival_function(new_customer)

# The CLV would then be the integral of the survival function * monthly profit
clv_advanced = np.trapz(new_customer_survival_function * average_monthly_profit, dx=1)
print(f'The advanced CLV of the new customer is ${clv_advanced[0]}')
