import numpy as np
from scipy import stats
# Revising the Python example to focus on the relationship t^2 = F in a two-sample t-test scenario

def t_test_as_f_test(data1, data2):
    """
    Perform a two-sample t-test and demonstrate how it relates to an F-test in this specific scenario.
    
    Parameters:
    - data1: array-like, the first data set
    - data2: array-like, the second data set
    
    Returns:
    - t_stat: float, the t-statistic
    - F_stat_from_t: float, the F-statistic derived from t
    - F_stat: float, the F-statistic
    - are_statistics_close: boolean, whether t^2 is close to F
    """
    # Perform the two-sample t-test
    t_stat, _ = stats.ttest_ind(data1, data2)
    
    # Calculate the F-statistic from the t-statistic
    F_stat_from_t = t_stat ** 2
    
    # Calculate the F-statistic using one-way ANOVA
    F_stat, _ = stats.f_oneway(data1, data2)
    
    # Check if t^2 is close to F
    are_statistics_close = np.isclose(F_stat_from_t, F_stat)
    
    return t_stat, F_stat_from_t, F_stat, are_statistics_close

# Test data sets
data1 = np.random.normal(loc=20, scale=5, size=50)
data2 = np.random.normal(loc=22, scale=5, size=50)

# Perform the tests and check the relationship
t_stat, F_stat_from_t, F_stat, are_statistics_close = t_test_as_f_test(data1, data2)

print(t_stat, F_stat_from_t, F_stat, are_statistics_close)
