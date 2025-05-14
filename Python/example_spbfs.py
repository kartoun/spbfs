from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('C:/Temp/spbfs/') #Path to "spbfs.py"
from spbfs import get_selected_features

# Load Pima Indians Diabetes dataset
data = fetch_openml('diabetes', version=1, as_frame=True)
df = data.frame

# Features and outcome names
feature_names = ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']
outcome_var = 'class'

# Convert outcome to binary
df[outcome_var] = LabelEncoder().fit_transform(df[outcome_var])

results = get_selected_features(
    data=df,
    feature_names=feature_names,
    outcome_var=outcome_var,
    num_iterations=100,
    num_random_variables_for_matching=3,
    final_selection_threshold=0.0,
    caliper_value=0.1,
    m_value=1,
    p_value_threshold=0.001,
    verbose=1
)

print(results)

# Plot
import matplotlib.pyplot as plt

results_sorted = results.sort_values(by='Frequency')
plt.barh(results_sorted['Feature_Name'], results_sorted['Frequency'])
plt.xlabel('Importance')
plt.title('SPBFS Feature Importance')
plt.tight_layout()
plt.show()
