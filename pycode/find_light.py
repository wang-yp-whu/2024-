import pandas as pd

# Load the data from the SV file
file_path = r'./src/附件3/C1.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
# data.head()

# Count the frequency of each 'x' value and sort them in descending order
x_counts = data['x'].value_counts().sort_values(ascending=False)
y_counts = data['y'].value_counts().sort_values(ascending=False)
print(x_counts.head())
print()
print(y_counts.head())