import pandas as pd
import os
from matplotlib import pyplot as plt

# Load the CSV data into a DataFrame
file_path = r'./src/附件1/A1.csv'
data = pd.read_csv(file_path)
figures = r'./data/figures/question1/'
# Display the first few rows of the DataFrame and its structure to understand the data better
data.head(), data.dtypes

# Group data by 'vehicle_id' and sort by 'time'
grouped_data = data.sort_values(by=['vehicle_id', 'time']).groupby('vehicle_id')

# for vehicle_id, group in grouped_data:
#     print(f"Vehicle ID: {vehicle_id}")
#     print(group)
#     print('-------------------------------------------------')

# Calculate speeds using the specified formula and store them in a dictionary
vehicle_data = {}
for vehicle_id, group in grouped_data:
    if len(group) > 2:  # Ensure there are at least three points to calculate middle speeds
        # Calculate speeds: (X_{i+1} - X_{i-1}) / 2 for each inner point
        speeds = (abs(group['x'].iloc[2:].values - group['x'].iloc[:-2].values)) / 2
        min_time = group['time'].min()
        max_time = group['time'].max()
        vehicle_data[vehicle_id] = {
            'min_time': min_time,
            'max_time': max_time,
            'speeds': speeds.tolist()
        }

# Display the calculated speeds for each vehicle
for car in vehicle_data:
    print(f"Vehicle ID: {car}")
    print(f"Min Time: {vehicle_data[car]['min_time']}")
    print(f"Max Time: {vehicle_data[car]['max_time']}")
    print(f"Speeds: {vehicle_data[car]['speeds']}")
    print('-------------------------------------------------')

# for vehicle_id, vehicle in vehicle_data.items():
#     times = list(range(vehicle['min_time'] + 1, vehicle['max_time']))
#     speeds = vehicle['speeds']
#
#     plt.figure(figsize=(10, 5))
#     plt.plot(times, speeds, marker='o', linestyle='-', color='b')
#     plt.title(f'Speed Variation for Vehicle {vehicle_id}')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Speed')
#     plt.grid(True)
#
#     # Save plot to file
#     filename = os.path.join(figures, f'vehicle_{vehicle_id}_speed_variation.png')
#     plt.savefig(filename)
#     plt.close()  # Close the plot to free up memory


