import pandas as pd
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# Load CSV data
data = pd.read_csv(r'./src/附件2/B3.csv')

mean_class_2 = 0
with open(r'mean_class_2.txt', 'r', encoding='utf-8') as tmp:
    mean_class_2 = float(tmp.readline())
# Ensure the data is sorted by vehicle ID and time
data.sort_values(['vehicle_id', 'time'], inplace=True)

# Calculate the speed for each vehicle
data['speed'] = np.sqrt(data.groupby('vehicle_id')['x'].diff()**2 + data.groupby('vehicle_id')['y'].diff()**2) / data.groupby('vehicle_id')['time'].diff()

# Fill NaN speed values with the first valid speed value per vehicle
data['speed'] = data.groupby('vehicle_id')['speed'].transform(lambda x: x.fillna(x.dropna().iloc[0] if not x.dropna().empty else 0))

# Check if vehicles are near -11.4 and have a speed less than 1
data['meets_conditions'] = (np.abs(data['x'] + 11.4) <= 1.5) & (data['speed'] < 1)

# Store multiple queues
queues_list = []
added_vehicles = set()
# Track the last queue creation time
last_queue_time = None

# Process each record
for _, record in data.iterrows():
    vehicle_id = record['vehicle_id']
    current_time = record['time']

    # Create a new queue or add to the latest queue based on conditions
    if record['meets_conditions'] and vehicle_id not in added_vehicles or record['speed'] < 1 and (last_queue_time is None or (current_time - last_queue_time > 2 * mean_class_2)):
        if last_queue_time is None or (current_time - last_queue_time > 2 * mean_class_2):
            # Create a new queue
            new_queue = deque([record.to_dict()])
            queues_list.append(new_queue)
            last_queue_time = current_time  # Update the time of the last created queue
            added_vehicles.add(vehicle_id)  # Mark vehicle as added
        else:
            # Add to the latest queue
            queues_list[-1].append(record.to_dict())
            added_vehicles.add(vehicle_id)

# Function to merge queues based on time difference of their first items
def merge_queues(queues_list):
    merged = []
    while queues_list:
        current = queues_list.pop(0)
        i = 0
        while i < len(queues_list):
            if abs(queues_list[i][0]['time'] - current[0]['time']) < 70:
                current.extend(queues_list.pop(i))
            else:
                i += 1
        merged.append(current)
    return merged

# Merge queues
merged_queues = merge_queues(queues_list)

# Print each merged queue
for index, queue in enumerate(merged_queues, start=1):
    print(f"Merged Queue {index}:")
    for record in queue:
        print(record)
