import math

import pandas as pd
from collections import deque
import numpy as np
import sys

from matplotlib import pyplot as plt

# Load CSV data
data = pd.read_csv(r'./src/附件2/B2.csv')
mean_class_2 = 35.333333
# with open(r'mean_class_2.txt', 'r', encoding='utf-8') as tmp:
#     mean_class_2 = float(tmp.readline())

# Ensure the data is sorted by vehicle ID and time
data.sort_values(['vehicle_id', 'time'], inplace=True)

# Calculate the speed for each vehicle
data['speed'] = np.sqrt(data.groupby('vehicle_id')['x'].diff()**2 + data.groupby('vehicle_id')['y'].diff()**2) / data.groupby('vehicle_id')['time'].diff()

# Fill NaN speed values with the first valid speed per vehicle
data['speed'] = data.groupby('vehicle_id')['speed'].transform(lambda x: x.fillna(x.dropna().iloc[0] if not x.dropna().empty else 0))

# 检查是否位于停车点
px = 11.83
py = 3.64
distance = math.sqrt((data['x'] - px) ** 2 + (data['y'] - py) ** 2)
data['meets_conditions'] = (distance <= 1.5) & (data['speed'] < 1)
#此处换为圆形半径判断是否位于停车点

# Create a list to store multiple queues
queues_list = []

# Track which vehicles have been added to any queue
added_vehicles = set()

# Track the latest queue for adding new braking vehicles
latest_queue = None

# Process each record in the condition_met_cars
for _, record in data.iterrows():
    vehicle_id = record['vehicle_id']
    if record['meets_conditions'] and vehicle_id not in added_vehicles:
        # This vehicle is near 11.4 and braking, create a new queue
        record_dict = record.to_dict()
        new_queue = deque([record_dict])
        queues_list.append(new_queue)
        latest_queue = new_queue
        added_vehicles.add(vehicle_id)
    elif record['speed'] < 1 and vehicle_id not in added_vehicles:
        # This vehicle is braking but not near 11.4, add to the latest queue
        if latest_queue is not None:
            latest_queue.append(record.to_dict())
            added_vehicles.add(vehicle_id)

# Function to merge queues based on time difference of their first items
def merge_queues(queues_list):
    merged = []
    while queues_list:
        current = queues_list.pop(0)
        i = 0
        while i < len(queues_list):
            if abs(queues_list[i][0]['time'] - current[0]['time']) < 2 * mean_class_2 :
                current.extend(queues_list.pop(i))
            else:
                i += 1
        merged.append(current)
    return merged

# Merge queues with first data time difference less than 70
merged_queues = merge_queues(queues_list)

# Print each merged queue
for index, queue in enumerate(merged_queues, start=1):
    print(f"Merged Queue {index}:")
    for record in queue:
        print(record)

start_times = [queue[0]['time'] for queue in merged_queues]

plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(start_times)), start_times, color='skyblue')
plt.xlabel('Queue Number')
plt.ylabel('Start Time')
plt.title('Start Times of Each Merged Queue')
plt.xticks(range(len(start_times)), [f"{i+1}" for i in range(len(start_times))])

# 在每个柱状图上添加文本显示其y坐标
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom')

# plt.savefig(r'./data/figures/')

# 假设已经加载和处理了数据，并且已经有了合并后的队列 merged_queues
# merged_queues = merge_queues(queues_list) # 这一行应该在实际环境中是有效的，前提是您已经定义并执行了merge_queues函数

# 新建一个列表来存储从第二个队列开始，每个队列首条记录与前一个队列首条记录的时间差
first_record_time_diffs = []

# 从第二个合并后的队列开始计算时间差
for i in range(1, len(merged_queues)):
    # 获取当前队列的第一条记录的时间
    current_queue_first_time = merged_queues[i][0]['time']
    # 获取前一个队列的第一条记录的时间
    previous_queue_first_time = merged_queues[i-1][0]['time']
    # 计算时间差
    time_diff = current_queue_first_time - previous_queue_first_time
    # 添加到时间差列表中
    first_record_time_diffs.append(time_diff)

# 输出每个适用队列的时间差
for index, d_time in enumerate(first_record_time_diffs, start=2):
    print(f"Time difference between the first record of Queue {index-1} and the first record of Queue {index}: {d_time}")

cnt = len(merged_queues)

# 遍历first_record_time_diffs列表，从第二个元素开始
for i in range(1, len(first_record_time_diffs)):
    if first_record_time_diffs[i - 1] < first_record_time_diffs[i] / 2:
        cnt += 1

print("Updated count:", cnt)
print(3600/cnt - mean_class_2)

# import pandas as pd
# from collections import deque
# import numpy as np
# import matplotlib.pyplot as plt
#
# mean_class_2 = 35.33333
# # with open(r'mean_class_2.txt', 'r', encoding='utf-8') as tmp:
# #     mean_class_2 = float(tmp.readline())
#
# # 加载CSV数据
# data = pd.read_csv(r'./src/附件2/B2.csv')
#
# # 确保数据按照车辆ID和时间排序
# data.sort_values(['vehicle_id', 'time'], inplace=True)
#
# # 计算每个车辆的速度
# data['speed'] = np.sqrt(data.groupby('vehicle_id')['x'].diff()**2 + data.groupby('vehicle_id')['y'].diff()**2) / data.groupby('vehicle_id')['time'].diff()
#
# # 用第一个有效的速度值填充NaN速度值
# data['speed'] = data.groupby('vehicle_id')['speed'].transform(lambda x: x.fillna(x.dropna().iloc[0] if not x.dropna().empty else 0))
#
# # 检查车辆是否在-11.4附近且速度小于1
#
# data['meets_conditions'] = (np.abs(data['x'] - 11.51)  + np.abs(data['y'] - 0.46) <= 1) & (data['speed'] < 1)
#
# # 存储多个队列
# queues_list = []
#
# # 记录已经添加到队列中的车辆
# added_vehicles = set()
#
# # 记录最后一个队列的创建时间
# last_queue_time = None
#
# # 处理每条记录
# for _, record in data.iterrows():
#     vehicle_id = record['vehicle_id']
#     current_time = record['time']
#
#     # 检查是否需要创建新队列
#     if record['meets_conditions'] and vehicle_id not in added_vehicles:
#         if last_queue_time is None or (current_time - last_queue_time > 2 * mean_class_2):
#             # 新建队列
#             new_queue = deque([record.to_dict()])
#             queues_list.append(new_queue)
#             last_queue_time = current_time  # 更新最后一个队列的时间
#             added_vehicles.add(vehicle_id)  # 标记车辆已加入队列
#     elif record['speed'] < 1 and vehicle_id not in added_vehicles:
#         # 速度小于1，同样检查是否需要新建队列
#         if last_queue_time is None or (current_time - last_queue_time > 2 * mean_class_2):
#             new_queue = deque([record.to_dict()])
#             queues_list.append(new_queue)
#             last_queue_time = current_time
#             added_vehicles.add(vehicle_id)
#
# # 合并队列的函数
# def merge_queues(queues_list):
#     merged = []
#     while queues_list:
#         current = queues_list.pop(0)
#         i = 0
#         while i < len(queues_list):
#             if abs(queues_list[i][0]['time'] - current[0]['time']) < 2 * mean_class_2:
#                 current.extend(queues_list.pop(i))
#             else:
#                 i += 1
#         merged.append(current)
#     return merged
#
# # 合并队列
# merged_queues = merge_queues(queues_list)
#
# # 打印合并后的每个队列
# for index, queue in enumerate(merged_queues, start=1):
#     print(f"Merged Queue {index}:")
#     for record in queue:
#         print(record)
#
# start_times = [queue[0]['time'] for queue in merged_queues]
#
# plt.figure(figsize=(10, 6))
# bars = plt.bar(range(len(start_times)), start_times, color='skyblue')
# plt.xlabel('Queue Number')
# plt.ylabel('Start Time')
# plt.title('Start Times of Each Merged Queue')
# plt.xticks(range(len(start_times)), [f"{i+1}" for i in range(len(start_times))])
#
# # 在每个柱状图上添加文本显示其y坐标
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom')
#
# # plt.savefig(r'./data/figures/')
#
# # 假设已经加载和处理了数据，并且已经有了合并后的队列 merged_queues
# # merged_queues = merge_queues(queues_list) # 这一行应该在实际环境中是有效的，前提是您已经定义并执行了merge_queues函数
#
# # 新建一个列表来存储从第二个队列开始，每个队列首条记录与前一个队列首条记录的时间差
# first_record_time_diffs = []
#
# # 从第二个合并后的队列开始计算时间差
# for i in range(1, len(merged_queues)):
#     # 获取当前队列的第一条记录的时间
#     current_queue_first_time = merged_queues[i][0]['time']
#     # 获取前一个队列的第一条记录的时间
#     previous_queue_first_time = merged_queues[i-1][0]['time']
#     # 计算时间差
#     time_diff = current_queue_first_time - previous_queue_first_time
#     # 添加到时间差列表中
#     first_record_time_diffs.append(time_diff)
#
# # 输出每个适用队列的时间差
# for index, d_time in enumerate(first_record_time_diffs, start=2):
#     print(f"Time difference between the first record of Queue {index-1} and the first record of Queue {index}: {d_time}")
#
# cnt = len(merged_queues)
#
# # 遍历first_record_time_diffs列表，从第二个元素开始
# for i in range(1, len(first_record_time_diffs)):
#     if first_record_time_diffs[i - 1] < first_record_time_diffs[i] / 2:
#         cnt += 1
#
# print("Updated count:", cnt)
# print(3600/cnt - mean_class_2)