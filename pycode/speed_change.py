# #
# # 当速度首次降至0.1时记录为 down_time，
# # 从 down_time 向左遍历直至找到速度首次超过10的时间点记为 start_down_time，
# # 从 down_time 向右遍历直到速度首次大于2的时间点记为 speed_up_time。
#
# import pandas as pd
# import numpy as np
#
# # Load CSV data
# data = pd.read_csv(r'./src/附件3/C1.csv')
#
# # Ensure the data is sorted by vehicle ID and time
# data.sort_values(['vehicle_id', 'time'], inplace=True)
#
# # Calculate the speed for each vehicle
# data['speed'] = np.sqrt(data.groupby('vehicle_id')['x'].diff()**2 + data.groupby('vehicle_id')['y'].diff()**2) / data.groupby('vehicle_id')['time'].diff()
#
# # Fill NaN speed values with the first valid speed per vehicle
# data['speed'] = data.groupby('vehicle_id')['speed'].transform(lambda x: x.fillna(x.dropna().iloc[0] if not x.dropna().empty else 0))
#
# # Initialize a dictionary to hold the vehicle data
# vehicles_data = {}
#
# # List to hold dt values greater than 50
# dts_over_50 = []
#
# # Process each vehicle
# for vehicle_id, group in data.groupby('vehicle_id'):
#     speeds = group['speed'].tolist()
#     times = group['time'].tolist()
#
#     # Find down_time, start_down_time and speed_up_time
#     down_time = None
#     start_down_time = None
#     speed_up_time = None
#
#     # Find the first instance where speed drops to 0.1 or below
#     for i in range(len(speeds)):
#         if speeds[i] <= 0.1:
#             down_time = times[i]
#             # Look left from down_time until speed exceeds 10
#             for j in range(i, -1, -1):
#                 if speeds[j] > 10:
#                     start_down_time = times[j]
#                     break
#             # Look right from down_time until speed exceeds 2
#             for k in range(i, len(speeds)):
#                 if speeds[k] > 2:
#                     speed_up_time = times[k]
#                     break
#             break  # Stop after the first valid down_time is found
#
#     # Only add vehicles with a valid down_time
#     if down_time is not None:
#         dt = None
#         if start_down_time is not None and speed_up_time is not None:
#             dt = speed_up_time - start_down_time  # Calculate dt
#             if dt > 50:
#                 dts_over_50.append(dt)  # Add to the list if dt is greater than 50
#
#         vehicles_data[vehicle_id] = {
#             'speeds': speeds,
#             'down_time': down_time,
#             'start_down_time': start_down_time,
#             'speed_up_time': speed_up_time,
#             'dt': dt  # Add dt to the dictionary
#         }
#
# # Calculate the mean of dt values over 50
# if dts_over_50:
#     mean_dt_over_50 = np.mean(dts_over_50) - 3
#     print(f"Mean of dt values over 50: {mean_dt_over_50}")
# else:
#     print("No dt values over 50.")
#
#
# import pandas as pd
# import numpy as np
#
# # 加载数据
# data = pd.read_csv(r'./src/附件2/B3.csv')
#
# # 确保按车辆ID和时间排序
# data.sort_values(['vehicle_id', 'time'], inplace=True)
#
# # 计算每个车辆的速度
# data['speed'] = np.sqrt(data.groupby('vehicle_id')['x'].diff()**2 + data.groupby('vehicle_id')['y'].diff()**2) / data.groupby('vehicle_id')['time'].diff()
#
# # 用第一个有效的速度填充NaN值
# data['speed'] = data.groupby('vehicle_id')['speed'].transform(lambda x: x.fillna(x.dropna().iloc[0] if not x.dropna().empty else 0))
#
# # 收集dt值
# dts = []
#
# # 处理每个车辆
# for vehicle_id, group in data.groupby('vehicle_id'):
#     speeds = group['speed'].tolist()
#     times = group['time'].tolist()
#     down_time = None
#     start_down_time = None
#     speed_up_time = None
#
#     # 寻找速度首次降到0.1以下的时间点
#     for i in range(len(speeds)):
#         if speeds[i] <= 0.1:
#             down_time = times[i]
#             for j in range(i, -1, -1):
#                 if speeds[j] > 10:
#                     start_down_time = times[j]
#                     break
#             for k in range(i, len(speeds)):
#                 if speeds[k] > 2:
#                     speed_up_time = times[k]
#                     break
#             break
#
#     # 如果有有效的down_time
#     if down_time is not None and start_down_time is not None and speed_up_time is not None:
#         dt = speed_up_time - start_down_time
#         if dt > 0:  # 仅收集正的dt值
#             dts.append(dt)
#
# # 计算三个类的分界点
# if dts:
#     dts = np.array(dts)
#     first_quantile = np.percentile(dts, 33)
#     second_quantile = np.percentile(dts, 66)
#
#     # 分类
#     class_1 = dts[dts <= first_quantile]
#     class_2 = dts[(dts > first_quantile) & (dts <= second_quantile)]
#     class_3 = dts[dts > second_quantile]
#
#     print("Class 1 (Low dt):", class_1)
#     print("Class 2 (Medium dt):", class_2)
#     print("Class 3 (High dt):", class_3)
# else:
#     print("No valid dt values available for clustering.")
#
# mean_class_2 = np.mean(class_2) - 3
# with open('mean_class_2.txt', 'w', encoding='utf-8') as t:
#     t.write(str(mean_class_2))
# print("Mean of Class 2 :", mean_class_2)


import pandas as pd
import numpy as np

# 加载数据
data = pd.read_csv(r'./src/附件1/A1.csv')

# 确保按车辆ID和时间排序
data.sort_values(['vehicle_id', 'time'], inplace=True)

# 计算每个车辆的速度
data['speed'] = np.sqrt(data.groupby('vehicle_id')['x'].diff()**2 + data.groupby('vehicle_id')['y'].diff()**2) / data.groupby('vehicle_id')['time'].diff()

# 用第一个有效的速度填充NaN值
data['speed'] = data.groupby('vehicle_id')['speed'].transform(lambda x: x.fillna(x.dropna().iloc[0] if not x.dropna().empty else 0))

# 收集dt值
dts = []

# 处理每个车辆
for vehicle_id, group in data.groupby('vehicle_id'):
    speeds = group['speed'].tolist()
    times = group['time'].tolist()
    down_time = None
    start_down_time = None
    speed_up_time = None

    # 寻找速度首次降到0.1以下的时间点
    for i in range(len(speeds)):
        if speeds[i] <= 0.1:
            down_time = times[i]
            # 寻找开始下降的时间点
            for j in range(i, -1, -1):
                if speeds[j] > 10:
                    start_down_time = times[j]
                    break
            # 寻找加速的时间点，确保下一个时刻的速度不小于当前时刻的速度
            for k in range(i, len(speeds) - 1):
                if speeds[k + 1] >= speeds[k]:
                    speed_up_time = times[k]
                    break
            break

    # 如果有有效的down_time
    if down_time is not None and start_down_time is not None and speed_up_time is not None:
        dt = speed_up_time - start_down_time
        if dt > 0:  # 仅收集正的dt值
            dts.append(dt)

# 计算三个类的分界点
if dts:
    dts = np.array(dts)
    first_quantile = np.percentile(dts, 33)
    second_quantile = np.percentile(dts, 66)

    # 分类
    class_1 = dts[dts <= first_quantile]
    class_2 = dts[(dts > first_quantile) & (dts <= second_quantile)]
    class_3 = dts[dts > second_quantile]

    print("Class 1 (Low dt):", class_1)
    print("Class 2 (Medium dt):", class_2)
    print("Class 3 (High dt):", class_3)
else:
    print("No valid dt values available for clustering.")

mean_class_2 = np.mean(class_2) - 3
with open('mean_class_2.txt', 'w', encoding='utf-8') as t:
    t.write(str(mean_class_2))
print("Mean of Class 2 :", mean_class_2)