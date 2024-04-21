
import pandas as pd
import numpy as np

# 加载数据
data = pd.read_csv(r'./src/附件2/B1.csv')

# 确保数据按车辆ID和时间排序
data.sort_values(['vehicle_id', 'time'], inplace=True)

# 计算速度
data['speed'] = np.sqrt(data.groupby('vehicle_id')['x'].diff()**2 + data.groupby('vehicle_id')['y'].diff()**2) / data.groupby('vehicle_id')['time'].diff()
# 用第一个有效的速度值填充NaN值
data['speed'] = data.groupby('vehicle_id')['speed'].transform(lambda x: x.fillna(method='bfill'))

# 初始化列表存储时间差和数据比例
time_diffs = []
data_proportions = []

# 总记录数
total_records = len(data)

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
            # 向前找到速度首次超过10的时间点
            for j in range(i, -1, -1):
                if speeds[j] > 10:
                    start_down_time = times[j]
                    break
            # 向后找到速度首次超过2的时间点
            for k in range(i, len(speeds)):
                if speeds[k] > 2:
                    speed_up_time = times[k]
                    break
            break  # 只考虑第一次下降到0.1以下的情况

    # 计算时间差并存入列表
    if start_down_time is not None and speed_up_time is not None:
        dt = speed_up_time - start_down_time
        time_diffs.append(dt)

    # 计算数据比例并存入列表
    proportion = (len(group) / total_records) * 100
    data_proportions.append(proportion)

# 输出结果
print("Time differences between speed_up_time and start_down_time:")
print(time_diffs)
print("Data proportions for each vehicle:")
print(data_proportions)
# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
#
# # 加载数据
# data = pd.read_csv(r'./src/附件2/B1.csv')
#
# # 确保按车辆ID和时间排序
# data.sort_values(['vehicle_id', 'time'], inplace=True)
#
# # 计算每个车辆的速度
# data['speed'] = data.groupby('vehicle_id').apply(
#     lambda group: np.sqrt(group['x'].diff()**2 + group['y'].diff()**2) / group['time'].diff()
# ).reset_index(level=0, drop=True)  # 重置索引以避免插入错误
#
# # 用前向填充处理NaN值，因为我们只能基于已知的数据进行填充
# data['speed'].fillna(method='ffill', inplace=True)
#
# # 初始化列表存储时间差和数据比例
# time_diffs = []
# data_proportions = []
#
# # 总记录数
# total_records = len(data)
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
#             # 向前找到速度首次超过10的时间点
#             for j in range(i, -1, -1):
#                 if speeds[j] > 10:
#                     start_down_time = times[j]
#                     break
#             # 向后找到速度首次超过2的时间点
#             for k in range(i, len(speeds)):
#                 if speeds[k] > 2:
#                     speed_up_time = times[k]
#                     break
#             break
#
#     if start_down_time is not None and speed_up_time is not None:
#         dt = speed_up_time - start_down_time
#         if dt > 0:
#             time_diffs.append(dt)
#             proportion = (len(group) / total_records) * 100
#             data_proportions.append(proportion)
#
# # 聚类分析
# time_diffs_np = np.array(time_diffs).reshape(-1, 1)
# data_proportions_np = np.array(data_proportions)
#
# kmeans = KMeans(n_clusters=3, random_state=42)
# labels = kmeans.fit_predict(time_diffs_np)
#
# # 计算每类的平均数据占比
# class_proportions = [data_proportions_np[labels == i].mean() for i in range(3)]
#
# # 找出平均数据占比最大的类别
# max_prop_index = np.argmax(class_proportions)
#
# # 计算该类别的time_diffs的平均值
# mean_time_diff = time_diffs_np[labels == max_prop_index].mean()
#
# print("Average time difference for the class with the highest average data proportion:", mean_time_diff)
#
import pandas as pd


def calculate_vehicle_data_proportions(csv_file_path):
    # 加载数据
    data = pd.read_csv(csv_file_path)

    # 计算总记录数
    total_records = len(data)

    # 按车辆ID分组，计算每组的记录数
    vehicle_counts = data['vehicle_id'].value_counts()

    # 计算每个车辆的数据占比
    proportions = (vehicle_counts / total_records) * 100

    # 打印每辆车的数据占比
    print("Data Proportions by Vehicle:")
    for vehicle_id, proportion in proportions.items():
        print(f"Vehicle {vehicle_id}: {proportion:.2f}%")


# 调用函数，传递CSV文件的路径
calculate_vehicle_data_proportions(r'./src/附件2/B4.csv')
