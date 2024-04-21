import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv(r'./src/附件3/C6.csv')

# 确保数据按照车辆ID和时间排序
data.sort_values(['vehicle_id', 'time'], inplace=True)

# 计算速度，假设数据中已经包含了x和y的坐标
data['speed'] = np.sqrt(data.groupby('vehicle_id')['x'].diff()**2 + data.groupby('vehicle_id')['y'].diff()**2) / data.groupby('vehicle_id')['time'].diff()
data['speed'].fillna(method='ffill', inplace=True)  # 前向填充处理NaN速度值

# 初始化列表存储start_down_time和dt
start_down_times = []
dts = []

# 遍历每辆车
for vehicle_id, group in data.groupby('vehicle_id'):
    speeds = group['speed'].tolist()
    times = group['time'].tolist()
    down_time = None
    start_down_time = None
    speed_up_time = None

    # 找到减速到0.1以下的第一次时间
    for i in range(len(speeds)):
        if speeds[i] <= 0.1:
            down_time = times[i]
            # 向前寻找速度超过10的时间
            for j in range(i, -1, -1):
                if speeds[j] > 10:
                    start_down_time = times[j]
                    break
            # 向后寻找速度超过2的时间
            for k in range(i, len(speeds)):
                if speeds[k] > 2:
                    speed_up_time = times[k]
                    break
            break

    if start_down_time is not None and speed_up_time is not None:
        dt = speed_up_time - start_down_time
        if dt > 0:
            start_down_times.append(start_down_time)
            dts.append(dt)

# 聚类分析，首先将数据标准化
X = np.array(list(zip(start_down_times, dts)))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用肘部方法确定聚类数量
inertia = []
range_values = range(1, 10)
for n in range_values:
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range_values, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# 选择一个合理的聚类数，这里假设通过观察肘部图后选择了
optimal_clusters = 5 # 假设通过观察选择了4作为最佳聚类数

# 根据选择的聚类数量重新进行聚类
kmeans_final = KMeans(n_clusters=optimal_clusters, random_state=42)
labels = kmeans_final.fit_predict(X_scaled)

# 计算每类的平均dt
cluster_averages = []
for i in range(optimal_clusters):
    cluster_dt_values = [dts[j] for j in range(len(dts)) if labels[j] == i]
    average_dt = np.mean(cluster_dt_values)
    cluster_averages.append(average_dt)
    print(f"Average dt for cluster {i}: {average_dt}")

# 可选：打印所有聚类的平均值列表
print("All cluster averages:", cluster_averages)

# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
#
# # 加载数据
# data = pd.read_csv(r'./src/附件3/C6.csv')
#
# # 确保按车辆ID和时间排序
# data.sort_values(['vehicle_id', 'time'], inplace=True)
#
# # 计算速度
# data['speed'] = np.sqrt(data.groupby('vehicle_id')['x'].diff()**2 + data.groupby('vehicle_id')['y'].diff()**2) / data.groupby('vehicle_id')['time'].diff()
# data['speed'].fillna(method='ffill', inplace=True)  # 前向填充处理NaN速度值
#
# # 初始化列表存储start_down_time和dt
# start_down_times = []
# dts = []
#
# # 遍历每辆车
# for vehicle_id, group in data.groupby('vehicle_id'):
#     speeds = group['speed'].tolist()
#     times = group['time'].tolist()
#     down_time = None
#     start_down_time = None
#     speed_up_time = None
#
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
#     if start_down_time is not None and speed_up_time is not None:
#         dt = speed_up_time - start_down_time
#         if dt > 0:
#             start_down_times.append(start_down_time)
#             dts.append(dt)
#
# # 聚类分析，首先将数据标准化
# X = np.array(list(zip(start_down_times, dts)))
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # 使用肘部方法确定聚类数量
# inertia = []
# range_values = range(1, 10)
# for n in range_values:
#     kmeans = KMeans(n_clusters=n, random_state=42)
#     kmeans.fit(X_scaled)
#     inertia.append(kmeans.inertia_)
#
# # 绘制肘部图
# plt.figure(figsize=(8, 4))
# plt.plot(range_values, inertia, 'bo-')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Inertia')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()
#
# # 选择一个合理的聚类数
# optimal_clusters = 5  # 假设通过观察选择了4作为最佳聚类数
#
# # 根据选择的聚类数量重新进行聚类
# kmeans_final = KMeans(n_clusters=optimal_clusters, random_state=42)
# labels = kmeans_final.fit_predict(X_scaled)
#
# # 输出每个聚类中的dt值和start_down_time
# for i in range(optimal_clusters):
#     print(f"\nCluster {i}:")
#     cluster_indices = np.where(labels == i)[0]
#     for index in cluster_indices:
#         print(f"  dt: {dts[index]}, start_down_time: {start_down_times[index]}")

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv(r'./src/附件3/C6.csv')

# 确保按车辆ID和时间排序
data.sort_values(['vehicle_id', 'time'], inplace=True)

# 计算速度
data['speed'] = np.sqrt(data.groupby('vehicle_id')['x'].diff()**2 + data.groupby('vehicle_id')['y'].diff()**2) / data.groupby('vehicle_id')['time'].diff()
data['speed'].fillna(method='ffill', inplace=True)  # 前向填充处理NaN速度值

# 初始化列表存储start_down_time和dt
start_down_times = []
dts = []

# 遍历每辆车
for vehicle_id, group in data.groupby('vehicle_id'):
    speeds = group['speed'].tolist()
    times = group['time'].tolist()
    down_time = None
    start_down_time = None
    speed_up_time = None

    for i in range(len(speeds)):
        if speeds[i] <= 0.1:
            down_time = times[i]
            for j in range(i, -1, -1):
                if speeds[j] > 10:
                    start_down_time = times[j]
                    break
            for k in range(i, len(speeds)):
                if speeds[k] > 2:
                    speed_up_time = times[k]
                    break
            break

    if start_down_time is not None and speed_up_time is not None:
        dt = speed_up_time - start_down_time
        if dt > 0:
            start_down_times.append(start_down_time)
            dts.append(dt)

# 聚类分析，首先将数据标准化
X = np.array(list(zip(start_down_times, dts)))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用肘部方法确定聚类数量
inertia = []
range_values = range(1, 10)
for n in range_values:
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# 绘制肘部图
plt.figure(figsize=(8, 4))
plt.plot(range_values, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# 选择一个合理的聚类数
optimal_clusters = 5 # 假设通过观察选择了4作为最佳聚类数

# 根据选择的聚类数量重新进行聚类
kmeans_final = KMeans(n_clusters=optimal_clusters, random_state=42)
labels = kmeans_final.fit_predict(X_scaled)

# 输出每个聚类的dt和start_down_time信息以及统计
for i in range(optimal_clusters):
    cluster_indices = np.where(labels == i)[0]
    print(f"\nCluster {i}, Count: {len(cluster_indices)}")

    # 获取每个聚类的start_down_times
    cluster_start_down_times = [start_down_times[index] for index in cluster_indices]
    cluster_start_down_times.sort()  # 对时间进行排序

    if len(cluster_start_down_times) > 0:
        time_diff = cluster_start_down_times[-1] - cluster_start_down_times[0]
        print(f"Time difference between the last and first start_down_time in cluster {i}: {time_diff}")
    else:
        print("No data available in this cluster to calculate time difference.")
