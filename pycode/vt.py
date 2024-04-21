import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv(r'./src/附件2/B5.csv')

# 确保数据按车辆ID和时间排序
data.sort_values(['vehicle_id', 'time'], inplace=True)

# 计算每辆车的速度，如果数据中没有速度信息
if 'speed' not in data.columns:
    # 计算两点间的距离除以时间差
    data['speed'] = data.groupby('vehicle_id').apply(
        lambda df: np.sqrt(df['x'].diff()**2 + df['y'].diff()**2) / df['time'].diff()
    ).reset_index(level=0, drop=True)
    # 填充由于计算产生的NaN值
    data['speed'].fillna(method='bfill', inplace=True)  # 用下一个有效值填充

# 获取所有车辆ID
vehicle_ids = data['vehicle_id'].unique()

# 为每辆车绘制速度时间图
for vehicle_id in vehicle_ids:
    vehicle_data = data[data['vehicle_id'] == vehicle_id]
    plt.figure(figsize=(10, 6))
    plt.plot(vehicle_data['time'], vehicle_data['speed'], marker='o', linestyle='-', label='Speed')
    plt.title(f'Speed-Time Graph for Vehicle {vehicle_id}')
    plt.xlabel('Time')
    plt.ylabel('Speed')
    plt.grid(True)
    plt.legend()
    plt.show()
