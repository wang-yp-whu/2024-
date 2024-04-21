import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv(r'./src/附件3/C1.csv')

# 确保数据按照车辆ID和时间排序
data.sort_values(['vehicle_id', 'time'], inplace=True)

# 获取所有车辆ID
vehicle_ids = data['vehicle_id'].unique()

# 为每个车辆绘制行驶轨迹，仅包括x坐标大于-50的部分
for vehicle_id in vehicle_ids:
    # 获取该车辆的数据并过滤
    vehicle_data = data[(data['vehicle_id'] == vehicle_id) & (data['x'] > -20)]

    # 如果过滤后没有数据，跳过绘图
    if vehicle_data.empty:
        continue

    plt.figure(figsize=(10, 6))
    plt.plot(vehicle_data['x'], vehicle_data['y'], marker='o', linestyle='-', label=f'Vehicle {vehicle_id}')
    plt.title(f'Trajectory for Vehicle {vehicle_id} (X > -50)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()
