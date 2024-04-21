# import pandas as pd
# import numpy as np
# from scipy.stats import linregress
#
# # 加载CSV数据
# data = pd.read_csv(r'./src/附件2/B2.csv')
#
# # 检查数据的列，我们假设数据中包含x和y坐标
# print(data.head())
#
# # 计算斜率，假设数据集被分成了两部分，每部分代表一条直线
# # 如果数据没有标签区分两条直线，可能需要先通过某种方式分割数据
# # 这里我们先整体估计一个斜率
# slope, intercept, r_value, p_value, std_err = linregress(data['x'], data['y'])
#
# print(f"Estimated slope for the line: {slope}")
# print(f"Intercept: {intercept}")
# print(f"R-squared: {r_value**2}")
#
# # 如果需要分别对两条直线进行估计，需要先对数据进行分割
# # 这里假设我们已经知道怎么分割数据或者所有点都在一条直线上

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# 加载CSV数据
data = pd.read_csv(r'./src/附件2/B3.csv')

# 可视化数据，查看是否可以明显分为两组
plt.scatter(data['x'], data['y'])
plt.title('Scatter Plot of Road Data')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

# 假设数据可以明显分为两组，应用K-means聚类
kmeans = KMeans(n_clusters=2, random_state=0).fit(data[['x', 'y']])
data['cluster'] = kmeans.labels_

# 分别计算每组的斜率
slopes = []
for cluster in sorted(data['cluster'].unique()):
    cluster_data = data[data['cluster'] == cluster]
    linreg = LinearRegression()
    linreg.fit(cluster_data[['x']], cluster_data['y'])
    slope = linreg.coef_[0]
    slopes.append(slope)
    print(f"Slope for cluster {cluster}: {slope}")

    # 绘制聚类结果与拟合线
    plt.scatter(cluster_data['x'], cluster_data['y'], label=f'Cluster {cluster}')
    plt.plot(cluster_data['x'], linreg.predict(cluster_data[['x']]), color='red')  # 绘制拟合线

plt.title('Clustered Data with Regression Lines')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.show()