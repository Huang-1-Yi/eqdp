import zarr
import numpy as np
from visualizer import Visualizer
import matplotlib.pyplot as plt

# 定义点云路径
pointcloud_path = r"C:\Users\ZH\Desktop\robot_learning\源代码\robot_zh\robot_zh\3D-Diffusion-Policy\3D-Diffusion-Policy\data\adroit_hammer_expert.zarr"
#pointcloud_path=r"C:\Users\ZH\Desktop\robot_learning\源代码\robot_zh\robot_zh\3D-Diffusion-Policy\3D-Diffusion-Policy\data\pour_40demo_1024.zarr"
# 打开 Zarr 文件
pointcloud_data = zarr.open(pointcloud_path, mode='r')
print(pointcloud_data.tree())

# 提取图像数据
if 'data' in pointcloud_data and 'img' in pointcloud_data['data']:
    img_data = np.array(pointcloud_data['data']['img'])
    print("Image shape:", img_data.shape)
else:
    raise ValueError("Image data not found in the provided Zarr file.")

# 可视化图像
# 假设你想显示第一个图像样本
plt.imshow(img_data[0])  # 显示第一个样本的图像
plt.axis('off')  # 关闭坐标轴显示
plt.show()

# 提取点云数据
if 'data' in pointcloud_data and 'point_cloud' in pointcloud_data['data']:
    pointcloud = np.array(pointcloud_data['data']['point_cloud'])
    print("Pointcloud shape:", pointcloud.shape)
else:
    raise ValueError("Pointcloud data not found in the provided Zarr file.")
print(pointcloud[1])
# 可视化点云
vis = Visualizer()
vis.visualize_pointcloud(pointcloud[1])




