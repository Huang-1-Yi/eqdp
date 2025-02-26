import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib.pyplot as plt


# 读取HDF5文件中的图像和点云数据
source_file_path = "/home/hy/equidiff/test/test_abs.hdf5"  # 一个.hdf5文件

# 设定文件夹路径
output_folder = "/home/hy/equidiff/test/test_output"
# 如果文件夹不存在，则创建文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
# 打开 HDF5 文件
with h5py.File(source_file_path, 'r') as f:
    # 获取数据集
    agentview_image_data = f['/data/demo_98/obs/agentview_image'][:]
    point_cloud_data = f['/data/demo_98/obs/point_cloud'][:]

    # 输出读取的形状
    print("agentview_image shape:", agentview_image_data.shape)
    print("point_cloud shape:", point_cloud_data.shape)

    # 保存图像和点云
    for i in range(min(5, agentview_image_data.shape[0])):  # 这里只保存前5张图像
        image = agentview_image_data[i]
        # 使用PIL保存图像为PNG格式
        image_path = os.path.join(output_folder, f"agentview_image_{i}.png")
        img = Image.fromarray(image)
        img.save(image_path)
        print(f"Saved image {i} to {image_path}")

        # 保存点云数据为Numpy格式
        point_cloud_path = os.path.join(output_folder, f"point_cloud_{i}.npy")
        np.save(point_cloud_path, point_cloud_data[i])
        print(f"Saved point cloud {i} to {point_cloud_path}")


