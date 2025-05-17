import zarr
from dp3.visualizer.visualizer.pointcloud import visualize_pointcloud  # 确保模块已安装

# 正确路径
zarr_path = '/home/robot/key/dp_learning/3D-Diffusion-Policy/dp3_real_robot_demo/real_robot_demo/drill_40demo_1024.zarr'

# 尝试加载根目录
try:
    root = zarr.open(zarr_path, mode='r')
except Exception as e:
    raise RuntimeError(f"无法加载Zarr文件: {e}")

# 打印根目录结构
print("根目录组:", list(root.group_keys()))
print("根目录数组:", list(root.array_keys()))

# 假设point_cloud在根目录下
if 'point_cloud' in root.array_keys():
    point_array = root['point_cloud']
# 或在data组下
elif 'data' in root.group_keys():
    data_group = root['data']
    point_array = data_group['point_cloud']
else:
    raise KeyError("未找到point_cloud数组")

# 验证形状
print("点云形状:", point_array.shape)

# 可视化第一个样本
visualize_pointcloud(point_array[0])