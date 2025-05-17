# 导入必要的库
import zarr  # 用于处理Zarr格式的存储库
import pandas as pd  # 用于数据处理和分析

# 定义Zarr文件的路径
# file_path = '/home/robot/key/dp_0314_v2/data/test_032317/replay_buffer.zarr'
file_path = 'F:\\Improved-3D-Diffusion-Policy0406\\test_data\\replay_buffer.zarr'

# 以只读模式打开Zarr文件
zarr_file = zarr.open(file_path, mode='r')

# 获取Zarr文件中的第一个组(group)的键名
group_key = list(zarr_file.group_keys())[0]  

# 根据组键名获取对应的组对象
group = zarr_file[group_key]  

# array_keys()返回该组下所有数组的名称列表
print("Array keys in group '%s': %s" % (group_key, list(group.array_keys())))
# 示例输出: ['action', 'robot_eef_pose', 'robot_eef_pose_vel', 'robot_joint', 'robot_joint_vel', 'stage', 'timestamp']

# ---------- 第一部分：处理机械臂末端位姿数据 ----------
# 获取组中第一个数组的键名
array_key = list(group.array_keys())[0]  # 索引0对应第一个数组（robot_eef_pose）

robot_eef_pose = group[array_key][:]  # 获取多维数组数据

# 将numpy数组转换为Pandas DataFrame便于查看和分析
df_robot_eef_pose = pd.DataFrame(robot_eef_pose)

# 打印DataFrame的前5行数据（默认head(5)）
print("\n机械臂末端位姿数据预览（前5行）:")
print(df_robot_eef_pose.head())  # 显示列名和前5行数据


# ---------- 第二部分：处理时间戳数据 ----------
# 获取组中第七个数组的键名（索引6）
array_key = list(group.array_keys())[3]  # 对应'timestamp'

# 读取时间戳数组数据
timestamp = group[array_key][:]  # 获取一维时间戳数组

# 转换为DataFrame
df_timestamp = pd.DataFrame(timestamp)

# 设置Pandas显示选项：浮点数显示小数点后10位
pd.set_option('display.float_format', '{:.10f}'.format)

# 打印时间戳数据前5行
print("\n时间戳数据预览（前5行）:")
print(df_timestamp.head())
