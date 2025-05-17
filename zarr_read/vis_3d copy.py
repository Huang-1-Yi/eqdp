import zarr
import numpy as np
import matplotlib.pyplot as plt
from visualizer import Visualizer  # 确保你有这个自定义可视化库

# 配置参数
zarr_path = r"/home/robot/dp_0314/data/test_0322/replay_buffer.zarr"
show_episode = 0     # 要显示的episode索引
show_timestep = 0    # 要显示的时间步
camera_index = 0     # 多相机时的相机索引

# ----------------------------------
# 1. 查看Zarr数据结构
# ----------------------------------
def print_zarr_structure(group, prefix=""):
    """递归打印Zarr数据结构"""
    for key in group.keys():
        item = group[key]
        if isinstance(item, zarr.Group):
            print(f"{prefix}Group '{key}':")
            print_zarr_structure(item, prefix + "  ")
        elif isinstance(item, zarr.Array):
            print(f"{prefix}Array '{key}': shape={item.shape}, dtype={item.dtype}")

print("完整Zarr结构:")
root = zarr.open(zarr_path, mode='r')
print_zarr_structure(root)

# ----------------------------------
# 2. 可视化图像数据
# ----------------------------------
def visualize_images(zarr_group, episode_idx=0, timestep=0):
    try:
        # 根据实际数据结构定位图像数据
        episode_group = zarr_group['episodes'][str(episode_idx)]
        
        # 提取图像数据（假设存储格式为[T, N, H, W, C]）
        img_array = episode_group['images'][:]  # 获取整个episode的图像
        
        print(f"\n图像数据维度: {img_array.shape}")
        print("数据解读: [时间步, 相机索引, 高度, 宽度, 通道]")
        
        # 选择指定时间步和相机的图像
        selected_img = img_array[timestep, camera_index]
        
        # 数据预处理
        if selected_img.dtype == np.float32:
            # 假设图像已经是[0,1]范围
            selected_img = np.clip(selected_img * 255, 0, 255).astype(np.uint8)
        elif selected_img.dtype == np.uint16:
            selected_img = (selected_img / 256).astype(np.uint8)
            
        # 可视化
        plt.figure(figsize=(10, 6))
        plt.imshow(selected_img)
        plt.title(f"Episode {episode_idx} | Timestep {timestep} | Camera {camera_index}")
        plt.axis('off')
        plt.show()
        
    except KeyError as e:
        print(f"数据路径错误，未找到: {e}")
    except Exception as e:
        print(f"可视化失败: {str(e)}")

visualize_images(root, show_episode, show_timestep)

# ----------------------------------
# 3. 可视化点云数据
# ----------------------------------
def visualize_pointclouds(zarr_group, episode_idx=0, timestep=0):
    try:
        episode_group = zarr_group['episodes'][str(episode_idx)]
        
        # 提取点云数据（假设存储格式为[T, N_points, 3]）
        pointcloud = episode_group['point_cloud'][timestep]
        
        print(f"\n点云数据维度: {pointcloud.shape}")
        print("数据解读: [点云数量, 坐标(x,y,z)]")
        
        # 数据清洗（去除无效点）
        valid_mask = ~np.isnan(pointcloud).any(axis=1)
        cleaned_pc = pointcloud[valid_mask]
        
        print(f"有效点数: {len(cleaned_pc)}/{len(pointcloud)}")
        
        # # 可视化
        # vis = Visualizer()
        # vis.visualize_pointcloud(
        #     cleaned_pc,
        #     title=f"Episode {episode_idx} | Timestep {timestep}"
        # )
        
    except KeyError as e:
        print(f"数据路径错误，未找到: {e}")
    except Exception as e:
        print(f"点云可视化失败: {str(e)}")

visualize_pointclouds(root, show_episode, show_timestep)