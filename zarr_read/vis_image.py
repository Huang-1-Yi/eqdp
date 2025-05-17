import zarr
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 定义Zarr文件路径
# zarr_path = '/home/robot/key/dp_0314_v2/data/test_032317/replay_buffer.zarr/data'
zarr_path = 'F:\\Improved-3D-Diffusion-Policy0406\\test_data\\replay_buffer.zarr\\data'

# 打开Zarr数据集
root = zarr.open(zarr_path, mode='r')

'''
 ├── data(root)
 │   ├── action (305, 7) float64
 │   ├── action_timestamps (305,) float64
 │   ├── camera_0_rgb (305, 480, 640, 3) uint8
 │   ├── camera_0_rgb_timestamps (305,) float64
 │   ├── camera_1_rgb (305, 480, 640, 3) uint8
 │   ├── camera_1_rgb_timestamps (305,) float64
 │   ├── robot_eef_pose (305, 6) float64
 │   ├── robot_gripper (305, 1) float64
 │   ├── robot_joint (305, 7) float64
 │   ├── robot_joint_vel (305, 7) float64
 │   ├── stage (305, 1) int64
 │   └── timestamp (305,) float64
 └── meta
     └── episode_ends (3,) int64
'''

# ---------------------- 关键修改点 ----------------------
# 修正数据路径：直接从根目录访问img组（非data/img）
# img_array = root['img']  # 键 'img' 对应的数据列表
# img_array = root['camera_0_rgb']  
img_array = root['camera_0']  

# 验证元数据（根据实际.zarray内容调整）
# 假设形状为 (num_samples, height, width, channels)
print("实际数组形状:", img_array.shape)  # 调试输出
assert img_array.ndim == 4, "应为4维数组 (样本, 高, 宽, 通道)"
# assert img_array.dtype == np.uint8, "数据类型应为uint8"

# ------------------------------------------------------
# 方法1：显示单张图像（优化颜色通道处理）
def show_single_image(img_array, index=1):
    """显示指定索引的图像"""
    img = img_array[index]
    
    # 颜色通道处理（根据实际存储格式调整）
    # 如果原始数据是RGB，直接使用；若为BGR则反转通道
    if img.shape[-1] == 3:  # 仅当通道数为3时处理
        img_rgb = img[..., ::-1]  # BGR->RGB（按需启用）
    else:
        img_rgb = img
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title(f"Sample {index}\nShape: {img.shape}")
    plt.axis('off')
    plt.savefig(f'sample_{index}.png')  # 非交互式时保存文件
    plt.close()

# 显示首张图像
# show_single_image(img_array, 0)

# 方法2：图像网格显示（优化批量加载）
def show_image_grid(img_array, num_images=153):
    """显示指定数量的图像网格"""
    grid_size = int(np.ceil(np.sqrt(num_images)))
    plt.figure(figsize=(153, 153))
    
    # 批量读取优化：使用OIndex切片加速
    images = img_array[:num_images]
    
    for i in range(num_images):
        ax = plt.subplot(grid_size, grid_size, i+1)
        img = images[i]
        if img.shape[-1] == 3:
            img = img[..., ::-1]  # 通道转换
        ax.imshow(img)
        ax.set_title(f"Index: {i}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('F:\\Improved-3D-Diffusion-Policy0406\\zarr_read\\sam2rt_image_grid_153.png')  # 保存结果
    plt.close()
    print(f"Saved {num_images} images to image_grid.png")

# 显示前16张图像（4x4网格）
show_image_grid(img_array, 16)

# 方法3：交互式浏览（Jupyter环境专用）
'''
def interactive_browser(img_array):
    """仅限Jupyter Notebook使用"""
    from ipywidgets import interact, IntSlider
    
    @interact(
        index=IntSlider(min=0, 
                       max=len(img_array)-1, 
                       step=1,
                       value=0,
                       description='Index')
    )
    def update(index):
        plt.figure(figsize=(6, 6))
        plt.imshow(img_array[index][..., ::-1])
        plt.axis('off')
        plt.show()
'''