import zarr
import os
import numpy as np
import cv2
# 正确导入方式
from tqdm.auto import tqdm  # 自动选择合适的环境（推荐）

# path = '/home/robot/key/dp_0314_v2/data/test_032317'
path = '/home/robot/dp_0314/data/test/replay_buffer.zarr'
# zarr_arr = zarr.open(f'{path}/replay_buffer.zarr', mode='r')
zarr_arr = zarr.open(f'{path}', mode='r')

'''
 ├── data
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

# print(zarr_arr)
# 列出根组下的所有子组和数组
print(zarr_arr.tree())
# print(zarr.load('{path}/data.zarr'))  # load只能加载数组,不能加载group
# print(os.path.exists(f'{path}/replay_buffer.zarr'))
print(os.path.exists(f'{path}'))

action_data = zarr_arr['data/action'][:]
print("action:", action_data)

timestamp_data = zarr_arr['data/timestamp'][:]%1000
print("timestamp:", timestamp_data)

#============================================= 查看rgb图像部分 ====================================================

# # camera_0_rgb_data = zarr_arr['data/camera_0'][:]
# # print("camera_0_rgb:", camera_0_rgb_data)
# # print("形状:", camera_0_rgb_data.shape)  # 示例输出: (1, 480, 640, 3)
# # print("数据类型:", camera_0_rgb_data.dtype)  # 示例输出: uint8
# # # 方法1：保存为PNG（OpenCV）
# # image = camera_0_rgb_data[110]
# # cv2.imwrite("F:\\Improved-3D-Diffusion-Policy0406\\zarr_read\\sam2rt_image_110.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# # # 方法2：保存为NumPy文件
# # np.save("F:\\Improved-3D-Diffusion-Policy0406\\zarr_read\\sam2rt_image_110.npy", camera_0_rgb_data)
# camera_0_rgb_data = zarr_arr['data/camera_1'][:]
# print("camera_0_rgb:", camera_0_rgb_data)
# print("形状:", camera_0_rgb_data.shape)
# print("数据类型:", camera_0_rgb_data.dtype)

# # 参数配置：选择需要展示的图片索引、每行图片数和调整尺寸
# # selected_indices = [0 : 153]  # 可自定义选择任意数量的索引
# # cols = 3  # 每行显示的图片数
# # target_size = (320, 240)  # 调整图片尺寸为(宽, 高)，设为None保持原尺寸
# # 参数配置
# selected_indices = list(range(0, 400))  # 包含0到153的154张图片
# cols = 14                              # 每行显示14张（154=11行x14列）
# target_size = None              # 调整尺寸为原图的1/5

# # 提取选中的图片
# selected_images = [camera_0_rgb_data[i] for i in selected_indices]

# # 调整图片尺寸
# if target_size is not None:
#     selected_images = [cv2.resize(img, target_size) for img in selected_images]

# # 确定图片尺寸（用于创建空白填充）
# if target_size is not None:
#     img_w, img_h = target_size
# else:
#     img_h, img_w = selected_images[0].shape[:2]

# # 计算网格布局参数
# rows = (len(selected_images) + cols - 1) // cols
# total = rows * cols
# num_missing = total - len(selected_images)

# # 用空白图片填充不足部分
# if num_missing > 0:
#     blank_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
#     selected_images += [blank_image] * num_missing

# # 创建图片网格
# grid_rows = []
# for i in range(rows):
#     start = i * cols
#     end = start + cols
#     row_images = selected_images[start:end]
#     row_grid = cv2.hconcat(row_images)
#     grid_rows.append(row_grid)

# final_grid = cv2.vconcat(grid_rows)

# # 保存结果（转换为BGR格式）
# output_path = "F:\\Improved-3D-Diffusion-Policy0406\\zarr_read\\image_grid_1_400.png"
# cv2.imwrite(output_path, cv2.cvtColor(final_grid, cv2.COLOR_RGB2BGR))
# print(f"图片网格已保存至：{output_path}")

#===============================================================================================================


#============================================= 查看mask图像部分 ====================================================
output_dir = '/home/robot/dp_0314/zarr_read'  # 输出目录

# 配置图像参数
selected_indices = list(range(0, 100))  # 选择前400张图像
cols = 14                              # 每行显示14张
target_size = (640//4, 480//4)         # 缩放到原图的1/4（宽，高）

# 读取四通道图像数据
camera_data = zarr_arr['data/camera_0_pred'][:]  # 根据实际路径修改
print("图像数据形状:", camera_data.shape)
print("数据类型:", camera_data.dtype)

# # 预处理函数：四通道转三通道 + 颜色转换
def preprocess_image(img):
    """将四通道图像转为BGR格式"""
    rgb = img[..., :3]  # 截取前三个通道
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # 转换为OpenCV的BGR格式
# def preprocess_image(img):
#     alpha = img[..., 3]  # 提取Alpha通道
#     return cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)  # 转为伪彩色
# def preprocess_image(img, threshold=127):
#     alpha = img[..., 3]
#     _, binary = cv2.threshold(alpha, threshold, 255, cv2.THRESH_BINARY)
    
#     # 创建红色蒙版
#     mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
#     mask[:, :, 0:2] = 0  # 保留红色通道
    
#     # 叠加原始图像
#     rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
#     return cv2.addWeighted(rgb, 0.7, mask, 0.3, 0)
# def preprocess_image(img):
#     alpha = img[..., 3]
#     # OTSU自动阈值
#     _, binary = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

# 生成图像网格
selected_images = []
for idx in selected_indices:
    img = camera_data[idx]
    processed = preprocess_image(img)
    
    if target_size:
        processed = cv2.resize(processed, target_size)
    
    selected_images.append(processed)

# 补充空白图像使总数能被cols整除
num_missing = (cols - (len(selected_images) % cols)) % cols
blank = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
selected_images += [blank] * num_missing

# 创建图像网格
rows = []
for i in range(0, len(selected_images), cols):
    row = cv2.hconcat(selected_images[i:i+cols])
    rows.append(row)
final_grid = cv2.vconcat(rows)

# 保存结果
output_path = f"{output_dir}/1channel_mask_{len(selected_indices)}images.png"
cv2.imwrite(output_path, final_grid)
print(f"图像网格已保存至：{output_path}")

# 可选：保存单张示例图像
sample_idx = 100
sample_img = preprocess_image(camera_data[sample_idx])
cv2.imwrite(f"{output_dir}/1channel_mask_{sample_idx}.png", sample_img)


# # ===========提取所有Alpha通道数据（假设数据存储在data/rgbm路径下）==============================
# alpha_data = zarr_arr['data/rgbm'][..., 3]  # 提取第4通道（索引3）
# alpha_subset = alpha_data[:5]  # 取前400张

# # 打印数据信息
# print("Alpha通道数据形状:", alpha_data.shape)
# print("数据类型:", alpha_data.dtype)
# print("数值范围:", (np.min(alpha_data), np.max(alpha_data)))

# # ================= 保存为NumPy二进制文件 =================
# npy_path = os.path.join(output_dir, 'alpha_channel.npy')
# np.save(npy_path, alpha_data)
# print(f"\n已保存NumPy二进制文件到: {npy_path}")
# print(f"文件大小: {os.path.getsize(npy_path)/1024**2:.2f} MB")

# # ================= 可选：保存为Zarr格式 =================
# zarr.save(os.path.join(output_dir, 'alpha_channel.zarr'), alpha_data)
# print(f"\n已保存Zarr文件到: {os.path.join(output_dir, 'alpha_channel.zarr')}")

# # ================= 可选：保存为PNG序列 =================
# png_dir = os.path.join(output_dir, 'alpha_pngs')
# os.makedirs(png_dir, exist_ok=True)

# for i in range(alpha_data.shape[0]):
#     # 转换数据范围：0→0（黑），1→255（白）
#     alpha_img = (alpha_data[i] * 255).astype(np.uint8)  # 关键修改
    
#     # # 验证转换结果
#     # if i == 0:  # 打印首张图像信息
#     #     print("首张图像验证:")
#     #     print("转换后数值范围:", (alpha_img.min(), alpha_img.max()))
#     #     print("数据类型:", alpha_img.dtype)
#     #     print("左上角10x10区域示例:\n", alpha_img[:10, :10])
    
#     cv2.imwrite(
#         os.path.join(png_dir, f"alpha_{i:04d}.png"),
#         alpha_img  # 使用转换后的图像数据
#     )

# print(f"\n已保存PNG序列到: {png_dir}")
# print(f"共 {alpha_data.shape[0]} 张PNG图像")

# # ================= 验证数据完整性 =================
# # 加载验证数据
# loaded_npy = np.load(npy_path)
# print("\n数据验证结果:")
# print("形状匹配:", loaded_npy.shape == alpha_data.shape)
# print("数据类型匹配:", loaded_npy.dtype == alpha_data.dtype)
# print("数值一致性:", np.allclose(loaded_npy, alpha_data))


# # 创建输出文件
# output_path = os.path.join(output_dir, 'alpha_top400.txt')

# # 写入文件（带进度条）
# with open(output_path, 'w') as f:
#     # 写入文件头信息
#     f.write(f"Alpha Channel Values for First 400 Images\n")
#     f.write(f"Image Shape: {alpha_subset.shape[1:]} (HxW)\n")
#     f.write(f"Total Images: {alpha_subset.shape[0]}\n")
#     f.write("="*80 + "\n\n")
    
#     # 遍历每张图像
#     for img_idx in tqdm(range(alpha_subset.shape[0]), desc="Saving images"):
#         f.write(f"Image #{img_idx:03d}\n")
#         f.write("-"*40 + "\n")
        
#         # 遍历图像每行
#         for row in alpha_subset[img_idx]:
#             # 将整行转换为字符串（优化内存）
#             line = " ".join([f"{pixel:3d}" for pixel in row.tolist()])
#             f.write(line + "\n")
        
#         f.write("\n\n")  # 图像间空行

# print(f"\n文件已保存至：{output_path}")
# print(f"预计文件大小：{(alpha_subset.nbytes / (1024**3)):.2f} GB (二进制) → 实际文本文件约增大3-5倍")
#===============================================================================================================


meta_data = zarr_arr['meta/episode_ends'][:]
print("episode_ends:", meta_data)

# # 打开文件
# with open('F:\\Improved-3D-Diffusion-Policy0406\\zarr_read\\sam2rt_image_0_data.txt', 'w') as f:
#     for row in camera_0_rgb_data:
#         # 将数组的每一行转换为字符串并写入文件
#         f.write(' '.join(map(str, row)) + '\n')
    
# with open('/home/robot/key/dp_0314_v2/zarr_read/timestamp_data.txt', 'w') as f:
#     for row in timestamp_data:
#         # 将数组的每一行转换为字符串并写入文件
#         f.write(' '.join(map(str, row)) + '\n')