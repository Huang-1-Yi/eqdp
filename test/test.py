

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    print(f"Working directory: {os.getcwd()}")

import pathlib

# 主函数
import os
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import hydra
from omegaconf import OmegaConf
from test.robomimic_replay_point_cloud_dataset import RobomimicReplayPointCloudDataset


def save_first_frame_image_and_point_cloud(dataset, idx, output_dir):
    """
    提取并保存数据集中的第一帧图像和点云数据。
    """
    # 获取样本数据
    data = dataset[idx]

    # 保存所有帧图像
    for t in range(data['obs'][dataset.rgb_keys[0]].shape[0]):
        first_frame_image = data['obs'][dataset.rgb_keys[0]][t]  # 获取当前帧图像，假设为 RGB 图像
        first_frame_image = np.moveaxis(first_frame_image, 0, -1)  # 将通道放在最后，方便保存

        # 保存图像
        image_path = os.path.join(output_dir, f"image_{idx}_frame_{t}.png")
        plt.imsave(image_path, first_frame_image)
        print(f"Saved image for sample {idx}, frame {t} to {image_path}")

    # 保存所有帧点云
    for t in range(data['obs'][dataset.pc_keys[0]].shape[0]):
        first_frame_point_cloud = data['obs'][dataset.pc_keys[0]][t]  # 获取当前帧点云

        # 将点云数据保存为 PLY 格式
        point_cloud_path = os.path.join(output_dir, f"point_cloud_{idx}_frame_{t}.ply")
        save_point_cloud_to_ply(first_frame_point_cloud, point_cloud_path)
        print(f"Saved point cloud for sample {idx}, frame {t} to {point_cloud_path}")


def save_point_cloud_to_ply(point_cloud, file_path):
    """
    将点云数据保存为 PLY 文件。
    """
    # 假设点云是 N x 3 数组 (每个点有 x, y, z 坐标)
    vertices = np.array([(x, y, z) for x, y, z in point_cloud], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    # 创建 PLY 元素
    vertex_element = PlyElement.describe(vertices, 'vertex')
    
    # 保存为 PLY 文件
    ply_data = PlyData([vertex_element])
    ply_data.write(file_path)
    print(f"Point cloud saved to {file_path}")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath(
        'test','config'))
)# 假设 dp3.yaml 配置文件在当前路径的前一个的前一个的equi_diffpo里面
def test(cfg: OmegaConf):
    
    print(f"Loaded config: {OmegaConf.to_yaml(cfg)}")
    print(f"shape_meta['action']['shape']: {cfg.shape_meta['action']['shape']}")
    
    # 创建一个数据集实例
    dataset = RobomimicReplayPointCloudDataset(
        shape_meta=cfg.shape_meta,          # 自动从配置中加载 shape_meta
        dataset_path=cfg.dataset_path,      # 自动从配置中加载 dataset_path
        n_demo=cfg.n_demo,
        n_obs_steps=cfg.n_obs_steps,
        horizon=cfg.horizon,
    )
    
    output_dir = './test/test_output'  # 输出文件夹路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存数据集中的前 10 个样本的第一帧图像和点云
    for idx in range(10):
        save_first_frame_image_and_point_cloud(dataset, idx, output_dir)

if __name__ == "__main__":

    test()
