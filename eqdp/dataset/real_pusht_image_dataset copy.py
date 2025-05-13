if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
from typing import Dict, List
import torch                # 导入torch模块
import numpy as np          # 导入numpy模块
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import cv2
import json
import hashlib
import copy                 # 导入copy模块
from diffusion_policy.common.pytorch_util import dict_apply     # 导入dict_apply函数
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer  # 导入ReplayBuffer类
from diffusion_policy.common.sampler import (                   # 从sampler模块导入SequenceSampler, get_val_mask, downsample_mask函数
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)

class RealPushTImageDataset(BaseImageDataset):
    # 从指定路径复制数据并生成验证掩码和训练掩码，并初始化序列采样器
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,      # zarr数据路径
            horizon=1,              # 时间跨度
            pad_before=0,           # 前填充步数
            pad_after=0,            # 后填充步数
            n_obs_steps=None,
            n_latency_steps=0,
            use_cache=False,
            seed=42,                # 随机种子
            val_ratio=0.0,          # 验证集比例
            max_train_episodes=None,# 最大训练集数
            delta_action=False      # 动作差分处理
        ):
        # 初始化参数验证
        assert os.path.isdir(dataset_path)
        # 缓存机制（核心亮点）
        replay_buffer = None
        if use_cache:
            # fingerprint shape_meta
            # 生成基于shape_meta的哈希指纹
            shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            # 文件锁避免并发冲突
            with FileLock(cache_lock_path):
                # 缓存不存在时创建
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _get_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore()
                        )
                        print('Saving cache to disk.')
                        # 保存为ZIP格式
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    print('dataset_path ==', dataset_path)
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore()
            )

        ####################################################
        # 动作差分处理 适用于需要速度控制而非绝对位置的控制系统
        if delta_action:
            # replace action as relative to previous frame
            actions = replay_buffer['action'][:]
            # support positions only at this time
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                # it should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                # 计算相邻帧动作差异
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            replay_buffer['action'][:] = actions_diff

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # 数据采样器配置
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,# n_latency_steps是动作延迟补偿步数
            pad_before=pad_before,      # 序列前填充长度（处理边界情况）
            pad_after=pad_after,
            episode_mask=train_mask,    # 设置训练掩码
            key_first_k=key_first_k     # 对图像观测仅取前k帧
            )
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon          # 设置时间跨度
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before    # 设置前填充步数
        self.pad_after = pad_after      # 设置后填充步数

    # 复制当前实例，并创建一个新的序列采样器，使用验证掩码来生成验证数据集
    def get_validation_dataset(self):           # 获取验证数据集的方法
        val_set = copy.copy(self)               # 复制当前实例
        val_set.sampler = SequenceSampler(      # 创建新的SequenceSampler实例用于验证集
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask   # 设置验证集的训练掩码
        return val_set                      # 返回验证数据集

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        # if 'action' not in available_keys:
        #     raise KeyError("数据集缺少必要的 'action' 字段")
        # if 'camero_0' not in available_keys:
        #     raise KeyError("数据集缺少必要的 'camero_0' 字段")
        # ✅ 动态获取可用字段
        available_keys = list(self.replay_buffer.keys())
        print(f"可用字段: {available_keys}")

        # ✅ Action 归一化
        if 'action' in available_keys:
            normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer['action'])
        else:
            raise KeyError("数据集缺少必要的 'action' 字段")
        
        # # ✅ Gripper 归一化（可选字段）
        # if 'gripper' in available_keys:
        #     normalizer['gripper'] = SingleFieldLinearNormalizer.create_fit(
        #         self.replay_buffer['gripper'])
        # else:
        #     print("⚠️ 警告: 数据集中缺少 'gripper' 字段")
        
        # ✅ Low-dim 观测归一化 统一使用 SingleFieldLinearNormalizer
        # required_lowdim = ['robot_eef_pose', 'stage', 'robot_gripper']
        # for key in required_lowdim:
        #     if key in available_keys:
        #         normalizer[key] = SingleFieldLinearNormalizer.create_fit(
        #             self.replay_buffer[key])
        #     else:
        #         print(f"⚠️ 警告: 缺失 low-dim 字段 {key}")
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])
        
        # ✅ 图像归一化（固定范围归一化 [0,255] -> [0,1]）
        for key in self.rgb_keys:
            if key in available_keys:
                normalizer[key] = get_image_range_normalizer()
            else:
                print(f"⚠️ 警告: 缺失图像字段 {key}")
        
        return normalizer

    # 从重放缓冲区获取动作并转换为 torch.Tensor
    def get_all_actions(self) -> torch.Tensor:          # 获取所有动作的方法
        return torch.from_numpy(self.replay_buffer['action'])
        # return torch.from_numpy(self.replay_buffer[self.action_key])  # 从重放缓冲区获取动作并转换为Tensor

    # 返回采样器的长度
    def __len__(self) -> int:               # 获取数据集长度的方法
        return len(self.sampler)            # 返回采样器的长度

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        # 数据加载（getitem）
        """

        # 线程池限制（提升稳定性）
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # # 验证关键字段形状
        # print("data['robot_eef_pose'].shape:", data['robot_eef_pose'].shape)
        # print("data['stage'].shape:", data['stage'].shape)
        # print("data['robot_gripper'].shape:", data['robot_gripper'].shape)
        # assert data['robot_eef_pose'].shape[-1] == 6, "末端执行器位姿维度错误"
        # assert data['stage'].shape[-1] == 1, "阶段指示器维度错误"
        # assert data['robot_gripper'].shape[-1] == 1, "夹爪状态维度错误"

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            # 通道顺序调整，数据类型转换
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            # save ram
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            # save ram
            del data[key]
        
        action = data['action'].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        # 延迟补偿
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }
        return torch_data

def zarr_resize_index_last_dim(zarr_arr, idxs):
    actions = zarr_arr[:]
    actions = actions[...,idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = actions
    return zarr_arr

def _get_replay_buffer(dataset_path, shape_meta, store):
    """
    # 读取真实数据集"""
    # parse shape meta
    rgb_keys = list()
    lowdim_keys = list()
    out_resolutions = dict()
    lowdim_shapes = dict()
    obs_shape_meta = shape_meta['obs']

    # ✅ 打印原始数据文件验证
    print("数据集原始字段:", os.listdir(dataset_path))

    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = tuple(attr.get('shape'))
        if type == 'rgb':
            rgb_keys.append(key)
            c,h,w = shape
            # 图像分辨率转换
            out_resolutions[key] = (w,h)
        elif type == 'low_dim':
            lowdim_keys.append(key)
            lowdim_shapes[key] = tuple(shape)
            if 'pose' in key:
                assert tuple(shape) in [(2,),(6,)]
    
    action_shape = tuple(shape_meta['action']['shape'])
    # 加了夹爪
    assert action_shape in [(2,),(6,),(7,)]
    print("action_shape ==",action_shape)

    # load data
    cv2.setNumThreads(1)
    # ✅ 修改后的数据加载调用
    with threadpool_limits(1):
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions=out_resolutions,
            lowdim_keys=lowdim_keys + ['action'],  # ✅ 显式添加 action , 'gripper'
            image_keys=rgb_keys
        )

    # ✅ 添加加载后验证
    print("加载后的数据字段:", list(replay_buffer.keys()))

    # transform lowdim dimensions
    # 动作空间降维
    # if action_shape == (2,):
    #     # 2D action space, only controls X and Y
    #     zarr_arr = replay_buffer['action']
    #     zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])
    
    # for key, shape in lowdim_shapes.items():
    #     if 'pose' in key and shape == (2,):
    #         # only take X and Y
    #         zarr_arr = replay_buffer[key]
    #         zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])

    return replay_buffer


def test():
    # Hydra配置加载
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    # with hydra.initialize('../diffusion_policy/config'):
    with hydra.initialize('../config'):
        cfg = hydra.compose('real_franka_train_diffusion_unet_real_hybrid_workspace')
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    from matplotlib import pyplot as plt
    normalizer = dataset.get_normalizer()
    nactions = normalizer['action'].normalize(dataset.replay_buffer['action'][:])
    # 动作速度分析
    diff = np.diff(nactions, axis=0)
    dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
    _ = plt.hist(dists, bins=100); plt.title('real action velocity')
    # 动作速度分布直方图：检测异常动作（如超过机械限位的速度值）
if __name__ == "__main__":
    test()
