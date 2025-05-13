"""
这三个类都是用于处理时间序列数据的PyTorch数据集抽象基类，但针对不同类型的数据输入进行了区分。

**1. BaseDataset**
适用场景 :处理多模态混合观测数据（如同时包含图像+传感器数据）

数据结构特征  
obs: Dict[str, torch.Tensor]  # 字典结构，支持多传感器
   |- camera: T, C, H, W      # 图像序列
   |- lidar: T, 360           # 雷达点云
action: T, Da                 # 动作序列

典型用例:机器人系统中同时使用摄像头（图像）和IMU（低维传感器）的场景

**2. BaseLowdimDataset**
适用场景 :处理纯低维观测数据（如关节角度、速度等结构化数据）

数据结构特征  
obs: T, Do      # 单一低维张量（如机械臂的7维关节角度）
action: T, Da   # 动作序列

典型用例:机械臂控制任务中仅使用关节传感器数据的场景

**3. BaseImageDataset**
适用场景 :处理纯图像观测数据（可支持多摄像头）

数据结构特征
obs: Dict[str, torch.Tensor]  # 字典结构，但值必须为图像
   |- front_cam: T, C, H, W
   |- top_cam: T, C, H, W
action: T, Da

典型用例:自动驾驶任务中使用多个摄像头输入的场景


**关键对比表格**
| 特征                | BaseDataset          | BaseLowdimDataset       | BaseImageDataset        |
|--------------------|----------------------|-------------------------|-------------------------|
| 观测数据类型         | 多模态混合             | 单一低维数据              | 多摄像头图像              |
| `obs`结构           | 字典（可含任意类型）    | 单一张量(T, Do)          | 字典（值必须为图像）        |
| 典型输入维度         | 任意（图像+低维混合）   | Do ≤ 100（如7维关节）     | (C, H, W)图像            |
| 常见应用场景         | 多传感器机器人         | 机械臂控制                | 自动驾驶/视觉导航          |

**设计意图解析**
1. BaseLowdimDataset  
   针对结构化低维数据优化（如内存效率更高），观测直接表示为张量，避免字典开销。
   
2. BaseImageDataset  
   强制图像数据的字典结构，便于处理多摄像头输入，预留图像预处理接口。

3. BaseDataset  
   作为通用父类，可扩展性最强，适合需要混合图像、语音、传感器等多模态输入的场景。


**选择建议**
• 如果只有单一种类的低维传感器数据 → 选`BaseLowdimDataset`

• 如果使用多个摄像头或其他图像源 → 选`BaseImageDataset`

• 如果需要混合图像+其他传感器数据 → 选`BaseDataset`

这些基类为不同类型的时间序列数据提供了统一的接口规范，方便后续模型处理不同形态的输入数据。
"""


from typing import Dict

import torch
import torch.nn
from eqdp.model.common.normalizer import LinearNormalizer


class BaseDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseDataset':
        # return an empty dataset by default
        return BaseDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()

class BaseLowdimDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseLowdimDataset':
        # return an empty dataset by default
        return BaseLowdimDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: T, Do
            action: T, Da
        """
        raise NotImplementedError()


class BaseImageDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseImageDataset':
        # return an empty dataset by default
        return BaseImageDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()
