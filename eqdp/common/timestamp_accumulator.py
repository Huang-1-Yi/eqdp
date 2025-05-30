# 版本：1.1 新增image累加器、depth累加器、点云累加器、mask累加器、原始时序累加器ObsAccumulator

"""
添加了 ObsAccumulator
ObsAccumulator 是一个用于按时间顺序累积多源观测数据的工具，其核心作用是为不同传感器或数据源维护独立的时间有序数据序列。以下是关键点解析：

---

**核心功能**
1. 时间有序存储  
   • 每个数据源（如摄像头、雷达）用独立键（key）标识，数据按时间戳严格递增存储。
   • 当新数据的时间戳 `t` 大于该数据源最后记录的时间戳时，才会被添加到列表中，避免旧数据覆盖或乱序。

2. 多源独立管理  
   • 使用 `defaultdict` 管理多源数据，不同传感器数据独立存储，互不影响。
   • 例如：`data["camera"]` 和 `data["lidar"]` 分别存储各自的观测值和时间戳。

**与Timestamp版本的区别**
• TimestampObsAccumulator  
  将数据对齐到固定时间间隔（`dt`），处理掉帧时可能重复数据，适用于强化学习等需要固定频率输入的场景。
• ObsAccumulator  
  更灵活，直接按原始时间戳存储数据，不强制对齐时间网格，适合记录原始观测数据供后续分析或异步处理。

**使用场景示例**
• 机器人数据记录  
  摄像头（30Hz）和雷达（10Hz）以不同频率发送数据，ObsAccumulator 分别记录它们的原始数据和时间戳。
• 异步数据处理  
  后续算法需要按实际发生时间获取传感器数据（如计算延迟或分析时序关系），ObsAccumulator 提供原始时间序列支持。

**代码关键逻辑**
```python
def put(self, data: Dict[str, np.ndarray], timestamps: np.ndarray):
    for key, value in data.items():
        for i, t in enumerate(timestamps):
            if (key not in self.timestamps) or (self.timestamps[key][-1] < t):
                self.timestamps[key].append(t)
                self.data[key].append(value[i])
```
• 逐时间戳检查：遍历每个数据点的时间戳，仅当时间递增时存储。
• 多源隔离：不同 `key` 的数据独立维护，避免交叉影响。

**总结**
ObsAccumulator 是一个轻量级的时间序列数据累积器，专注于按实际发生时间存储多源观测数据，适合需要保留原始时序信息的场景。与基于固定时间窗口的累积器相比，它更灵活，适用于无需强制对齐时间的应用。
"""

from typing import List, Tuple, Optional, Dict
import math
import numpy as np
import collections  # 新增

def get_accumulate_timestamp_idxs(
    timestamps: List[float],  
    start_time: float, 
    dt: float, 
    eps:float=1e-5,
    next_global_idx: Optional[int]=0,
    allow_negative=False
    ) -> Tuple[List[int], List[int], int]:
    """
    For each dt window, choose the first timestamp in the window.
    Assumes timestamps sorted. One timestamp might be chosen multiple times due to dropped frames.
    next_global_idx should start at 0 normally, and then use the returned next_global_idx. 
    However, when overwiting previous values are desired, set last_global_idx to None.

    Returns:
    local_idxs: which index in the given timestamps array to chose from
    global_idxs: the global index of each chosen timestamp
    next_global_idx: used for next call.
    """
    local_idxs = list()
    global_idxs = list()
    for local_idx, ts in enumerate(timestamps):
        # add eps * dt to timestamps so that when ts == start_time + k * dt 
        # is always recorded as kth element (avoiding floating point errors)
        global_idx = math.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats
    return local_idxs, global_idxs, next_global_idx


def align_timestamps(    
        timestamps: List[float], 
        target_global_idxs: List[int], 
        start_time: float, 
        dt: float, 
        eps:float=1e-5):
    if isinstance(target_global_idxs, np.ndarray):
        target_global_idxs = target_global_idxs.tolist()
    assert len(target_global_idxs) > 0

    local_idxs, global_idxs, _ = get_accumulate_timestamp_idxs(
        timestamps=timestamps,
        start_time=start_time,
        dt=dt,
        eps=eps,
        next_global_idx=target_global_idxs[0],
        allow_negative=True
    )
    if len(global_idxs) > len(target_global_idxs):
        # if more steps available, truncate
        global_idxs = global_idxs[:len(target_global_idxs)]
        local_idxs = local_idxs[:len(target_global_idxs)]
    
    if len(global_idxs) == 0:
        import pdb; pdb.set_trace()

    for i in range(len(target_global_idxs) - len(global_idxs)):
        # if missing, repeat
        local_idxs.append(len(timestamps)-1)
        global_idxs.append(global_idxs[-1] + 1)
    assert global_idxs == target_global_idxs
    assert len(local_idxs) == len(global_idxs)
    return local_idxs


class TimestampObsAccumulator:
    def __init__(self, 
            start_time: float, 
            dt: float, 
            eps: float=1e-5):
        self.start_time = start_time
        self.dt = dt
        self.eps = eps
        self.obs_buffer = dict()
        self.timestamp_buffer = None
        self.next_global_idx = 0
    
    def __len__(self):
        return self.next_global_idx
    
    @property
    def data(self):
        if self.timestamp_buffer is None:
            return dict()
        result = dict()
        for key, value in self.obs_buffer.items():
            result[key] = value[:len(self)]
        return result

    @property
    def actual_timestamps(self):
        if self.timestamp_buffer is None:
            return np.array([])
        return self.timestamp_buffer[:len(self)]
    
    @property
    def timestamps(self):
        if self.timestamp_buffer is None:
            return np.array([])
        return self.start_time + np.arange(len(self)) * self.dt

    def put(self, data: Dict[str, np.ndarray], timestamps: np.ndarray):
        """
        data:
            key: T,*
        """

        local_idxs, global_idxs, self.next_global_idx = get_accumulate_timestamp_idxs(
            timestamps=timestamps,
            start_time=self.start_time,
            dt=self.dt,
            eps=self.eps,
            next_global_idx=self.next_global_idx
        )

        if len(global_idxs) > 0:
            if self.timestamp_buffer is None:
                # first allocation
                self.obs_buffer = dict()
                for key, value in data.items():
                    self.obs_buffer[key] = np.zeros_like(value)
                self.timestamp_buffer = np.zeros(
                    (len(timestamps),), dtype=np.float64)
            
            this_max_size = global_idxs[-1] + 1
            if this_max_size > len(self.timestamp_buffer):
                # reallocate
                new_size = max(this_max_size, len(self.timestamp_buffer) * 2)
                for key in list(self.obs_buffer.keys()):
                    new_shape = (new_size,) + self.obs_buffer[key].shape[1:]
                    self.obs_buffer[key] = np.resize(self.obs_buffer[key], new_shape)
                self.timestamp_buffer = np.resize(self.timestamp_buffer, (new_size))
            
            # write data
            for key, value in self.obs_buffer.items():
                value[global_idxs] = data[key][local_idxs]
            self.timestamp_buffer[global_idxs] = timestamps[local_idxs]


class TimestampActionAccumulator:
    def __init__(self, 
            start_time: float, 
            dt: float, 
            eps: float=1e-5):
        """
        Different from Obs accumulator, the action accumulator
        allows overwriting previous values.
        """
        self.start_time = start_time
        self.dt = dt
        self.eps = eps
        self.action_buffer = None
        self.timestamp_buffer = None
        self.size = 0
    
    def __len__(self):
        return self.size
    
    @property
    def actions(self):
        if self.action_buffer is None:
            return np.array([])
        return self.action_buffer[:len(self)]
    
    @property
    def actual_timestamps(self):
        if self.timestamp_buffer is None:
            return np.array([])
        return self.timestamp_buffer[:len(self)]
    
    @property
    def timestamps(self):
        if self.timestamp_buffer is None:
            return np.array([])
        return self.start_time + np.arange(len(self)) * self.dt

    def put(self, actions: np.ndarray, timestamps: np.ndarray):
        """
        Note: timestamps is the time when the action will be issued, 
        not when the action will be completed (target_timestamp)
        """

        local_idxs, global_idxs, _ = get_accumulate_timestamp_idxs(
            timestamps=timestamps,
            start_time=self.start_time,
            dt=self.dt,
            eps=self.eps,
            # allows overwriting previous actions
            next_global_idx=None
        )

        if len(global_idxs) > 0:
            if self.timestamp_buffer is None:
                # first allocation
                self.action_buffer = np.zeros_like(actions)
                self.timestamp_buffer = np.zeros((len(actions),), dtype=np.float64)

            this_max_size = global_idxs[-1] + 1
            if this_max_size > len(self.timestamp_buffer):
                # reallocate
                new_size = max(this_max_size, len(self.timestamp_buffer) * 2)
                new_shape = (new_size,) + self.action_buffer.shape[1:]
                self.action_buffer = np.resize(self.action_buffer, new_shape)
                self.timestamp_buffer = np.resize(self.timestamp_buffer, (new_size,))
            
            # potentially rewrite old data (as expected)
            self.action_buffer[global_idxs] = actions[local_idxs]
            self.timestamp_buffer[global_idxs] = timestamps[local_idxs]
            self.size = max(self.size, this_max_size)

# 图像的时间戳累加器
class TimestampImageAccumulator:
    def __init__(self, 
                 start_time: float,                                      # 开始时间
                 dt: float,                                              # 时间间隔
                 eps: float = 1e-5,                                      # 浮点数精度
                 image_shape: Optional[Tuple[int, int, int]] = None):    # 图像形状
        """
        Accumulator for timestamped image data.
        
        Args:
            start_time: Initial time reference
            dt: Time interval between frames
            eps: Small epsilon value to handle floating point precision
            image_shape: Optional expected shape of images (H, W, C). 
                        If not provided, will be inferred from first input.
        """
        self.start_time = start_time         # 开始时间
        self.dt = dt                         # 时间间隔
        self.eps = eps                       # 浮点数精度
        self.image_buffer = None             # 图像缓存
        self.timestamp_buffer = None         # 时间戳缓存
        self.next_global_idx = 0             # 下一个全局索引
        self.image_shape = image_shape       # 图像形状
    
    def __len__(self):
        """Returns number of accumulated frames."""
        return self.next_global_idx

    @property
    def images(self):
        """Returns accumulated images up to current length."""
        return self.image_buffer[:self.next_global_idx] if self.image_buffer is not None else np.array([])

    @property
    def actual_timestamps(self):
        """Returns actual timestamps of accumulated frames."""
        return self.timestamp_buffer[:self.next_global_idx] if self.timestamp_buffer is not None else np.array([])

    @property
    def timestamps(self):
        """Returns ideal timestamps based on start_time and dt."""
        return self.start_time + np.arange(self.next_global_idx) * self.dt

    def put(self, images: np.ndarray, timestamps: np.ndarray):
        """Accumulate new images with corresponding timestamps."""
        # 获取索引信息
        local_idxs, global_idxs, self.next_global_idx = get_accumulate_timestamp_idxs(
            timestamps=timestamps,
            start_time=self.start_time,
            dt=self.dt,
            eps=self.eps,
            next_global_idx=self.next_global_idx
        )

        if len(global_idxs) > 0:  # 确保有数据
            # 初始化缓存
            if self.image_buffer is None:
                if self.image_shape is None:
                    self.image_shape = images.shape[1:]
                elif images.shape[1:] != self.image_shape:
                    raise ValueError(f"Image shape {images.shape[1:]} doesn't match expected {self.image_shape}")
                
                self.image_buffer = np.empty((len(images), *self.image_shape), dtype=images.dtype)
                self.timestamp_buffer = np.empty((len(timestamps),), dtype=np.float64)

            # 计算新尺寸（扩容时使用 1.5 倍策略）
            this_max_size = global_idxs[-1] + 1
            if this_max_size > len(self.timestamp_buffer):
                new_size = max(this_max_size, int(len(self.timestamp_buffer) * 1.5))

                # 扩容 image_buffer
                new_shape = (new_size,) + self.image_shape
                temp_image_buffer = np.empty(new_shape, dtype=self.image_buffer.dtype)
                temp_image_buffer[:len(self.image_buffer)] = self.image_buffer
                self.image_buffer = temp_image_buffer

                # 扩容 timestamp_buffer
                temp_timestamp_buffer = np.empty((new_size,), dtype=np.float64)
                temp_timestamp_buffer[:len(self.timestamp_buffer)] = self.timestamp_buffer
                self.timestamp_buffer = temp_timestamp_buffer

            # 写入数据
            self.image_buffer[global_idxs] = images[local_idxs]
            self.timestamp_buffer[global_idxs] = timestamps[local_idxs]

    def clear(self):
            """Clear all accumulated data and reset the accumulator."""
            self.image_buffer = None
            self.timestamp_buffer = None
            self.next_global_idx = 0

# 图像的深度时间戳累加器
class TimestampDepthAccumulator:
    def __init__(self, 
                 start_time: float,
                 dt: float,
                 eps: float = 1e-5,
                 depth_shape: Optional[Tuple[int, int]] = None,
                 depth_dtype: type = np.float32):
        """
        用于深度图像数据的时间戳累加器
        
        参数:
            start_time: 初始时间参考点
            dt: 帧之间的时间间隔(秒)
            eps: 处理浮点数精度的小量
            depth_shape: 可选的深度图像形状 (H, W)
            depth_dtype: 深度数据的numpy数据类型 (默认np.float32)
        """
        self.start_time = start_time
        self.dt = dt
        self.eps = eps
        self.depth_buffer = None  # 深度数据缓存
        self.timestamp_buffer = None  # 时间戳缓存
        self.next_global_idx = 0  # 下一个全局索引
        self.depth_shape = depth_shape  # 深度图像形状
        self.depth_dtype = depth_dtype  # 深度数据类型

    def __len__(self):
        """返回已累积的帧数"""
        return self.next_global_idx

    @property
    def depths(self):
        """返回累积的深度图像数据(到当前长度)"""
        return self.depth_buffer[:self.next_global_idx] if self.depth_buffer is not None else np.array([])

    @property
    def actual_timestamps(self):
        """返回实际采集到的时间戳"""
        return self.timestamp_buffer[:self.next_global_idx] if self.timestamp_buffer is not None else np.array([])

    @property
    def timestamps(self):
        """返回基于start_time和dt的理想时间戳序列"""
        return self.start_time + np.arange(self.next_global_idx) * self.dt

    def put(self, depths: np.ndarray, timestamps: np.ndarray):
        """
        累积新的深度图像数据及其对应时间戳
        
        参数:
            depths: 深度图像数组 (N, H, W)
            timestamps: 对应的时间戳数组 (N,)
        """
        # 验证输入
        if len(depths) != len(timestamps):
            raise ValueError("depths和timestamps长度必须相同")
        
        if depths.ndim != 3:
            raise ValueError("depths必须是3维数组 (N, H, W)")

        # 获取索引信息
        local_idxs, global_idxs, self.next_global_idx = get_accumulate_timestamp_idxs(
            timestamps=timestamps,
            start_time=self.start_time,
            dt=self.dt,
            eps=self.eps,
            next_global_idx=self.next_global_idx
        )

        if len(global_idxs) > 0:  # 确保有数据
            # 初始化缓存
            if self.depth_buffer is None:
                if self.depth_shape is None:
                    self.depth_shape = depths.shape[1:]
                elif depths.shape[1:] != self.depth_shape:
                    raise ValueError(f"深度图像形状 {depths.shape[1:]} 与预期 {self.depth_shape} 不匹配")
                
                # 初始化缓冲区
                self.depth_buffer = np.empty((len(depths), *self.depth_shape), dtype=self.depth_dtype)
                self.timestamp_buffer = np.empty((len(timestamps),), dtype=np.float64)

            # 计算新尺寸 (使用1.5倍扩容策略)
            this_max_size = global_idxs[-1] + 1
            if this_max_size > len(self.timestamp_buffer):
                new_size = max(this_max_size, int(len(self.timestamp_buffer) * 1.5))

                # 扩容深度缓冲区
                new_shape = (new_size,) + self.depth_shape
                temp_depth_buffer = np.empty(new_shape, dtype=self.depth_dtype)
                if self.depth_buffer is not None:
                    temp_depth_buffer[:len(self.depth_buffer)] = self.depth_buffer
                self.depth_buffer = temp_depth_buffer

                # 扩容时间戳缓冲区
                temp_timestamp_buffer = np.empty((new_size,), dtype=np.float64)
                if self.timestamp_buffer is not None:
                    temp_timestamp_buffer[:len(self.timestamp_buffer)] = self.timestamp_buffer
                self.timestamp_buffer = temp_timestamp_buffer

            # 写入数据
            self.depth_buffer[global_idxs] = depths[local_idxs]
            self.timestamp_buffer[global_idxs] = timestamps[local_idxs]

    def clear(self):
        """清除所有累积数据并重置累加器"""
        self.depth_buffer = None
        self.timestamp_buffer = None
        self.next_global_idx = 0

# 新增点云累加器
class TimestampPointCloudAccumulator:
    def __init__(self, 
                 start_time: float,
                 dt: float,
                 eps: float = 1e-5,
                 point_shape: Optional[Tuple[int]] = None,
                 point_dtype: type = np.float32):
        """
        用于点云数据的时间戳累加器
        
        参数:
            start_time: 初始时间参考点
            dt: 帧之间的时间间隔(秒)
            eps: 处理浮点数精度的小量
            point_shape: 可选的点云数据形状 (point_num, 6)
            point_dtype: 点云数据的numpy数据类型 (默认np.float32)，每个点包含xyz和rgb
        """
        self.start_time = start_time
        self.dt = dt
        self.eps = eps
        self.point_buffer = None  # 点云数据缓存 (每帧点云数据：n, 6)
        self.timestamp_buffer = None  # 时间戳缓存
        self.next_global_idx = 0  # 下一个全局索引
        self.point_shape = point_shape  # 每帧点云数据的形状 (点数量, 6)
        self.point_dtype = point_dtype  # 点云数据类型

    def __len__(self):
        """返回已累积的帧数"""
        return self.next_global_idx

    @property
    def point_clouds(self):
        """返回累积的点云数据 (到当前长度)"""
        return self.point_buffer[:self.next_global_idx] if self.point_buffer is not None else np.array([])

    @property
    def actual_timestamps(self):
        """返回实际采集到的时间戳"""
        return self.timestamp_buffer[:self.next_global_idx] if self.timestamp_buffer is not None else np.array([])

    @property
    def timestamps(self):
        """返回基于start_time和dt的理想时间戳序列"""
        return self.start_time + np.arange(self.next_global_idx) * self.dt

    def put(self, point_clouds: np.ndarray, timestamps: np.ndarray):
        """
        累积新的点云数据及其对应时间戳
        
        参数:
            point_clouds: 点云数组 (N, P, 6)，其中 P 为每帧的点数量，6 为 [x, y, z, r, g, b]
            timestamps: 对应的时间戳数组 (N,)
        """
        # 验证输入
        if len(point_clouds) != len(timestamps):
            raise ValueError("point_clouds和timestamps长度必须相同")
        
        if point_clouds.ndim != 3 or point_clouds.shape[2] != 6:
            raise ValueError("point_clouds必须是3维数组 (N, P, 6)，其中 P 为每帧的点数量，6 为 [x, y, z, r, g, b]")
        
        # 获取索引信息
        local_idxs, global_idxs, self.next_global_idx = get_accumulate_timestamp_idxs(
            timestamps=timestamps,
            start_time=self.start_time,
            dt=self.dt,
            eps=self.eps,
            next_global_idx=self.next_global_idx
        )

        if len(global_idxs) > 0:  # 确保有数据
            # 初始化缓存
            if self.point_buffer is None:
                # 初始化每帧点云的形状
                if self.point_shape is None:
                    self.point_shape = point_clouds.shape[1:]  # (P, 6)
                elif point_clouds.shape[1:] != self.point_shape:
                    raise ValueError(f"点云数据形状 {point_clouds.shape[1:]} 与预期 {self.point_shape} 不匹配")
                
                # 初始化缓冲区
                self.point_buffer = np.empty((len(point_clouds), *self.point_shape), dtype=self.point_dtype)
                self.timestamp_buffer = np.empty((len(timestamps),), dtype=np.float64)

            # 计算新尺寸 (使用1.5倍扩容策略)
            this_max_size = global_idxs[-1] + 1
            if this_max_size > len(self.timestamp_buffer):
                new_size = max(this_max_size, int(len(self.timestamp_buffer) * 1.5))

                # 扩容点云缓冲区
                new_shape = (new_size,) + self.point_shape
                temp_point_buffer = np.empty(new_shape, dtype=self.point_dtype)
                if self.point_buffer is not None:
                    temp_point_buffer[:len(self.point_buffer)] = self.point_buffer
                self.point_buffer = temp_point_buffer

                # 扩容时间戳缓冲区
                temp_timestamp_buffer = np.empty((new_size,), dtype=np.float64)
                if self.timestamp_buffer is not None:
                    temp_timestamp_buffer[:len(self.timestamp_buffer)] = self.timestamp_buffer
                self.timestamp_buffer = temp_timestamp_buffer

            # 写入数据
            self.point_buffer[global_idxs] = point_clouds[local_idxs]
            self.timestamp_buffer[global_idxs] = timestamps[local_idxs]

    def clear(self):
        """清除所有累积数据并重置累加器"""
        self.point_buffer = None
        self.timestamp_buffer = None
        self.next_global_idx = 0

# 新增掩膜累加器
class TimestampMaskAccumulator:
    def __init__(self, 
                 start_time: float,
                 dt: float,
                 eps: float = 1e-5,
                 mask_shape: Optional[Tuple[int, int]] = None,
                 mask_dtype: type = np.uint8):
        """
        用于掩膜数据的时间戳累加器
        
        参数:
            start_time: 初始时间参考点
            dt: 帧之间的时间间隔(秒)
            eps: 处理浮点数精度的小量
            mask_shape: 可选的掩膜形状 (H, W)
            mask_dtype: 掩膜数据的numpy数据类型 (默认np.uint8)
        """
        self.start_time = start_time
        self.dt = dt
        self.eps = eps
        self.mask_buffer = None  # 掩膜数据缓存
        self.timestamp_buffer = None  # 时间戳缓存
        self.next_global_idx = 0  # 下一个全局索引
        self.mask_shape = mask_shape  # 掩膜形状
        self.mask_dtype = mask_dtype  # 数据类型

    def __len__(self):
        """返回已累积的帧数"""
        return self.next_global_idx

    @property
    def masks(self):
        """返回累积的掩膜数据(到当前长度)"""
        return self.mask_buffer[:self.next_global_idx] if self.mask_buffer is not None else np.array([], dtype=self.mask_dtype)

    @property
    def actual_timestamps(self):
        """返回实际采集到的时间戳"""
        return self.timestamp_buffer[:self.next_global_idx] if self.timestamp_buffer is not None else np.array([])

    @property
    def timestamps(self):
        """返回基于start_time和dt的理想时间戳序列"""
        return self.start_time + np.arange(self.next_global_idx) * self.dt

    def put(self, masks: np.ndarray, timestamps: np.ndarray):
        """
        累积新的掩膜数据及其对应时间戳
        
        参数:
            masks: 掩膜数组 (N, H, W)
            timestamps: 对应的时间戳数组 (N,)
        """
        # 验证输入
        if len(masks) != len(timestamps):
            raise ValueError("masks和timestamps长度必须相同")
        
        if masks.ndim != 3:
            raise ValueError("masks必须是3维数组 (N, H, W)")

        # 获取索引信息
        local_idxs, global_idxs, self.next_global_idx = get_accumulate_timestamp_idxs(
            timestamps=timestamps.tolist(),
            start_time=self.start_time,
            dt=self.dt,
            eps=self.eps,
            next_global_idx=self.next_global_idx
        )

        if len(global_idxs) > 0:  # 确保有数据
            # 初始化缓存
            if self.mask_buffer is None:
                if self.mask_shape is None:
                    self.mask_shape = masks.shape[1:]
                elif masks.shape[1:] != self.mask_shape:
                    raise ValueError(f"掩膜形状 {masks.shape[1:]} 与预期 {self.mask_shape} 不匹配")
                
                # 初始化缓冲区
                self.mask_buffer = np.empty((len(masks), *self.mask_shape), dtype=self.mask_dtype)
                self.timestamp_buffer = np.empty((len(timestamps),), dtype=np.float64)

            # 计算新尺寸 (使用1.5倍扩容策略)
            this_max_size = global_idxs[-1] + 1
            if this_max_size > len(self.timestamp_buffer):
                new_size = max(this_max_size, int(len(self.timestamp_buffer) * 1.5))

                # 扩容mask缓冲区
                new_shape = (new_size,) + self.mask_shape
                temp_mask_buffer = np.empty(new_shape, dtype=self.mask_dtype)
                if self.mask_buffer is not None:
                    temp_mask_buffer[:len(self.mask_buffer)] = self.mask_buffer
                self.mask_buffer = temp_mask_buffer

                # 扩容时间戳缓冲区
                temp_timestamp_buffer = np.empty((new_size,), dtype=np.float64)
                if self.timestamp_buffer is not None:
                    temp_timestamp_buffer[:len(self.timestamp_buffer)] = self.timestamp_buffer
                self.timestamp_buffer = temp_timestamp_buffer

            # 写入数据
            self.mask_buffer[global_idxs] = masks[local_idxs]
            self.timestamp_buffer[global_idxs] = timestamps[local_idxs]

    def clear(self):
        """清除所有累积数据并重置累加器"""
        self.mask_buffer = None
        self.timestamp_buffer = None
        self.next_global_idx = 0


class ObsAccumulator:
    def __init__(self):
        self.data = collections.defaultdict(list)
        self.timestamps = collections.defaultdict(list)
    
    def put(self, data: Dict[str, np.ndarray], timestamps: np.ndarray):
        """
        data:
            key: T,*
        """
        for key, value in data.items():
            for i, t in enumerate(timestamps):
                if (key not in self.timestamps) or (self.timestamps[key][-1] < t):
                    self.timestamps[key].append(t)
                    self.data[key].append(value[i])