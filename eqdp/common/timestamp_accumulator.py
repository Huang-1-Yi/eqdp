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