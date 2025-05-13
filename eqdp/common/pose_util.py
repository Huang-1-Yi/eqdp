"""
这些代码实现了三维空间刚体变换（位置+旋转）的多种表示形式之间的转换，以及相关的几何运算。以下是核心功能的分步解析：

---

**1. 基本表示形式互转**（核心功能）
**(1) 位置+旋转 ↔ 齐次矩阵**
```python
pos_rot_to_mat(position, rotation) → 4x4矩阵
mat_to_pos_rot(4x4矩阵) → (位置, 旋转对象)
```
• 输入/输出示例：

  ```python
  pos = [1,2,3]
  rot = st.Rotation.from_euler('xyz', [30,45,60], degrees=True)
  mat = [[R11 R12 R13 1],
         [R21 R22 R23 2],
         [R31 R32 R33 3],
         [0   0   0   1]]
  ```

**(2) 位置+旋转 ↔ 6D姿态向量**
```python
pos_rot_to_pose(position, rotation) → [x,y,z, rx,ry,rz]
pose_to_pos_rot([x,y,z, rx,ry,rz]) → (位置, 旋转对象)
```
• 用途：紧凑表示姿态（如机器人末端位姿）


---

**2. 高级表示形式互转**（面向深度学习）
**(1) 6D旋转表示 ↔ 旋转矩阵**
```python
rot6d_to_mat([a1x,a1y,a1z, a2x,a2y,a2z]) → 3x3旋转矩阵
mat_to_rot6d(3x3矩阵) → 6D向量
```
• 原理：用两个正交向量重建旋转矩阵

• 优势：避免万向节锁，适合神经网络训练


**(2) 10D姿态表示 ↔ 齐次矩阵**
```python
mat_to_pose10d(4x4矩阵) → [x,y,z, a1x,a1y,a1z, a2x,a2y,a2z]
pose10d_to_mat(10D向量) → 4x4矩阵
```
• 结构：位置(3D) + 6D旋转表示

• 用途：适合作为神经网络的输出层


---

**3. 几何变换操作**
**(1) 坐标系变换**
```python
transform_pose(T, pose) → 将姿态从旧坐标系转换到新坐标系
transform_point(T, point) → 转换点的坐标
```
• 数学本质：`T_new_old * pose_old_obj`


**(2) 3D点投影**
```python
project_point(K, point) → [u,v]像素坐标
```
• 参数：`K`为相机内参矩阵

• 计算流程：

  1. `x = K * point`
  2. `u = x[0]/x[2], v = x[1]/x[2]`

---

**4. 实用工具函数**
**(1) 增量姿态更新**
```python
apply_delta_pose(原姿态, 增量姿态) → 新姿态
```
• 位置更新：直接相加

• 旋转更新：旋转矩阵相乘 → 旋转向量


**(2) 方向对齐旋转**
```python
rot_from_directions(起始方向, 目标方向) → 旋转对象
```
• 实现步骤：

  1. 计算两向量的旋转轴（叉乘）
  2. 计算旋转角度（点乘反余弦）
  3. 生成旋转向量表示

---

**关键应用场景**
1. 机器人运动控制  
   • 末端执行器位姿计算

   • 坐标系间的坐标变换


2. 三维视觉处理  
   • 点云数据与图像投影

   • SLAM中的位姿优化


3. 深度学习模型  
   • 6D/10D表示用于姿态估计网络

   • 数据预处理中的几何标准化


---

**性能特点对比**
| 表示形式       | 存储空间 | 计算效率 | 适用场景                 |
|----------------|----------|----------|--------------------------|
| 齐次矩阵       | 16浮点数 | 低       | 矩阵链式变换             |
| 6D姿态向量     | 6浮点数  | 中       | 神经网络训练             |
| 旋转向量       | 3浮点数  | 高       | 物理引擎/优化算法        |
| 10D紧凑表示    | 10浮点数 | 中       | 端到端位姿回归网络       |

---

这些代码构成了一个完整的三维几何运算工具库，特别适合需要处理刚体变换且对表示形式有多样化需求的场景（如同时需要传统几何计算和深度学习接口）。
"""

import numpy as np
import scipy.spatial.transform as st

def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos
    mat[...,:3,:3] = rot.as_matrix()
    mat[...,3,3] = 1
    return mat

def mat_to_pos_rot(mat):
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = st.Rotation.from_matrix(mat[...,:3,:3])
    return pos, rot

def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=pos.dtype)
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose

def pose_to_pos_rot(pose):
    pos = pose[...,:3]
    rot = st.Rotation.from_rotvec(pose[...,3:])
    return pos, rot

def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))

def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))

def transform_pose(tx, pose):
    """
    tx: tx_new_old
    pose: tx_old_obj
    result: tx_new_obj
    """
    pose_mat = pose_to_mat(pose)
    tf_pose_mat = tx @ pose_mat
    tf_pose = mat_to_pose(tf_pose_mat)
    return tf_pose

def transform_point(tx, point):
    return point @ tx[:3,:3].T + tx[:3,3]

def project_point(k, point):
    x = point @ k.T
    uv = x[...,:2] / x[...,[2]]
    return uv

def apply_delta_pose(pose, delta_pose):
    new_pose = np.zeros_like(pose)

    # simple add for position
    new_pose[:3] = pose[:3] + delta_pose[:3]

    # matrix multiplication for rotation
    rot = st.Rotation.from_rotvec(pose[3:])
    drot = st.Rotation.from_rotvec(delta_pose[3:])
    new_pose[3:] = (drot * rot).as_rotvec()

    return new_pose

def normalize(vec, tol=1e-7):
    return vec / np.maximum(np.linalg.norm(vec), tol)

def rot_from_directions(from_vec, to_vec):
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)
    axis = np.cross(from_vec, to_vec)
    axis = normalize(axis)
    angle = np.arccos(np.dot(from_vec, to_vec))
    rotvec = axis * angle
    rot = st.Rotation.from_rotvec(rotvec)
    return rot

def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out

def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out

def mat_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

def mat_to_pose10d(mat):
    pos = mat[...,:3,3]
    rotmat = mat[...,:3,:3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10

def pose10d_to_mat(d10):
    pos = d10[...,:3]
    d6 = d10[...,3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d10.shape[:-1]+(4,4), dtype=d10.dtype)
    out[...,:3,:3] = rotmat
    out[...,:3,3] = pos
    out[...,3,3] = 1
    return out